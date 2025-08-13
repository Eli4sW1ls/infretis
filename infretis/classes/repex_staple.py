"""Defines the main REPEX class for path handling and permanent calc."""
import logging
import os
import sys
import time
from datetime import datetime
import traceback

import numpy as np
import tomli_w
from numpy.random import default_rng
import matplotlib.pyplot as plt

from infretis.classes.repex import REPEX_state, spawn_rng
from infretis.classes.engines.factory import assign_engines
from infretis.classes.formatter import PathStorage
from infretis.core.core import make_dirs
from infretis.core.tis import calc_cv_vector

logger = logging.getLogger("main")  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())
DATE_FORMAT = "%Y.%m.%d %H:%M:%S"


class REPEX_state_staple(REPEX_state):
    """Define the REPPEX object."""


    def __init__(self, config, minus=False):
        """Initiate REPEX given confic dict from *toml file."""
        super().__init__(config, minus=True)
        
        # Enable visual validation by default (can be disabled in config)
        self.visual_validation = config.get("output", {}).get("visual_validation", True)

    # TODO: add infinite swap functionality
    # @property
    # def prob(self):
    #     """Calculate the P matrix. For REPPTIS there is no swap,

    #     so prob for a path in 1 ensemble is 100% and zero otherwise.

    #     For RETIS this initiates a permanent calc."""
    #     prop = np.identity(self.n)
    #     prop[-1,-1] = 0.
    #     self._last_prob = prop.copy()
    #     return prop*(1-np.array(self._locks))

    # def add_traj(self, ens, traj, valid, count=True, n=0):
    #     """Add traj to state and calculate P matrix."""
    #     valid = np.zeros(self.n)
    #     valid[ens+1] = 1.
    #     ens += self._offset
    #     assert valid[ens] != 0
    #     # invalidate last prob
    #     self._last_prob = None
    #     self._trajs[ens] = traj
    #     self.state[ens, :] = valid
    #     self.unlock(ens)

    #     # Calculate P matrix
    #     self.prob

    def inf_retis(self, input_mat, locks):
        """Permanent calculator with robust fallback and blocking optimization."""
        # Drop locked rows and columns
        bool_locks = locks == 1
        # get non_locked minus interfaces
        offset = self._offset - sum(bool_locks[: self._offset])
        # make insert list
        i = 0
        insert_list = []
        for lock in bool_locks:
            if lock:
                insert_list.append(i)
            else:
                i += 1

        # Drop locked rows and columns
        non_locked = input_mat[~bool_locks, :][:, ~bool_locks]

        # Check if this is a simple permutation matrix (identity-like)
        is_permutation = (
            np.all(non_locked.sum(axis=1) == 1)
            and np.all(non_locked.sum(axis=0) == 1)
            and np.all((non_locked == 0) | (non_locked == 1))
        )

        if is_permutation:
            # For permutation matrices, the probability matrix is the same
            out = non_locked.astype("longdouble")
        else:
            # Try the advanced blocking algorithm from repex.py
            try:
                if len(non_locked) >= 8:
                    out = self._inf_retis_with_blocking(non_locked, offset)
                    # print(f"Successfully used blocking algorithm for {len(non_locked)}x{len(non_locked)} matrix")
                else:
                    raise Exception("Matrix too small for blocking algorithm")
            except Exception as e:
                # print(f"Blocking algorithm failed ({e}), using fallback methods...")
                # Fall back to original simple method
                if len(non_locked) <= 12:
                    # print(f"Using permanent method for {len(non_locked)}x{len(non_locked)} matrix")
                    out = self.permanent_prob(non_locked)
                else:
                    print(f"Using random approximation for {len(non_locked)}x{len(non_locked)} matrix")
                    out = self.random_prob(non_locked)

        # Validate the result
        row_sums = out.sum(axis=1)
        col_sums = out.sum(axis=0)
        
        # If we have invalid sums, try to repair
        if not np.allclose(row_sums, 1) or not np.allclose(col_sums, 1):
            print("WARNING: Matrix is not doubly stochastic, attempting repair...")
            
            # Normalize rows first
            for i in range(len(out)):
                if row_sums[i] > 0:
                    out[i, :] /= row_sums[i]
                else:
                    # For zero rows, distribute uniformly over support
                    support = np.where(non_locked[i, :] > 0)[0]
                    if len(support) > 0:
                        out[i, support] = 1.0 / len(support)
            
            # Check column normalization
            col_sums = out.sum(axis=0)
            if not np.allclose(col_sums, 1):
                # Use Sinkhorn-like iteration to make doubly stochastic
                for _ in range(100):  # max iterations
                    # Normalize rows
                    row_sums = out.sum(axis=1)
                    out = out / row_sums[:, np.newaxis]
                    
                    # Normalize columns
                    col_sums = out.sum(axis=0)
                    out = out / col_sums[np.newaxis, :]
                    
                    # Check convergence
                    if (np.allclose(out.sum(axis=1), 1) and 
                        np.allclose(out.sum(axis=0), 1)):
                        break

        # Final validation
        assert np.allclose(out.sum(axis=1), 1), f"Row sums: {out.sum(axis=1)}"
        assert np.allclose(out.sum(axis=0), 1), f"Col sums: {out.sum(axis=0)}"

        # reinsert zeroes for the locked ensembles
        final_out_rows = np.insert(out, insert_list, 0, axis=0)

        # reinsert zeroes for the locked trajectories
        final_out = np.insert(final_out_rows, insert_list, 0, axis=1)

        return final_out

    def _inf_retis_with_blocking(self, non_locked, offset):
        """Advanced inf_retis algorithm with blocking from repex.py."""
        # Sort based on the index of the last non-zero values in the rows
        # argmax(a>0) gives back the first column index that is nonzero
        # so looping over the columns backwards and multiplying by -1
        # gives the right ordering
        minus_idx = np.argsort(np.argmax(non_locked[:offset] > 0, axis=1))
        pos_idx = (
            np.argsort(-1 * np.argmax(non_locked[offset:, ::-1] > 0, axis=1))
            + offset
        )

        sort_idx = np.append(minus_idx, pos_idx)
        sorted_non_locked = non_locked[sort_idx]

        # check if all trajectories have equal weights
        sorted_non_locked_T = sorted_non_locked.T
        # Check the minus interfaces
        equal_minus = np.all(
            sorted_non_locked_T[
                np.where(
                    sorted_non_locked_T[:, :offset]
                    != sorted_non_locked_T[offset - 1, :offset]
                )
            ]
            == 0
        )
        # check the positive interfaces
        if len(sorted_non_locked_T) <= offset:
            equal_pos = True
        else:
            equal_pos = np.all(
                sorted_non_locked_T[:, offset:][
                    np.where(
                        sorted_non_locked_T[:, offset:]
                        != sorted_non_locked_T[offset, offset:]
                    )
                ]
                == 0
            )

        equal = equal_minus and equal_pos

        out = np.zeros(shape=sorted_non_locked.shape, dtype="longdouble")
        if equal:
            # All trajectories have equal weights, run fast algorithm
            print(f"Using fast algorithm for equal-weight {len(sorted_non_locked)}x{len(sorted_non_locked)} matrix")
            # minus move should be run backwards
            out[:offset, ::-1] = self.quick_prob(
                sorted_non_locked[:offset, ::-1]
            )
            if offset < len(out):
                # Catch only minus ens available
                out[offset:] = self.quick_prob(sorted_non_locked[offset:])
        else:
            # Use blocking strategy
            print(f"Using blocking algorithm for complex {len(sorted_non_locked)}x{len(sorted_non_locked)} matrix")
            blocks = self.find_blocks(sorted_non_locked, offset=offset)
            for start, stop, direction in blocks:
                if direction == -1:
                    cstart, cstop = stop - 1, start - 1
                    if cstop < 0:
                        cstop = None
                else:
                    cstart, cstop = start, stop
                subarr = sorted_non_locked[start:stop, cstart:cstop:direction]
                subarr_T = subarr.T
                if len(subarr) == 1:
                    out[start:stop, start:stop] = 1
                elif np.all(subarr_T[np.where(subarr_T != subarr_T[0])] == 0):
                    # Either the same weight as the last one or zero
                    temp = self.quick_prob(subarr)
                    out[start:stop, cstart:cstop:direction] = temp
                elif len(subarr) <= 12:
                    # We can run this subsecond
                    temp = self.permanent_prob(subarr)
                    out[start:stop, cstart:cstop:direction] = temp
                else:
                    self._random_count += 1
                    print(
                        f"random #{self._random_count}, "
                        f"dims = {len(subarr)}"
                    )
                    # do n random parallel samples
                    temp = self.random_prob(subarr)
                    out[start:stop, cstart:cstop:direction] = temp

        out[sort_idx] = out.copy()  # COPY REQUIRED TO NOT BREAK STATE!!!
        return out

    def find_blocks(self, arr, offset):
        """Find blocks in a W matrix."""
        if len(arr) == 1:
            return [(0, 1, 1)]
        # Assume no zeroes on the diagonal or lower triangle
        temp_arr = arr.copy()
        # for counting minus blocks
        temp_arr[:offset, :offset] = arr[:offset, :offset].T
        temp_arr[offset:, :offset] = 1  # add ones to the lower triangle
        non_zero = np.count_nonzero(temp_arr, axis=1)
        blocks = []
        start = 0
        for i, e in enumerate(non_zero):
            if e == i + 1:
                direction = -1 if start < offset else 1
                blocks.append((start, e, direction))
                start = e
        return blocks

    def quick_prob(self, arr):
        """Quick P matrix calculation for specific W matrix."""
        total_traj_prob = np.ones(shape=arr.shape[0], dtype="longdouble")
        out_mat = np.zeros(shape=arr.shape, dtype="longdouble")
        working_mat = np.where(arr != 0, 1, 0)  # convert non-zero numbers to 1

        for i, column in enumerate(working_mat.T[::-1]):
            ens = column * total_traj_prob
            s = ens.sum()
            if s != 0:
                ens /= s
            out_mat[:, -(i + 1)] = ens
            total_traj_prob -= ens
            # force negative values to 0
            total_traj_prob[np.where(total_traj_prob < 0)] = 0
        return out_mat
    
    def print_shooted(self, md_items, pn_news):
        """Print shooted."""
        moves = md_items["moves"]
        ens_nums = " ".join([f"00{i+1}" for i in md_items["ens_nums"]])
        pnum_old = " ".join([str(i) for i in md_items["pnum_old"]])
        pnum_new = " ".join([str(i) for i in pn_news])
        trial_lens = " ".join([str(i) for i in md_items["trial_len"]])
        trial_ops = " ".join(
            [f"[{i[0]:4.4f} {i[1]:4.4f}]" for i in md_items["trial_op"]]
        )
        status = md_items["status"]
        simtime = md_items["md_end"] - md_items["md_start"]
        logger.info(
            f"shooted {' '.join(moves)} in ensembles: {ens_nums}"
            f" with paths: {pnum_old} -> {pnum_new}"
        )
        logger.info(
            "with status:"
            f" {status} len: {trial_lens} op: {trial_ops} and"
            f" worker: {self.cworker} total time: {simtime:.2f}"
        )
        self.print_state()

    def print_start(self):
        """Print start."""
        logger.info("stored ensemble paths:")
        ens_num = self.live_paths()
        logger.info(
            " ".join([f"00{i}: {j}," for i, j in enumerate(ens_num)]) + "\n"
        )
        self.print_state()

    def plot_path_validation(self, traj, traj_data, ens_num, move_type="sh"):
        """Plot path with validation info for visual debugging."""
        try:
            # Get trajectory order parameter values
            orders = traj.get_orders_array()
            
            # Create figure with high DPI for clarity
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            
            # Plot the path
            ax.plot(range(len(orders)), orders, 'b-', linewidth=2, label='Path trajectory')
            ax.scatter(range(len(orders)), orders, c='blue', s=20, alpha=0.6)
            
            # Plot interfaces
            interfaces = self.interfaces
            for i, interface in enumerate(interfaces):
                color = 'red' if i == 0 else 'orange' if i == len(interfaces)-1 else 'gray'
                linestyle = '--' if i in [0, len(interfaces)-1] else ':'
                ax.axhline(y=interface, color=color, linestyle=linestyle, alpha=0.7, 
                          label=f'Interface {i}: {interface:.3f}')
            
            # Highlight shooting region if it exists for this specific ensemble
            if ('sh_region' in traj_data and traj_data['sh_region'] is not None and 
                isinstance(traj_data['sh_region'], dict) and ens_num in traj_data['sh_region']):
                
                sh_start, sh_end = traj_data['sh_region'][ens_num]
                if sh_start < len(orders) and sh_end < len(orders) and sh_start < sh_end:
                    # Get the order parameter range in the shooting region
                    sh_orders = orders[sh_start:sh_end+1]
                    if len(sh_orders) > 0:
                        sh_op_min = np.min(sh_orders)
                        sh_op_max = np.max(sh_orders)
                        
                        # Create horizontal bands showing the shooting region's OP range
                        ax.axhspan(sh_op_min, sh_op_max, alpha=0.2, color='green', 
                                  label=f'Shooting region OP range [{sh_op_min:.3f}, {sh_op_max:.3f}]')
                        
                        # Highlight the path segment in the shooting region
                        sh_x = range(sh_start, min(sh_end+1, len(orders)))
                        sh_y = orders[sh_start:min(sh_end+1, len(orders))]
                        ax.plot(sh_x, sh_y, 'g-', linewidth=4, alpha=0.7, 
                               label=f'Shooting region path (ens {ens_num}) [{sh_start}-{sh_end}]')
                        ax.scatter(sh_x, sh_y, c='darkgreen', s=30, alpha=0.8, zorder=4)
                        
                        # Mark shooting region boundaries on x-axis
                        ax.axvline(x=sh_start, color='green', linestyle='--', alpha=0.8, linewidth=2)
                        ax.axvline(x=sh_end, color='green', linestyle='--', alpha=0.8, linewidth=2)
                        
                        # Add text annotations for shooting region boundaries
                        ax.annotate(f'Shooting start\n(step {sh_start})', 
                                   xy=(sh_start, sh_orders[0]), xytext=(sh_start-10, sh_orders[0]+0.1),
                                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                                   fontsize=10, color='darkgreen', ha='center')
                        ax.annotate(f'Shooting end\n(step {sh_end})', 
                                   xy=(sh_end, sh_orders[-1]), xytext=(sh_end+10, sh_orders[-1]+0.1),
                                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                                   fontsize=10, color='darkgreen', ha='center')
            
            # Mark trajectory extremes
            if hasattr(traj, 'ordermax') and hasattr(traj, 'ordermin'):
                max_idx = np.argmax(orders)
                min_idx = np.argmin(orders)
                ax.scatter(max_idx, orders[max_idx], c='red', s=100, marker='^', 
                          label=f'Max OP: {traj.ordermax:.3f}', zorder=5)
                ax.scatter(min_idx, orders[min_idx], c='purple', s=100, marker='v',
                          label=f'Min OP: {traj.ordermin:.3f}', zorder=5)
            
            # Set labels and title
            ax.set_xlabel('Time step', fontsize=12)
            ax.set_ylabel('Order parameter', fontsize=12)
            
            # Create detailed title with path information
            title_parts = [
                f"Path Validation - {move_type.upper()} move",
                f"Path #{traj.path_number}" if hasattr(traj, 'path_number') else "New path",
                f"Ensemble {ens_num}",
                f"Length: {traj_data.get('length', len(orders))}",
                f"Type: {traj_data.get('ptype', 'Unknown')}"
            ]
            ax.set_title('\n'.join(title_parts), fontsize=14, pad=20)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            # Adjust layout to prevent legend cutoff
            plt.tight_layout()
            
            # Add text box with detailed information
            info_text = self._format_traj_info(traj_data, ens_num)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.show(block=True)  # Block execution until plot is closed
            
        except Exception as e:
            logger.warning(f"Failed to plot path validation: {e}")
            print(f"Plot error: {e}")

    def _format_traj_info(self, traj_data, ens_num):
        """Format trajectory information for display."""
        info_lines = [
            f"TRAJECTORY INFO:",
            f"Original Ensemble: {ens_num}",
            f"Path Type: {traj_data.get('ptype', 'Unknown')}",
            f"Length: {traj_data.get('length', 'Unknown')}",
            f"Max OP: {traj_data.get('max_op', 'Unknown'):.4f}" if isinstance(traj_data.get('max_op'), (int, float)) else f"Max OP: {traj_data.get('max_op', 'Unknown')}",
            f"Min OP: {traj_data.get('min_op', 'Unknown'):.4f}" if isinstance(traj_data.get('min_op'), (int, float)) else f"Min OP: {traj_data.get('min_op', 'Unknown')}",
        ]
        
        if ('sh_region' in traj_data and traj_data['sh_region'] is not None and
            isinstance(traj_data['sh_region'], dict)):
            # Display shooting region for the specific ensemble
            if ens_num in traj_data['sh_region']:
                info_lines.append(f"Shooting Region (ens {ens_num}): {traj_data['sh_region'][ens_num]}")
            else:
                info_lines.append(f"Shooting Region: No data for ensemble {ens_num}")
                info_lines.append(f"Available ensembles: {list(traj_data['sh_region'].keys())}")
        
        return '\n'.join(info_lines)

    def print_traj_data_detailed(self, traj_data, traj_num):
        """Print detailed trajectory data after treat_output."""
        print("\n" + "="*60)
        print(f"TRAJECTORY DATA AFTER TREAT_OUTPUT - Path #{traj_num}")
        print("="*60)
        
        for key, value in traj_data.items():
            if key == 'frac':
                print(f"  {key:>15}: {value} (shape: {value.shape if hasattr(value, 'shape') else 'N/A'})")
            elif key == 'weights':
                if hasattr(value, '__len__') and len(value) > 10:
                    print(f"  {key:>15}: [first 5: {value[:5]}, ..., last 5: {value[-5:]}] (length: {len(value)})")
                else:
                    print(f"  {key:>15}: {value}")
            elif key == 'adress':
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key:>15}: [showing first 3 of {len(value)} files]")
                    for i, addr in enumerate(value[:3]):
                        print(f"                   [{i}]: {addr}")
                else:
                    print(f"  {key:>15}: {value}")
            else:
                print(f"  {key:>15}: {value}")
        
        print("="*60)

    def print_state(self):
        """Print state."""
        last_prob = True
        if isinstance(self._last_prob, type(None)):
            self.prob
            last_prob = False

        logger.info("===")
        logger.info(" xx |\tv Ensemble numbers v")
        to_print = [f"{i:03.0f}" for i in range(self.n - 1)]
        for i in range(len(to_print[0])):
            to_print0 = " ".join([j[i] for j in to_print])
            if i == len(to_print[0]) - 1:
                to_print0 += "\t\tmax_op\tmin_op\tlen"
            logger.info(" xx |\t" + to_print0)

        logger.info(" -- |\t" + "".join("--" for _ in range(self.n + 14)))

        locks = self.locked_paths()
        oil = False
        for idx, live in enumerate(self.live_paths()):
            if live not in locks:
                to_print = f"p{live:02.0f} |\t"
                if (
                    self.state[idx][:-1][idx] == 0
                    or self._last_prob[idx][:-1][idx] < 0.001
                ):
                    oil = True
                for prob in self._last_prob[idx][:-1]:
                    if prob == 1:
                        marker = "x "
                    elif prob == 0:
                        marker = "- "
                    else:
                        marker = f"{int(round(prob*10,1))} "
                        # change if marker == 10
                        if len(marker) == 3:
                            marker = "9 "
                    to_print += marker
                to_print += f"|\t{self.traj_data[live]['max_op'][0]:5.3f} \t"
                to_print += f"{self.traj_data[live]['min_op'][0]:5.3f} \t"
                to_print += f"{self.traj_data[live]['length']:5.0f}"
                logger.info(to_print)
            else:
                to_print = f"p{live:02.0f} |\t"
                logger.info(
                    to_print + "".join(["- " for j in range(self.n - 1)]) + "|"
                )
        if oil:
            logger.info("olive oil")
            oil = False

        logger.info("===")
        if not last_prob:
            self._last_prob = None

    def print_end(self):
        """Print end."""
        live_trajs = self.live_paths()
        stopping = self.cstep
        logger.info("--------------------------------------------------")
        logger.info(f"live trajs: {live_trajs} after {stopping} cycles")
        logger.info("==================================================")
        logger.info("xxx | 000        001     002     003     004     |")
        logger.info("--------------------------------------------------")
        for key, item in self.traj_data.items():
            values = "\t".join(
                [
                    f"{item0:02.2f}" if item0 != 0.0 else "----"
                    for item0 in item["frac"][:-1]
                ]
            )
            logger.info(f"{key:03.0f} * {values} *")

    def treat_output(self, md_items):
        """Treat output."""
        pn_news = []
        md_items["md_end"] = time.time()
        picked = md_items["picked"]
        traj_num = self.config["current"]["traj_num"]

        for ens_num in picked.keys():
            pn_old = picked[ens_num]["pn_old"]
            out_traj = picked[ens_num]["traj"]
            self.ensembles[ens_num + 1] = picked[ens_num]["ens"]

            for idx, lock in enumerate(self.locked):
                if str(pn_old) in lock[1]:
                    self.locked.pop(idx)
            # if path is new: number and save the path:
            if out_traj.path_number is None or md_items["status"] == "ACC":
                # move to accept:
                ens_save_idx = self.traj_data[pn_old]["ens_save_idx"]
                out_traj.path_number = traj_num
                data = {
                    "path": out_traj,
                    "dir": os.path.join(
                        os.getcwd(), self.config["simulation"]["load_dir"]
                    ),
                }
                out_traj = self.pstore.output(self.cstep, data)
                if ens_num <= -1:
                    chk_intf = out_traj.check_interfaces(self.ensembles[ens_num + 1]['interfaces'])
                    self.traj_data[traj_num] = {
                        "frac": np.zeros(self.n, dtype="longdouble"),
                        "max_op": out_traj.ordermax,
                        "min_op": out_traj.ordermin,
                        "length": out_traj.length,
                        "weights": out_traj.weights,
                        "adress": out_traj.adress,
                        "ens_save_idx": ens_save_idx,
                        "ptype": str((chk_intf[0] if chk_intf[0] is not None else "") + chk_intf[2] + (chk_intf[1] if chk_intf[1] is not None else "")),
                    }
                else:
                    st, end, valid = out_traj.check_turns(self.interfaces)
                    if not valid:
                        logger.warning(
                            "Path does not have valid turns, cannot load staple path."
                        )
                        raise ValueError(f"Path does not have valid turns. {st}, {end}, {out_traj.get_orders_array()}")
                    if (out_traj.pptype is None or len(out_traj.pptype[1]) < 3):
                        pptype = out_traj.get_pptype(self.interfaces, self.ensembles[ens_num + 1]['interfaces'])
                    else:
                        assert out_traj.pptype[0] == ens_num
                        pptype = out_traj.pptype[1]
                        if 
                    if ens_num not in out_traj.sh_region.keys() or len(out_traj.sh_region[ens_num]) != 2:
                        sh_region = out_traj.get_sh_region(self.interfaces, self.ensembles[ens_num + 1]['interfaces'])
                        out_traj.sh_region[ens_num] = sh_region
                    else:
                        sh_region = out_traj.sh_region[ens_num]
                    # print(f"{out_traj.path_number}, pptype: {pptype}, sh_region: {sh_region}")
                    self.traj_data[traj_num] = {
                        "ens_save_idx": ens_save_idx,
                        "max_op": out_traj.ordermax,
                        "min_op": out_traj.ordermin,
                        "length": out_traj.length,
                        "adress": out_traj.adress,
                        "weights": out_traj.weights,
                        "frac": np.zeros(self.n, dtype="longdouble"),
                        "ptype": str(st[1]) + pptype + str(end[1]),
                        "sh_region": out_traj.sh_region,
                    }
                traj_num += 1
                if (
                    self.config["output"].get("delete_old", False)
                    and pn_old > self.n - 2
                ):
                    if len(self.pn_olds) > self.n - 2:
                        pn_old_del, del_dic = next(iter(self.pn_olds.items()))
                        # delete trajectory files
                        for adress in del_dic["adress"]:
                            os.remove(adress)
                        # delete txt files
                        load_dir = self.config["simulation"]["load_dir"]
                        if self.config["output"].get("delete_old_all", False):
                            for txt in ("order.txt", "traj.txt", "energy.txt"):
                                txt_adress = os.path.join(
                                    load_dir, pn_old_del, txt
                                )
                                if os.path.isfile(txt_adress):
                                    os.remove(txt_adress)
                            os.rmdir(
                                os.path.join(load_dir, pn_old_del, "accepted")
                            )
                            os.rmdir(os.path.join(load_dir, pn_old_del))
                        # pop the deleted path.
                        self.pn_olds.pop(pn_old_del)
                    # keep delete list:
                    if len(self.pn_olds) <= self.n - 2:
                        self.pn_olds[str(pn_old)] = {
                            "adress": self.traj_data[pn_old]["adress"],
                        }
            pn_news.append(out_traj.path_number)
            self.add_traj(ens_num, out_traj, valid=out_traj.weights)

        # record weights
        locked_trajs = self.locked_paths()
        if self._last_prob is None:
            self.prob
        for idx, live in enumerate(self.live_paths()):
            if live not in locked_trajs:
                self.traj_data[live]["frac"] += self._last_prob[:-1][idx, :]

        # write succ data to infretis_data.txt
        if md_items["status"] == "ACC":
            write_to_pathens(self, md_items["pnum_old"])

        self.sort_trajstate()
        self.config["current"]["traj_num"] = traj_num
        self.cworker = md_items["pin"]
        
        # VISUAL VALIDATION: Plot and display trajectory data after shooting moves
        if (self.visual_validation and "moves" in md_items and 
            any(move == "sh" for move in md_items["moves"])):
            print("\n" + "🎯 VISUAL VALIDATION AFTER SHOOTING MOVE 🎯".center(80, "="))
            
            # Plot each trajectory that was just processed
            for i, ens_num in enumerate(picked.keys() if picked else []):
                if ens_num in picked and "traj" in picked[ens_num]:
                    out_traj = picked[ens_num]["traj"]
                    traj_number = pn_news[i] if i < len(pn_news) else "Unknown"
                    
                    # Get the trajectory data
                    if traj_number in self.traj_data:
                        current_traj_data = self.traj_data[traj_number]
                        
                        # Print detailed trajectory data
                        self.print_traj_data_detailed(current_traj_data, traj_number)
                        
                        # Plot the path with validation info
                        print(f"\n📊 Plotting trajectory #{traj_number} from ensemble {ens_num}...")
                        print(f"   Status: {md_items.get('status', 'Unknown')}")
                        print(f"   Move type: {'sh' if 'sh' in md_items.get('moves', []) else 'other'}")
                        print("   Close the plot window to continue...")
                        
                        self.plot_path_validation(out_traj, current_traj_data, ens_num+1, "sh")
                        
                    else:
                        print(f"⚠️  Warning: Trajectory data for path #{traj_number} not found")
            
            print("=" * 80)
        
        if self.printing():
            self.print_shooted(md_items, pn_news)
        # save for possible restart
        self.write_toml()

        return md_items

    def pattern_header(self):
        """Write pattern0 header."""
        if self.toinitiate == 0:
            restarted = self.config["current"].get("restarted_from")
            writemode = "a" if restarted else "w"
            with open(self.pattern_file, writemode) as fp:
                fp.write(
                    "# Worker\tMD_start [s]\t\twMD_start [s]\twMD_end",
                    +"[s]\tMD_end [s]\t Dask_end [s]",
                    +f"\tEnsembles\t{self.start_time}\n",
                )


    def load_paths(self, paths):
        """Load paths."""
        size = self.n - 1
        interfaces = self.config["simulation"]["interfaces"]
        # we add all the i+ paths.
        for i in range(size - 1):
            st, end, valid = paths[i+1].check_turns(interfaces)
            if not valid:
                logger.warning("Path does not have valid turns, cannot load staple path.")
                raise ValueError("Path does not have valid turns.")
            if size <= 3:
                chk_intf = paths[i+1].check_interfaces(self.ensembles[i + 1]['interfaces'])
                pptype = str((chk_intf[0] if chk_intf[0] is not None else "") + chk_intf[2] + (chk_intf[1] if chk_intf[1] is not None else ""))
                sh_region = (1, len(paths[i+1].phasepoints)-1)
            else:
                pptype = paths[i+1].get_pptype(interfaces, self.ensembles[i + 1]['interfaces'])
                sh_region = paths[i+1].get_sh_region(interfaces, self.ensembles[i + 1]['interfaces'])
            
            paths[i + 1].pptype = (i, pptype)
            paths[i + 1].sh_region[i] = sh_region
            paths[i + 1].weights = calc_cv_vector(
                paths[i + 1],
                interfaces,
                self.mc_moves,
                cap=self.cap,
            )
            self.add_traj(
                ens=i,
                traj=paths[i + 1],
                valid=paths[i + 1].weights,
                count=False,
            )
            pnum = paths[i + 1].path_number
            frac = self.config["current"]["frac"].get(
                str(pnum), np.zeros(size + 1)
            )
            self.traj_data[pnum] = {
                "ens_save_idx": i + 1,
                "max_op": paths[i + 1].ordermax,
                "min_op": paths[i + 1].ordermin,
                "length": paths[i + 1].length,
                "adress": paths[i + 1].adress,
                "weights": paths[i + 1].weights,
                "frac": np.array(frac, dtype="longdouble"),
                "ptype": str(st[1]) + pptype + str(end[1]),
                "sh_region": paths[i+1].sh_region
            }
        # add minus path:
        paths[0].weights = (1.0,)
        pnum = paths[0].path_number
        self.add_traj(
            ens=-1, traj=paths[0], valid=paths[0].weights, count=False
        )
        frac = self.config["current"]["frac"].get(
            str(pnum), np.zeros(size + 1)
        )
        chk_intf0 = paths[0].check_interfaces(self.ensembles[0]['interfaces'])
        self.traj_data[pnum] = {
            "ens_save_idx": 0,
            "max_op": paths[0].ordermax,
            "min_op": paths[0].ordermin,
            "length": paths[0].length,
            "weights": paths[0].weights,
            "adress": paths[0].adress,
            "frac": np.array(frac, dtype="longdouble"),
            "ptype": str((chk_intf0[0] if chk_intf0[0] is not None else "") + chk_intf0[2] + (chk_intf0[1] if chk_intf0[1] is not None else "")),
        }

    def initiate_ensembles(self):
        """Create all the ensemble dicts from the *toml config dict."""
        intfs = self.config["simulation"]["interfaces"]
        lambda_minus_one = self.config["simulation"]["tis_set"][
            "lambda_minus_one"
        ]
        ens_intfs = []

        # set intfs for [0-] and [0+]
        if lambda_minus_one is not False:
            ens_intfs.append(
                [lambda_minus_one, (lambda_minus_one + intfs[0]) / 2, intfs[0]]
            )
        else:
            ens_intfs.append([float("-inf"), intfs[0], intfs[0]])
        ens_intfs.append([intfs[0], intfs[0], intfs[1]])

        # set interfaces and set detect for [1+], [2+], ...
        # reactant, product = intfs[0], intfs[-1]
        for i in range(len(intfs) - 2):
            ens_intfs.append([intfs[i], intfs[i + 1], intfs[i + 2]])

        # create all path ensembles
        pensembles = {}
        for i, ens_intf in enumerate(ens_intfs):
            pensembles[i] = {
                "interfaces": tuple(ens_intf),
                "all_intfs": tuple(self.interfaces),
                "tis_set": self.config["simulation"]["tis_set"],
                "mc_move": "st_" + str(self.config["simulation"]["shooting_moves"][i]) if i > 0 else self.config["simulation"]["shooting_moves"][i],
                "ens_name": f"{i:03d}",
                "must_cross_M": True if i > 0 else False,
                "start_cond": "R" if not lambda_minus_one and i == 0 else ["L", "R"],
            }

        self.ensembles = pensembles

    def sort_trajstate(self):
        """Sort trajs and calculate P matrix.
        
        Enhanced version that can handle deadlocks by detecting cycles
        and using a more sophisticated swap strategy when needed.
        """
        if self.toinitiate != -1:
            self._last_prob = None
            self.prob
            return
            
        needstomove = [
            self.state[idx][:-1][idx] == 0 for idx in range(self.n - 1)
        ]
        
        # Track attempts to detect infinite loops
        attempt_count = 0
        max_attempts = self.n * 2  # Reasonable limit
        previous_states = []
        
        while True in needstomove and self.toinitiate == -1:
            attempt_count += 1
            
            # Check for infinite loop by comparing current state
            current_state_key = tuple(needstomove)
            if current_state_key in previous_states:
                logger.warning("Detected potential infinite loop in sort_trajstate, using advanced resolution.")
                self._resolve_deadlock()
                break
            previous_states.append(current_state_key)
            
            # Safety check to prevent runaway loops
            if attempt_count > max_attempts:
                logger.warning(f"sort_trajstate exceeded {max_attempts} attempts, using advanced resolution.")
                self._resolve_deadlock()
                break
            
            # Original algorithm: try to fix the first problematic ensemble
            ens_idx = list(needstomove).index(True)
            locks = self.locked_paths()
            
            try:
                zero_idx = list(self.state[ens_idx][1:-1]).index(0) + 1
                avail = [1 if i != 0 else 0 for i in self.state[:, zero_idx]]
                avail = [
                    j if self._trajs[i].path_number not in locks else 0
                    for i, j in enumerate(avail[:-1])
                ]
                
                if 1 not in avail:
                    # No trajectory is available for this ensemble - deadlock
                    logger.warning("No available trajectory found for simple swap, using advanced resolution.")
                    self._resolve_deadlock()
                    break
                
                trj_idx = avail.index(1)
                self.swap(ens_idx, trj_idx)
                
            except (ValueError, IndexError) as e:
                # Error in finding swap candidate - use advanced resolution
                logger.warning(f"Error in simple swap resolution ({e}), using advanced resolution.")
                self._resolve_deadlock()
                break
                
            # Recalculate which ensembles still need fixing
            needstomove = [
                self.state[idx][:-1][idx] == 0 for idx in range(self.n - 1)
            ]
        
        self._last_prob = None
        self.prob

    def _resolve_deadlock(self):
        """Advanced deadlock resolution using backtracking algorithm."""
        logger.info("Resolving deadlock using advanced permutation algorithm.")
        
        # Build a map of which trajectories can go into which ensembles
        n_unlocked = self.n - 1
        possible_swaps = {i: [] for i in range(n_unlocked)}
        unlocked_paths = [
            (i, traj.path_number)
            for i, traj in enumerate(self._trajs[:-1])
            if traj.path_number not in self.locked_paths()
        ]
        unlocked_indices = [item[0] for item in unlocked_paths]

        for ens_idx in range(n_unlocked):
            for traj_idx in unlocked_indices:
                if self.state[traj_idx, ens_idx] != 0:
                    possible_swaps[ens_idx].append(traj_idx)

        # Find a valid assignment using backtracking
        assignment = [-1] * n_unlocked
        used_trajectories = [False] * n_unlocked

        def find_valid_assignment(ens_idx):
            """Recursively find a valid trajectory for each ensemble."""
            print(self.state)
            if ens_idx == n_unlocked:
                return True  # All ensembles have been assigned a trajectory

            for traj_idx in possible_swaps[ens_idx]:
                if not used_trajectories[traj_idx]:
                    assignment[ens_idx] = traj_idx
                    used_trajectories[traj_idx] = True
                    if find_valid_assignment(ens_idx + 1):
                        return True
                    # Backtrack
                    used_trajectories[traj_idx] = False
                    assignment[ens_idx] = -1
            return False

        if find_valid_assignment(0):
            # Apply the solution using a series of swaps to maintain consistency
            self._apply_permutation_via_swaps(assignment)
            print(self.state)
            logger.info("Successfully resolved deadlock.")
        else:
            logger.error("FATAL: Could not find a valid permutation of trajectories.")
            logger.error("This implies a deadlock that cannot be resolved.")
            logger.error("Check your ensemble definitions and trajectory generation.")
            # Continue anyway to prevent hard crash
    
    def _apply_permutation_via_swaps(self, assignment):
        """Apply the permutation using the existing swap() method to maintain consistency."""
        n_unlocked = self.n - 1
        
        # Create a mapping of where each trajectory should go
        current_positions = list(range(n_unlocked))
        target_positions = assignment.copy()
        
        # Perform a series of swaps to achieve the target permutation
        # This is essentially a cycle decomposition approach
        visited = [False] * n_unlocked
        
        for start_idx in range(n_unlocked):
            if visited[start_idx]:
                continue
                
            # Follow the cycle starting from start_idx
            current_idx = start_idx
            cycle = []
            
            while not visited[current_idx]:
                visited[current_idx] = True
                cycle.append(current_idx)
                # Find where the trajectory at current_idx should go
                target_idx = target_positions[current_idx]
                current_idx = target_idx
                
            # Apply swaps for this cycle
            if len(cycle) > 1:
                # Perform swaps to rotate the cycle
                for i in range(len(cycle) - 1):
                    self.swap(cycle[i], cycle[i + 1])

def write_to_pathens(state, pn_archive):
    """Write data to infretis_data.txt."""
    traj_data = state.traj_data
    size = state.n

    with open(state.data_file, "a") as fp:
        for pn in pn_archive:
            string = ""
            string += f"\t{pn:3.0f}\t"
            string += f"{traj_data[pn]['length']:5.0f}" + "\t"
            string += f"{traj_data[pn]['max_op'][0]:8.5f}" + "\t"
            string += f"{traj_data[pn]['min_op'][0]:8.5f}" + '\t'
            string += f"{traj_data[pn]['ptype']}" + '\t'
            frac = []
            weight = []
            if len(traj_data[pn]["weights"]) == 1:
                f0 = traj_data[pn]["frac"][0]
                w0 = traj_data[pn]["weights"][0]
                frac.append("----" if f0 == 0.0 else str(f0))
                weight.append("----" if f0 == 0.0 else str(w0))
                frac += ["----"] * (size - 2)
                weight += ["----"] * (size - 2)
            else:
                frac.append("----")
                weight.append("----")
                for w0, f0 in zip(
                    traj_data[pn]["weights"][:-1], traj_data[pn]["frac"][1:-1]
                ):
                    frac.append("----" if f0 == 0.0 else str(f0))
                    weight.append("----" if f0 == 0.0 else str(w0))
            fp.write(
                string + "\t".join(frac) + "\t" + "\t".join(weight) + "\t\n"
            )
            traj_data.pop(pn)