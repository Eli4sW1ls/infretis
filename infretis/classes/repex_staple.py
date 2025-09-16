"""Defines the main REPEX class for path handling and permanent calc."""
import logging
import os
import sys
import time
from datetime import datetime
import traceback

import numpy as np
import itertools
import tomli_w
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components

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
        self.visual_validation = config.get("output", {}).get("visual_validation", False)
        
        # Disable infinite swap for STAPLE by default (can be re-enabled in config)
        # This prevents weird results in STAPLE simulations by using identity matrix
        # Set "staple_infinite_swap": true in config to re-enable infinite swap
        self.staple_infinite_swap = config.get("simulation", {}).get("staple_infinite_swap", False)

    @property
    def prob(self):
        """Calculate the P matrix for STAPLE.
        
        By default, STAPLE simulations disable infinite swap (similar to REPPTIS)
        to prevent weird results. This returns an identity matrix where each path
        stays in its own ensemble (100% probability) with zero for the ghost ensemble.
        
        To re-enable infinite swap (full permanent calculation), set 
        "staple_infinite_swap": true in the simulation config.
        """
        if not self.staple_infinite_swap:
            # Disable infinite swap: return identity matrix like repex_pp
            if self._last_prob is None:
                logger.info("STAPLE: Infinite swap DISABLED - using identity matrix (no path swaps)")
            prop = np.identity(self.n)
            prop[-1, -1] = 0.  # Zero probability for ghost ensemble
            self._last_prob = prop.copy()
            return prop * (1 - np.array(self._locks))
        else:
            # Use full infinite swap calculation from base REPEX class
            if self._last_prob is None:
                logger.info("STAPLE: Infinite swap ENABLED - using full permanent calculation")
                prob = self.inf_retis(abs(self.state), self._locks)
                self._last_prob = prob.copy()
            return self._last_prob

    def find_blocks(self, arr):
        """
        Find blocks in a W matrix using graph theory.
        This is a robust method that can handle any matrix structure.
        """
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Input matrix must be square.")
        
        # Create a graph from the non-zero elements of the matrix
        # Symmetrize the graph to ensure undirected components are found correctly
        graph = (arr != 0) | (arr.T != 0)
        
        # Find connected components, which correspond to the blocks
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        blocks = []
        for i in range(n_components):
            # Find the indices of the nodes in the current component
            block_indices = np.where(labels == i)[0]
            blocks.append(block_indices)
            
        # Sort blocks by their starting index for deterministic processing
        blocks.sort(key=lambda b: b[0])
        return blocks

    def inf_retis(self, input_mat, locks):
        """
        Permanent calculator for STAPLE, optimized with robust block detection.
        """
        # 1. Handle locked rows/columns
        bool_locks = locks == 1
        live_indices = np.where(~bool_locks)[0]
        
        if len(live_indices) == 0:
            return np.zeros_like(input_mat, dtype="longdouble")

        # Create a submatrix with only the "live" (unlocked) rows and columns
        live_mat = input_mat[np.ix_(live_indices, live_indices)]

        # 2. Find independent blocks in the live matrix (use binary version for connectivity)
        binary_mat = np.where(live_mat != 0, 1, 0)
        print("live matrix:\n", live_mat)
        print("Binary matrix:\n", binary_mat)
        
        sym_mask = (live_mat != 0) & (live_mat.T != 0)
        masked_live_mat = live_mat * sym_mask
        # binary_mat = np.where(masked_live_mat != 0, 1, 0)
        # warn if masking removed rows/columns entirely
        removed_rows = np.where(np.all(masked_live_mat == 0, axis=1))[0]
        if removed_rows.size > 0:
            logger.warning(
                "staple_enforce_diag removed all nonzero entries for rows %s; no valid permutations remain for those rows",
                removed_rows.tolist(),
            )
            
        blocks = self.find_blocks(binary_mat)
        
        out = np.zeros_like(live_mat, dtype="longdouble")

        # 3. Process each block independently
        for block_indices in blocks:
            # Create a view of the sub-matrix for the current block using ACTUAL WEIGHTS
            # Use the weighted live matrix (not the binary connectivity matrix)
            sub_matrix = live_mat[np.ix_(block_indices, block_indices)]

            if sub_matrix.shape[0] == 0:
                continue
            
            prob_matrix = np.zeros_like(sub_matrix, dtype="longdouble")
            
            # Heuristics for choosing the best permanent algorithm
            if sub_matrix.shape[0] == 1:
                prob_matrix = np.array([[1.0]])
            elif np.all(np.isclose(sub_matrix, sub_matrix[0, :]) | (sub_matrix == 0)):
                prob_matrix = self.quick_prob(sub_matrix)
            elif sub_matrix.shape[0] <= 12:
                # Use exact Glynn formula for small matrices
                prob_matrix = self.permanent_prob(sub_matrix)
            else:
                # Fallback to probabilistic method for large, complex matrices
                self._random_count += 1
                logger.info(f"Using random method for block of size {sub_matrix.shape[0]}")
                prob_matrix = self.random_prob(sub_matrix)

            # Place the calculated probability matrix into the correct block of the output matrix
            out[np.ix_(block_indices, block_indices)] = prob_matrix

        # 4. Normalize and verify the probability matrix
        row_sums = np.sum(out, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero for empty rows
        if np.any(row_sums > 0):
            if np.any(row_sums > 1):
                print(f"Warning: Row sums exceed 1: {out}")
            out /= row_sums
        
        # 5. Re-insert rows/columns for locked ensembles
        final_out = np.zeros_like(input_mat, dtype="longdouble")
        final_out[np.ix_(live_indices, live_indices)] = out
        try:
            out_repex = REPEX_state(self.config, minus=True).inf_retis(input_mat, locks)
        except Exception as e:
            logger.info(f"Error occurred while calculating REPEX: {e}")
            out_repex = np.zeros_like(input_mat, dtype="longdouble")
        print("locks:", locks)
        print("staple out:\n", final_out)
        print("repex out:\n", out_repex)
        return final_out

    def add_traj(self, ens, traj, valid, count=True, n=0):
        """Add traj to state and calculate P matrix.
        
        When infinite swap is disabled, behaves like REPPTIS where each path
        stays in its own ensemble with 100% probability.
        """
        if not self.staple_infinite_swap:
            # Disable infinite swap: use REPPTIS-style behavior
            valid = np.zeros(self.n)
            valid[ens + self._offset] = 1.
            ens += self._offset
            assert valid[ens] != 0
            # invalidate last prob
            self._last_prob = None
            self._trajs[ens] = traj
            self.state[ens, :] = valid
            self.unlock(ens)
            # Calculate P matrix (will return identity matrix)
            self.prob
        else:
            # Use full infinite swap behavior from base class
            super().add_traj(ens, traj, valid, count, n)

    def permanent_prob(self, arr):
        """P matrix calculation for specific W matrix."""
        out = np.zeros(shape=arr.shape, dtype="longdouble")
        # Don't overwrite input arr
        scaled_arr = arr.copy()
        n = len(scaled_arr)
        # Rescaling the W-matrix avoids numerical instabilities when the
        # matrix is large and contains large weights from
        # high-acceptance moves
        if n > 0:
            row_max = np.max(scaled_arr, axis=1, keepdims=True)
            row_max[row_max == 0] = 1.0
            scaled_arr = scaled_arr / row_max

        for i in range(n):
            rows = [r for r in range(n) if r != i]
            sub_arr = scaled_arr[rows, :]
            for j in range(n):
                if scaled_arr[i][j] == 0:
                    continue
                columns = [r for r in range(n) if r != j]
                M = sub_arr[:, columns]
                f = self.fast_glynn_perm(M)
                out[i][j] = f * scaled_arr[i][j]
        
        # Use the same normalization as base REPEX
        row_sums = np.sum(out, axis=1)
        max_row_sum = max(row_sums) if len(row_sums) > 0 else 1.0
        if max_row_sum > 0:
            return out / max_row_sum
        return out

    def quick_prob(self, arr):
        """Optimized quick P matrix calculation leveraging STAPLE's contiguous row structure."""
        # STAPLE property: each row has exactly one contiguous subsequence of non-zeros
        # This allows for much more efficient permanent calculation
        
        # Convert to working matrix (0/1) to analyze structure
        working_mat = np.where(arr != 0, 1, 0)
        n = arr.shape[0]
        
        # Check if all rows are identical (simple uniform case)
        first_row = working_mat[0]
        if np.all(np.all(working_mat == first_row, axis=1)):
            # All rows are identical - create uniform distribution
            out_mat = working_mat.astype("longdouble")
            # Normalize each row to sum to 1
            row_sums = out_mat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            out_mat = out_mat / row_sums
            return out_mat

        # For small matrices (< 5x5), optimization overhead isn't worth it
        if n < 5:
            return self.permanent_prob(arr)
        
        # For STAPLE matrices, check if we can use contiguous structure optimization
        if self._is_staple_structured(working_mat):
            return self._contiguous_permanent_prob(arr)
        else:
            # Fall back to standard permanent_prob for non-STAPLE structures
            return self.permanent_prob(arr)

    def _is_staple_structured(self, working_mat):
        """Check if matrix has STAPLE structure: contiguous non-zero subsequences in each row."""
        n = working_mat.shape[0]
        
        for i in range(n):
            row = working_mat[i]
            # Find the first and last non-zero indices
            nonzero_indices = np.where(row != 0)[0]
            
            if len(nonzero_indices) == 0:
                continue  # Empty row is valid
            
            # Check if all elements between first and last non-zero are also non-zero
            first_nonzero = nonzero_indices[0] 
            last_nonzero = nonzero_indices[-1]
            
            # All elements in range [first_nonzero, last_nonzero] should be non-zero
            for j in range(first_nonzero, last_nonzero + 1):
                if row[j] == 0:
                    return False  # Found a "hole" - not STAPLE structured
        
        return True
    
    def _contiguous_permanent_prob(self, arr):
        """Optimized permanent calculation for STAPLE's contiguous row structure."""
        n = arr.shape[0]
        out = np.zeros(shape=arr.shape, dtype="longdouble")
        
        # Scale the array to avoid numerical instabilities
        scaled_arr = arr.copy()
        if n > 0:
            row_max = np.max(scaled_arr, axis=1, keepdims=True)
            row_max[row_max == 0] = 1.0
            scaled_arr = scaled_arr / row_max

        # For each matrix element (i,j)
        for i in range(n):
            for j in range(n):
                if scaled_arr[i][j] == 0:
                    continue
                
                # Create submatrix by removing row i and column j
                rows = [r for r in range(n) if r != i]
                columns = [c for c in range(n) if c != j]
                submatrix = scaled_arr[np.ix_(rows, columns)]
                
                # Calculate permanent of submatrix using optimized method for STAPLE structure
                if submatrix.size == 0:
                    perm = 1.0
                elif submatrix.shape[0] == 1:
                    perm = submatrix[0, 0] if submatrix.shape[1] == 1 else np.sum(submatrix[0])
                else:
                    # Use contiguous structure to optimize permanent calculation
                    perm = self._fast_contiguous_permanent(submatrix)
                
                out[i][j] = perm * scaled_arr[i][j]
        
        # Normalize using the same method as base implementation
        row_sums = np.sum(out, axis=1)
        max_row_sum = max(row_sums) if len(row_sums) > 0 else 1.0
        if max_row_sum > 0:
            return out / max_row_sum
        return out
    
    def _fast_contiguous_permanent(self, matrix):
        """Fast permanent calculation leveraging contiguous structure."""
        n = matrix.shape[0]
        
        if n <= 2:
            # For small matrices, use direct calculation
            return self.fast_glynn_perm(matrix)
        
        # For larger matrices with contiguous structure, we can potentially
        # decompose the permanent calculation more efficiently
        
        # Check if matrix has a special structure we can exploit
        working_mat = np.where(matrix != 0, 1, 0)
        
        # If it's upper triangular (common in STAPLE), permanent = product of diagonal
        if self._is_upper_triangular(working_mat):
            return np.prod(np.diag(matrix))
        
        # If it's block diagonal, we can compute permanent as product of block permanents
        blocks = self._find_diagonal_blocks(working_mat)
        if len(blocks) > 1:
            total_perm = 1.0
            for block_indices in blocks:
                if len(block_indices) == 1:
                    total_perm *= matrix[block_indices[0], block_indices[0]]
                else:
                    block_matrix = matrix[np.ix_(block_indices, block_indices)]
                    total_perm *= self.fast_glynn_perm(block_matrix)
            return total_perm
        
        # For other contiguous structures, fall back to Glynn's algorithm
        # but the contiguous structure might still provide numerical advantages
        return self.fast_glynn_perm(matrix)
    
    def _is_upper_triangular(self, working_mat):
        """Check if a 0/1 matrix is upper triangular."""
        n = working_mat.shape[0]
        for i in range(n):
            for j in range(i):
                if working_mat[i, j] != 0:
                    return False
        return True
    
    def _find_diagonal_blocks(self, working_mat):
        """Find diagonal blocks in a matrix."""
        n = working_mat.shape[0]
        visited = np.zeros(n, dtype=bool)
        blocks = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Find connected component starting from i
            block = []
            stack = [i]
            
            while stack:
                current = stack.pop()
                if visited[current]:
                    continue
                    
                visited[current] = True
                block.append(current)
                
                # Add connected nodes (non-zero elements)
                for j in range(n):
                    if not visited[j] and (working_mat[current, j] != 0 or working_mat[j, current] != 0):
                        stack.append(j)
            
            if block:
                blocks.append(sorted(block))
        
        return blocks

    def enumerated_prob(self, arr, enforce_diag=False):
        """Exact P matrix via enumeration for small blocks.

        If enforce_diag is True, only permutations p with both
        arr[i, p[i]] != 0 and arr[p[i], i] != 0 for all i are allowed.
        """
        n = arr.shape[0]
        P = np.zeros_like(arr, dtype="longdouble")
        if n == 0:
            return P

        perms = itertools.permutations(range(n))
        valid = []
        for p in perms:
            if all(W[i, p[i]] != 0 for i in range(n)):    # diagonal-nonzero + row->ensemble validity
                valid.append(p)
            # ok_forward = True
            # ok_diag = True
            # for i, j in enumerate(p):
                # if arr[i, j] == 0:
                #     ok_forward = False      # 
                #     break   
                # if enforce_diag and arr[j, i] == 0:
                #     ok_diag = False
                #     break
            # if ok_forward and (not enforce_diag or ok_diag):
            #     valid.append(p)

        Z = len(valid)
        if Z == 0:
            # No valid permutations - return zero matrix (caller must handle)
            return P

        for p in valid:
            for i, j in enumerate(p):
                P[i, j] += 1.0 / Z

        return P

    def random_prob(self, arr, n=10_000):
        """P matrix calculation for specific W matrix."""
        out = np.eye(len(arr), dtype="longdouble")
        current_state = np.eye(len(arr))
        choices = len(arr) // 2
        even = choices * 2 == len(arr)

        # The probability to go right
        prob_right = np.nan_to_num(np.roll(arr, -1, axis=1) / arr)

        # The probability to go left
        prob_left = np.nan_to_num(np.roll(arr, 1, axis=1) / arr)

        start = 0
        zero_one = np.array([0, 1])
        p_m = np.array([1, -1])
        temp = np.where(current_state == 1)

        for i in range(n):
            direction = self.rgen.choice(p_m)
            if not even:
                start = self.rgen.choice(zero_one)

            temp_left = prob_left[temp]
            temp_right = prob_right[temp]

            if not even:
                start = self.rgen.choice(zero_one)

            if direction == -1:
                probs = (
                    temp_left[start:-1:2]
                    * np.roll(temp_right, 1, axis=0)[start:-1:2]
                )
            else:
                probs = temp_right[start:-1:2] * temp_left[start + 1 :: 2]

            r_nums = self.rgen.random(choices)
            success = r_nums < probs

            for j in np.where(success)[0]:
                idx = j * 2 + start
                temp_state = current_state[:, [idx + direction, idx]]
                current_state[:, [idx, idx + direction]] = temp_state
                temp_state_2 = temp[0][[idx + direction, idx]]
                temp[0][[idx, idx + direction]] = temp_state_2

            out += current_state

        return out / (n + 1)

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
            try:
                orders = traj.get_orders_array()
            except:
                orders = np.array([php.order[0] for php in traj.phasepoints])
 
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
                isinstance(traj_data['sh_region'], dict)):
                
                sh_regions = list(traj_data['sh_region'].items())
                sh_start, sh_end = sh_regions[0][1]
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
                          label=f'Max OP: {traj.ordermax[0]:.3f}', zorder=5)
                ax.scatter(min_idx, orders[min_idx], c='purple', s=100, marker='v',
                          label=f'Min OP: {traj.ordermin[0]:.3f}', zorder=5)

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

    def plot_path_comparison(self, old_path, new_path, ens_num, old_ptype=None, new_ptype=None, 
                           old_sh_region=None, new_sh_region=None):
        """Compare old and new path properties with side-by-side visualization."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get order parameter arrays
            try:
                old_orders = old_path.get_orders_array()
            except:
                old_orders = np.array([php.order[0] for php in old_path.phasepoints])
            
            try:
                new_orders = new_path.get_orders_array()
            except:
                new_orders = np.array([php.order[0] for php in new_path.phasepoints])
            
            # Create side-by-side comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
            
            # Plot old path
            ax1.plot(range(len(old_orders)), old_orders, 'b-', linewidth=2, label='Original path')
            ax1.scatter(range(len(old_orders)), old_orders, c='blue', s=20, alpha=0.6)
            
            # Plot new path
            ax2.plot(range(len(new_orders)), new_orders, 'r-', linewidth=2, label='New path')
            ax2.scatter(range(len(new_orders)), new_orders, c='red', s=20, alpha=0.6)
            
            # Add interfaces to both plots
            interfaces = self.interfaces
            for ax in [ax1, ax2]:
                for i, interface in enumerate(interfaces):
                    color = 'red' if i == 0 else 'orange' if i == len(interfaces)-1 else 'gray'
                    linestyle = '--' if i in [0, len(interfaces)-1] else ':'
                    ax.axhline(y=interface, color=color, linestyle=linestyle, alpha=0.7, 
                              label=f'Interface {i}: {interface:.3f}')
            
            # Highlight old shooting region
            if old_sh_region is not None and isinstance(old_sh_region, (tuple, list)) and len(old_sh_region) == 2:
                sh_start, sh_end = old_sh_region
                if sh_start < len(old_orders) and sh_end < len(old_orders) and sh_start < sh_end:
                    sh_orders = old_orders[sh_start:sh_end+1]
                    if len(sh_orders) > 0:
                        sh_x = range(sh_start, min(sh_end+1, len(old_orders)))
                        sh_y = old_orders[sh_start:min(sh_end+1, len(old_orders))]
                        ax1.plot(sh_x, sh_y, 'g-', linewidth=4, alpha=0.7, 
                               label=f'Old shooting region [{sh_start}-{sh_end}]')
                        ax1.axvline(x=sh_start, color='green', linestyle='--', alpha=0.8, linewidth=2)
                        ax1.axvline(x=sh_end, color='green', linestyle='--', alpha=0.8, linewidth=2)
            
            # Highlight new shooting region
            if new_sh_region is not None and isinstance(new_sh_region, (tuple, list)) and len(new_sh_region) == 2:
                sh_start, sh_end = new_sh_region
                if sh_start < len(new_orders) and sh_end < len(new_orders) and sh_start < sh_end:
                    sh_orders = new_orders[sh_start:sh_end+1]
                    if len(sh_orders) > 0:
                        sh_x = range(sh_start, min(sh_end+1, len(new_orders)))
                        sh_y = new_orders[sh_start:min(sh_end+1, len(new_orders))]
                        ax2.plot(sh_x, sh_y, 'g-', linewidth=4, alpha=0.7, 
                               label=f'New shooting region [{sh_start}-{sh_end}]')
                        ax2.axvline(x=sh_start, color='green', linestyle='--', alpha=0.8, linewidth=2)
                        ax2.axvline(x=sh_end, color='green', linestyle='--', alpha=0.8, linewidth=2)
            
            # Set titles and labels
            ax1.set_title(f'BEFORE Shooting Move\nEnsemble {ens_num}\nType: {old_ptype or "Unknown"}\nLength: {len(old_orders)}', 
                         fontsize=12, pad=15)
            ax2.set_title(f'AFTER Shooting Move\nEnsemble {ens_num}\nType: {new_ptype or "Unknown"}\nLength: {len(new_orders)}', 
                         fontsize=12, pad=15)
            
            for ax in [ax1, ax2]:
                ax.set_xlabel('Time step', fontsize=11)
                ax.set_ylabel('Order parameter', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9, loc='upper right')
            
            # Add overall title
            fig.suptitle(f'🎯 STAPLE PATH COMPARISON - Ensemble {ens_num} 🎯', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.show(block=True)  # Block execution until plot is closed
            
        except Exception as e:
            logger.warning(f"Failed to plot path comparison: {e}")
            print(f"Plot comparison error: {e}")

    def plot_sh_region_comparison(self, path, ens_num, original_sh_region=None, 
                                 current_sh_region=None, was_calculated=False):
        """Plot comparison of sh_regions on the same trajectory."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get order parameter array
            try:
                orders = path.get_orders_array()
            except:
                orders = np.array([php.order[0] for php in path.phasepoints])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
            
            # Plot the path
            ax.plot(range(len(orders)), orders, 'b-', linewidth=2, label='Path trajectory', alpha=0.7)
            ax.scatter(range(len(orders)), orders, c='blue', s=15, alpha=0.5)
            
            # Add interfaces
            interfaces = self.interfaces
            for i, interface in enumerate(interfaces):
                color = 'red' if i == 0 else 'orange' if i == len(interfaces)-1 else 'gray'
                linestyle = '--' if i in [0, len(interfaces)-1] else ':'
                ax.axhline(y=interface, color=color, linestyle=linestyle, alpha=0.7, 
                          label=f'Interface {i}: {interface:.3f}')
            
            # Plot original sh_region (if it existed)
            if original_sh_region is not None and isinstance(original_sh_region, (tuple, list)) and len(original_sh_region) == 2:
                sh_start, sh_end = original_sh_region
                if sh_start < len(orders) and sh_end < len(orders) and sh_start <= sh_end:
                    # Highlight the original shooting region
                    sh_x = range(sh_start, min(sh_end+1, len(orders)))
                    sh_y = orders[sh_start:min(sh_end+1, len(orders))]
                    ax.plot(sh_x, sh_y, 'purple', linewidth=6, alpha=0.8, 
                           label=f'ORIGINAL sh_region [{sh_start}-{sh_end}]')
                    
                    # Mark boundaries with vertical lines
                    ax.axvline(x=sh_start, color='purple', linestyle='--', alpha=0.9, linewidth=3)
                    ax.axvline(x=sh_end, color='purple', linestyle='--', alpha=0.9, linewidth=3)
                    
                    # Add text annotations
                    mid_point = (sh_start + sh_end) // 2
                    if mid_point < len(orders):
                        ax.annotate(f'ORIGINAL\n[{sh_start}-{sh_end}]', 
                                   xy=(mid_point, orders[mid_point]), 
                                   xytext=(mid_point, orders[mid_point] + 0.2),
                                   arrowprops=dict(arrowstyle='->', color='purple', alpha=0.8),
                                   fontsize=10, color='purple', ha='center', fontweight='bold')
            
            # Plot current sh_region (after get_sh_region call or unchanged)
            if current_sh_region is not None and isinstance(current_sh_region, (tuple, list)) and len(current_sh_region) == 2:
                sh_start, sh_end = current_sh_region
                if sh_start < len(orders) and sh_end < len(orders) and sh_start <= sh_end:
                    # Only plot if different from original OR if it was calculated
                    if original_sh_region != current_sh_region or was_calculated:
                        # Highlight the current shooting region
                        sh_x = range(sh_start, min(sh_end+1, len(orders)))
                        sh_y = orders[sh_start:min(sh_end+1, len(orders))]
                        color = 'green' if was_calculated else 'orange'
                        label_suffix = ' (CALCULATED)' if was_calculated else ' (UNCHANGED)'
                        ax.plot(sh_x, sh_y, color, linewidth=4, alpha=0.7, 
                               label=f'CURRENT sh_region [{sh_start}-{sh_end}]{label_suffix}')
                        
                        # Mark boundaries with vertical lines
                        ax.axvline(x=sh_start, color=color, linestyle=':', alpha=0.8, linewidth=2)
                        ax.axvline(x=sh_end, color=color, linestyle=':', alpha=0.8, linewidth=2)
                        
                        # Add text annotations
                        mid_point = (sh_start + sh_end) // 2
                        if mid_point < len(orders):
                            y_offset = 0.1 if was_calculated else -0.1
                            ax.annotate(f'CURRENT\n[{sh_start}-{sh_end}]', 
                                       xy=(mid_point, orders[mid_point]), 
                                       xytext=(mid_point, orders[mid_point] + y_offset),
                                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.8),
                                       fontsize=10, color=color, ha='center', fontweight='bold')
            
            # Set labels and title
            ax.set_xlabel('Time step', fontsize=12)
            ax.set_ylabel('Order parameter', fontsize=12)
            
            # Create detailed title
            status = "CALCULATED" if was_calculated else "ALREADY EXISTS"
            title_parts = [
                f'🎯 SHOOTING REGION COMPARISON - Ensemble {ens_num}',
                f'Status: sh_region {status}',
                f'Path Length: {len(orders)}'
            ]
            ax.set_title('\n'.join(title_parts), fontsize=14, pad=20, fontweight='bold')
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper right')
            
            # Add info box
            info_text = f"SHOOTING REGION ANALYSIS:\n"
            if original_sh_region is None:
                info_text += "• ORIGINAL: None (not set before)\n"
            else:
                info_text += f"• ORIGINAL: {original_sh_region}\n"
            
            if current_sh_region is None:
                info_text += "• CURRENT: None"
            else:
                info_text += f"• CURRENT: {current_sh_region}"
                
            if was_calculated:
                info_text += " (NEWLY CALCULATED)"
            else:
                info_text += " (UNCHANGED FROM ORIGINAL)"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.show(block=True)  # Block execution until plot is closed
            
        except Exception as e:
            logger.warning(f"Failed to plot sh_region comparison: {e}")
            print(f"Plot sh_region comparison error: {e}")

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
                    s_offset, e_offset = 0, 0
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
                    if ens_num in [0, 1] and st[1] == end[1] == 0:
                        if ens_num == 0:
                            if pptype == "LMR":
                                e_offset = 1
                            elif pptype == "RML":
                                s_offset = 1 
                        elif ens_num == 1:
                            if pptype != "LML":
                                raise ValueError(
                                    f"Ensemble {ens_num} has invalid pptype {pptype}."
                                )
                            e_offset = 1 
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
                        "ptype": str(st[1] + s_offset) + pptype + str(end[1] + e_offset),
                        "sh_region": out_traj.sh_region,
                        "mc_move": getattr(out_traj, 'generated', 'initial'),
                        "ensemble": ens_num,
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
            s_offset, e_offset = 0, 0
            if size <= 3:
                chk_intf = paths[i+1].check_interfaces(self.ensembles[i + 1]['interfaces'])
                pptype = str((chk_intf[0] if chk_intf[0] is not None else "") + chk_intf[2] + (chk_intf[1] if chk_intf[1] is not None else ""))
                sh_region = (1, len(paths[i+1].phasepoints)-1)
            else:
                pptype = paths[i+1].get_pptype(interfaces, self.ensembles[i + 1]['interfaces'])
                sh_region = paths[i+1].get_sh_region(interfaces, self.ensembles[i + 1]['interfaces'])
            if i in [0, 1] and st[1] == end[1] == 0:
                if i == 0:
                    if pptype == "LMR":
                        e_offset = 1
                    elif pptype == "RML":
                        s_offset = 1 
                elif i == 1:
                    if pptype != "LML":
                        raise ValueError(
                            f"Ensemble {i} has invalid pptype {pptype}."
                        )
                    e_offset = 1 
            if not valid:
                logger.warning("Path does not have valid turns, cannot load staple path.")
                raise ValueError("Path does not have valid turns.")
            
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
                "ptype": str(st[1] + s_offset) + pptype + str(end[1] + e_offset),
                "sh_region": paths[i+1].sh_region,
                "mc_move": getattr(paths[i + 1], 'generated', 'initial'),
                "ensemble": i,
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
            "mc_move": getattr(paths[0], 'generated', 'initial'),
            "ensemble": -1,
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
                "visual_validation": self.visual_validation,  # Pass visual validation setting to each ensemble
                "repex_state": self,  # Pass reference to repex state for plotting
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
                string + "\t".join(frac) + "\t" + "\t".join(weight) + "\t" 
                + str(traj_data[pn].get("mc_move", "unknown")) + "\t" + str(traj_data[pn].get("ensemble", "unknown")) + "\t\n"
            )
            traj_data.pop(pn)