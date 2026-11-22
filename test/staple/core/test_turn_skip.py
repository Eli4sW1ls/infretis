"""check_turns accepts a turn whose last (or first) step skips one interface, as the staple extension's
turn_detected does: a staple built by the extension must never be rejected by check_turns (it raised in
REPEX_state_staple.treat_output and stopped the run)."""
import numpy as np

from infretis.classes.staple_path import StaplePath, turn_detected
from infretis.classes.system import System

INTF = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def _path(orders):
    p = StaplePath()
    for v in orders:
        f = System()
        f.order = [v]
        p.phasepoints.append(f)
    return p


# a staple of the window (0.4, 0.5, 0.6): start turn around 0.5, up to 0.7, and a last step from 0.52 to 0.35 that
# crosses 0.5 and 0.4 at once (the path is never below 0.4 before, so the nearest interface cannot be recrossed)
ORDERS = [0.62, 0.55, 0.45, 0.52, 0.61, 0.65, 0.7, 0.63, 0.55, 0.52, 0.35]


def test_end_turn_with_one_skipped_interface():
    st, end, valid = _path(ORDERS).check_turns(INTF)
    assert valid and st[:2] == (True, 3) and end[:2] == (True, 4)       # end turn around 0.6, as turn_detected


def test_start_turn_with_one_skipped_interface():
    st, end, valid = _path(ORDERS[::-1]).check_turns(INTF)
    assert valid and st[:2] == (True, 4) and end[:2] == (True, 3)


def test_skipped_interface_does_not_count_toward_the_turn():
    # the extremum 0.58 does not pass 0.6: no turn around 0.6, also not with the skipped 0.5 counted
    assert not _path([0.62, 0.55, 0.45, 0.52, 0.58, 0.55, 0.35]).check_turns(INTF)[1][0]


def _extend(lam, intf, m_idx, j, d):
    L, R = intf[m_idx - 1], intf[m_idx + 1]
    k = j + d
    lr = -1 if lam[k] < L else (1 if lam[k] > R else None)
    if lr is None:
        return None
    orders = [lam[j], lam[k]]
    while True:
        k += d
        if not 0 <= k < len(lam):
            return None
        orders.append(lam[k])
        if lam[k] < intf[0] or lam[k] > intf[-1] or turn_detected(np.asarray(orders), list(intf), m_idx, lr):
            return k


def test_staples_built_by_the_extension_are_valid():
    """Random walks: a window segment crossing M, extended at both ends with turn_detected (as add_to_path does);
    if no step crosses more than two interfaces, check_turns must accept both turns."""
    rng = np.random.default_rng(0)
    n_tested = 0
    while n_tested < 3000:
        intf = np.r_[0.0, np.sort(rng.uniform(0.05, 0.95, 8)), 1.0]
        lam = np.cumsum(rng.normal(0, rng.choice([0.02, 0.05, 0.1]), 3000)) % 1.6
        lam = np.where(lam > 0.8, 1.6 - lam, lam) + 0.1
        m_idx = int(rng.integers(1, len(intf) - 1))
        L, M, R = intf[m_idx - 1], intf[m_idx], intf[m_idx + 1]
        inside = (lam > L) & (lam < R)
        edges = np.flatnonzero(np.diff(np.r_[False, inside, False]))
        for a, b in zip(edges[::2], edges[1::2] - 1):
            if a < 2 or b > len(lam) - 3 or not (lam[a - 1:b + 2].min() < M <= lam[a - 1:b + 2].max()):
                continue
            e, s = _extend(lam, intf, m_idx, b, +1), _extend(lam, intf, m_idx, a, -1)
            if e is None or s is None:
                continue
            path = lam[s:e + 1]
            lo, hi = np.minimum(path[:-1], path[1:]), np.maximum(path[:-1], path[1:])
            if np.max(np.sum((intf[None, :] > lo[:, None]) & (intf[None, :] <= hi[:, None]), axis=1)) > 2:
                continue
            assert StaplePath()._check_start_turn(path, intf)[0]
            assert StaplePath()._check_end_turn(path, intf)[0]
            n_tested += 1
            break


# a steep (learned) order parameter crosses several interfaces in one step: a production path of the staggered
# channels whose first step crossed four interfaces (cycle 6869; treat_output raised "Path does not have valid
# turns" and stopped the run). Its start turn lies at lambda_2 (down to the minimum at frame 4, back up past
# lambda_3 at frame 8) and its end turn at lambda_4 (the maximum at frame 8, back down past lambda_3).
STEEP_INTF = [-1.6, -1.4741, -1.2245, -1.0638, -0.9991, -0.9245, -0.7248, 0.184, 1.1789, 1.5986, 1.6]
STEEP = [-0.65971255, -1.09321847, -1.07310752, -1.22541446, -1.35621381, -1.26074359, -1.31393825,
         -1.15024791, -0.93043589, -1.23742464]


def test_turns_with_several_skipped_interfaces():
    st, end, valid = _path(STEEP).check_turns(STEEP_INTF)
    assert valid
    assert st == (True, 2, 4) and end == (True, 4, 8)


def test_start_turn_agrees_with_turn_detected_on_random_walks():
    """For random walks with large steps (several interfaces per step), wherever the extension's turn_detected
    reports a completed turn, check_turns finds a start turn at the same interface (read backward, the walk is the
    path start made by a backward extension)."""
    rng = np.random.default_rng(3)
    intf = np.linspace(0.0, 1.0, 11)
    n_checked = 0
    for _ in range(3000):
        x = [rng.uniform(0.3, 0.7)]
        for _ in range(40):
            x.append(float(np.clip(x[-1] + rng.normal(0, 0.12), 0.01, 0.99)))
            lr = 1 if x[1] > x[0] else -1
            # m_idx: the middle interface the extension may not pass; here one the walk never reaches
            if len(x) >= 3 and turn_detected(np.array(x), list(intf), len(intf) - 1 if lr == -1 else 0, lr):
                ext = max(x) if lr == 1 else min(x)
                elig = [k for k, v in enumerate(intf) if (v <= ext if lr == 1 else v >= ext)]
                elig = [k for k in elig if (intf[k] >= x[0] if lr == 1 else intf[k] <= x[0])]
                if len(elig) >= 2:                                  # at least two interfaces from the start
                    ok, k, _ = StaplePath()._check_start_turn(np.array(x), intf)
                    assert ok and k == (elig[-1] if lr == 1 else elig[0])
                    n_checked += 1
                break
    assert n_checked > 100
