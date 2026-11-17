"""A path that skips the window of an ensemble in one step (no frame in the window, so no shooting region) is still
a valid path of that ensemble: it has the weight given by its pptype (as in `calc_cv_vector`) and gets there through a
swap. It can not be shot from: `staple_sh` rejects the move ("NSG") and `staple_swap` must not reject the swap
because of the missing region (that would make the weights of the permanent and of the swaps inconsistent)."""
import numpy as np

from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.core.tis import calc_cv_vector, staple_sh, staple_swap

INTFS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
# A -> B, the step 0.8 -> 3.3 skips the window (1, 2, 3) of ensemble 2 (key 2 in `picked`)
SKIP = [-0.2, 0.5, 0.8, 3.3, 3.6, 4.4, 5.2]
# the same, but it visited the window before (the region search used to raise UnboundLocalError)
SKIP_VISITED = [-0.2, 0.5, 1.3, 0.9, 3.4, 4.4, 5.2]
NORMAL = [-0.2, 0.5, 1.5, 2.5, 3.5, 4.5, 5.2]


def _path(orders, number=1):
    p = StaplePath(maxlen=100)
    for v in orders:
        f = System()
        f.order = [v]
        p.phasepoints.append(f)
    p.path_number = number
    return p


def _ens(idx):
    """Ensemble settings as numbered in `picked` (`[0+]` is 0, so ens_name is idx + 1)."""
    return {
        "ens_name": f"{idx + 1:03d}",
        "interfaces": tuple(INTFS[idx - 1:idx + 2]) if idx > 0 else (INTFS[0], INTFS[0], INTFS[1]),
        "all_intfs": tuple(INTFS),
        "rgen": np.random.default_rng(0),
        "tis_set": {"maxlength": 1000, "staple": True},
    }


def test_get_sh_region_raises_value_error():
    for orders in (SKIP, SKIP_VISITED):
        p = _path(orders)
        try:
            p.get_sh_region(INTFS, [1.0, 2.0, 3.0])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_weight_is_kept_in_the_permanent():
    p = _path(SKIP)
    p.pptype = (1, "LMR")
    assert calc_cv_vector(p, INTFS, ["sh"] * len(INTFS))[2] == 1.0


def test_staple_sh_rejects_without_region():
    for orders in (SKIP, SKIP_VISITED):
        p = _path(orders)
        accept, trial, status = staple_sh(_ens(2), p, engine=None)
        assert accept is False and status == "NSG" and trial.status == "NSG"
        assert trial.length >= 1 and p.length == len(orders)


def test_staple_swap_accepts_path_without_region():
    # ensemble 1 holds a normal path, ensemble 2 the path that skips its window, the swap puts the latter in ensemble 1
    # (where it has a shooting region) and the former in ensemble 2
    p1 = _path(NORMAL, 1)
    p1.pptype = (1, "LMR")
    p2 = _path(SKIP, 2)
    p2.pptype = (2, "LMR")
    picked = {
        1: {"ens": _ens(1), "traj": p1},
        2: {"ens": _ens(2), "traj": p2},
    }
    accept, paths, status = staple_swap(picked, {})
    assert accept and status == "ACC"
    assert paths[0].pptype == (1, "LMR") and paths[1].pptype == (2, "LMR")


def test_staple_swap_into_ensemble_without_region():
    # now the skipping path goes into ensemble 2 (no shooting region there) instead of out of it
    p1 = _path(SKIP, 1)
    p1.pptype = (1, "LMR")
    p2 = _path(NORMAL, 2)
    p2.pptype = (2, "LMR")
    picked = {
        1: {"ens": _ens(1), "traj": p1},
        2: {"ens": _ens(2), "traj": p2},
    }
    accept, paths, status = staple_swap(picked, {})
    assert accept and status == "ACC"
    assert paths[1].pptype == (2, "LMR") and paths[1].phasepoints[3].order[0] == 3.3
    assert 2 not in paths[1].sh_region                      # nothing stored for the ensemble without region
