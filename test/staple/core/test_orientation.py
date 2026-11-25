"""A path with both ends in A that goes above the window of [1*] and back crosses that window twice: it is two
staples with the same frames, 0LMR2 (turn at the end, first pass) and 2RML0 (turn at the start, last pass). The
path's orientation fixes which one it is, so its shooting region does not depend on how it entered an ensemble."""
import numpy as np

from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.core.tis import _set_staple_orientation, staple_swap

INTFS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
# A -> window of [1*] -> above lambda_2 -> window -> A, both lambda_1 crossings are jumps from/into A (path 8242)
DOUBLE = [-0.2, 1.5, 1.6, 1.4, 2.4, 2.6, 2.3, 1.7, 1.5, 1.6, -0.3]
FIRST, LAST = (1, 3), (7, 9)
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
        "interfaces": tuple(INTFS[idx - 1:idx + 2]),
        "all_intfs": tuple(INTFS),
        "rgen": np.random.default_rng(0),
        "tis_set": {"maxlength": 1000, "staple": True},
    }


def test_passes():
    p = _path(DOUBLE)
    assert p.get_sh_passes([0.0, 1.0, 2.0]) == [(1, 3, "LMR"), (7, 9, "RML")]
    assert p.get_sh_passes([1.0, 2.0, 3.0]) == [(1, 9, "LML")]


def test_oriented_region():
    p = _path(DOUBLE)
    p.orientation = "end"
    assert p.get_oriented_sh_region(INTFS, [0.0, 1.0, 2.0]) == (FIRST, "LMR")
    p.orientation = "start"
    assert p.get_oriented_sh_region(INTFS, [0.0, 1.0, 2.0]) == (LAST, "RML")
    # one pass in [2*]: not ambiguous there
    assert p.get_oriented_sh_region(INTFS, [1.0, 2.0, 3.0]) is None
    # a path between A and B is never ambiguous
    q = _path(NORMAL)
    q.orientation = "start"
    assert q.get_oriented_sh_region(INTFS, [0.0, 1.0, 2.0]) is None


def test_undecided_orientation_is_drawn_once():
    p = _path(DOUBLE)
    assert p.get_oriented_sh_region(INTFS, [0.0, 1.0, 2.0]) is None  # no rgen: not drawn
    assert p.orientation is None
    rgen = np.random.default_rng(1)
    first = p.get_oriented_sh_region(INTFS, [0.0, 1.0, 2.0], rgen)
    assert p.orientation in ("start", "end")
    for _ in range(5):
        assert p.get_oriented_sh_region(INTFS, [0.0, 1.0, 2.0], rgen) == first
    assert p.copy().orientation == p.orientation
    # both orientations are drawn
    seen = set()
    for seed in range(20):
        q = _path(DOUBLE)
        q.get_oriented_sh_region(INTFS, [0.0, 1.0, 2.0], np.random.default_rng(seed))
        seen.add(q.orientation)
    assert seen == {"start", "end"}


def test_orientation_set_by_extender():
    # the shot pass fixes it for LMR/RML
    for region, expected in ((LAST, "start"), (FIRST, "end")):
        p = _path(DOUBLE)
        p.sh_region[1] = region
        _set_staple_orientation(p, _ens(1), "RML", None)
        assert p.orientation == expected
    # the extension coin fixes it for LML/RMR
    for rev, expected in ((True, "start"), (False, "end")):
        p = _path(DOUBLE)
        p.sh_region[2] = (1, 9)
        _set_staple_orientation(p, _ens(2), "LML", rev)
        assert p.orientation == expected


def test_swap_into_ensemble_follows_orientation():
    for orientation, region, pptype in (("start", LAST, "RML"), ("end", FIRST, "LMR")):
        p1 = _path(NORMAL, 1)
        p1.pptype = (1, "LMR")
        p2 = _path(DOUBLE, 2)
        p2.pptype = (2, "LML")
        p2.sh_region[2] = (1, 9)
        p2.orientation = orientation
        picked = {
            1: {"ens": _ens(1), "traj": p1},
            2: {"ens": _ens(2), "traj": p2},
        }
        accept, paths, status = staple_swap(picked, {})
        assert accept and status == "ACC"
        assert paths[0].sh_region[1] == region and paths[0].pptype == (1, pptype)
        assert paths[0].orientation == orientation
