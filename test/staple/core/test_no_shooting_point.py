"""A staple path with no frame to shoot from in its ensemble (empty shooting region, and the cut path from
`get_pp_path` is None) must not kill the run: `staple_sh` rejects the move with status NSG, as the wire fencing
move does without frames. Before, `prepare_shooting_point(None, ...)` raised AttributeError in the worker
(`'NoneType' object has no attribute 'get_shooting_point'`) and the scheduler stopped (seen in a 50k-cycle Z run:
'Invalid shooting region: {0: (1, 0)} for path length 15')."""
import numpy as np
import pytest

from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.core.tis import staple_sh

ALL_INTFS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _path(orders):
    p = StaplePath()
    for v in orders:
        f = System()
        f.order = [v]
        p.phasepoints.append(f)
    p.path_number = 7
    return p


def _ens_set(name=1):
    return {
        "ens_name": name,
        "interfaces": ALL_INTFS[max(name - 2, 0):name + 2][:3] if name > 1 else ALL_INTFS[:3],
        "all_intfs": ALL_INTFS,
        "rgen": np.random.default_rng(0),
        "tis_set": {"maxlength": 1000},
    }


def test_empty_region_without_cut_path_is_rejected_not_fatal(monkeypatch):
    path = _path([0.05, 0.1, 0.15, 0.1, 0.05])
    ens = _ens_set(1)
    path.sh_region = {0: (1, 0)}                                     # empty region, as in the failed run
    monkeypatch.setattr(path, "get_sh_region", lambda *a, **k: (1, 0))
    monkeypatch.setattr(path, "get_pp_path", lambda *a, **k: None)   # the fallback finds nothing
    accept, trial, status = staple_sh(ens, path, engine=None)
    assert accept is False and status == "NSG" and trial.status == "NSG"
    # what run_md reads from every trial path (a rejected move with an empty trial path crashed it)
    assert trial.length >= 1 and np.isfinite(trial.ordermin[0]) and np.isfinite(trial.ordermax[0])
    assert trial.generated[0] == "st_sh"
    # the old path is untouched
    assert path.length == 5 and path.sh_region == {0: (1, 0)}


def test_usable_cut_path_still_used(monkeypatch):
    """When get_pp_path returns a path that has a shooting region, the fallback still shoots from it (no NSG)."""
    bad = _path([0.05, 0.1, 0.15, 0.1, 0.05])
    bad.sh_region = {0: (1, 0)}
    good = _path([0.05, 0.1, 0.15, 0.12, 0.1, 0.05])
    good.sh_region = {0: (1, 3)}
    monkeypatch.setattr(bad, "get_sh_region", lambda *a, **k: (1, 0))
    monkeypatch.setattr(bad, "get_pp_path", lambda *a, **k: good)
    seen = {}

    def fake_prepare(path, rgen, engine, ens_set):
        if path is bad:
            raise ValueError("Invalid shooting region")
        seen["from"] = path
        raise ValueError("stop here")                                # not testing the engine part

    monkeypatch.setattr("infretis.core.tis.prepare_shooting_point", fake_prepare)
    accept, trial, status = staple_sh(_ens_set(1), bad, engine=None)
    assert seen["from"] is good                                       # the cut path was tried first
    assert status == "NSG"                                            # and its own failure is a rejection too
