"""Tests for the per-ensemble weights of staple paths."""
import pytest

from infretis.core.tis import staple_pptype_weight

N_INTF = 5  # ensembles [0+]=0, [1+]=1, ..., [3+]=3 (the last one)


@pytest.mark.parametrize(
    "pptype, ens_idx, expected",
    [
        ("***", 2, 0.0),
        ("LMR", 2, 1.0),
        ("RML", 2, 1.0),
        # Inner LML/RMR: the extender picks one of two distinct staples.
        ("LML", 2, 0.5),
        ("RMR", 2, 0.5),
        ("RMR", 1, 0.5),
        ("LML", 3, 0.5),
        # The extension ends in A or B at once: only one staple.
        ("LML", 1, 1.0),
        ("RMR", 3, 1.0),
        # [0+] is not a staple ensemble.
        ("LML", 0, 1.0),
    ],
)
def test_staple_pptype_weight(pptype, ens_idx, expected):
    assert staple_pptype_weight(pptype, ens_idx, N_INTF) == expected
