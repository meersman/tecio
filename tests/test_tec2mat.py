#!/usr/bin/env python3
"""pytest test suite for the tec2mat CLI tool.

``tec2mat`` converts a Tecplot file to a MATLAB ``.mat`` file via
:func:`scipy.io.savemat` (one input file -> one output file).  These tests
cover the output structure (``info`` struct + ``zone_<n>`` structs), dtype
preservation, the ``-c`` / ``--oned-as`` options, the file-naming and
force/overwrite behaviour, and the passive/shared/active variable handling.

Onera test file contents (confirmed via tecdump):
    Zones  : 2
        Zone 1 — FluidVolume   FEBRICK          46 417 nodes / 43 008 elements
        Zone 2 — WingSurface   FEQUADRILATERAL   1 453 nodes /  1 408 elements
    Variables : 18 (x, y, z, Density, ... Eddy_Viscosity)
    Both zones: static, all variables NODAL and active (no passive / shared /
    cell-centred).  Both zones are FE, so both carry a node map.

Design notes:
    - SciPy backs tec2mat, so the whole module is skipped via
      ``pytest.importorskip`` when the ``mat`` extra is not installed.
    - ``main()`` returns int; error-path assertions use ``assert ret == 1``.
    - File-writing tests use pytest's ``tmp_path`` fixture; the ``tests/``
      directory is never modified.
    - The ``onera_path`` fixture (from conftest.py) is parametrised over all
      three formats; single-format tests reference ``_ONERA["szplt"]``.
    - Onera has no passive / shared variables, so those code paths are
      exercised with a small synthetic ASCII DAT written by ``_write_synthetic``
      (zone 2 shares variable 1 from zone 1 and marks variable 2 passive).
    - dtype preservation is checked against what the reader returns (DAT yields
      float64; PLT / SZL yield the on-disk float32), not a hard-coded type.

Test files required in ``tests/``:
    Onera.szplt  Onera.plt  Onera.dat
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tecio
from tecio.cli.tec2mat import main as tec2mat_main

# SciPy is required by tec2mat; skip the whole module if the mat extra is absent.
sio = pytest.importorskip("scipy.io")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_DIR = Path(tecio.__file__).parent.parent / "tests"
_FORMATS = ["szplt", "plt", "dat"]
_ONERA: dict[str, Path] = {fmt: _TEST_DIR / f"Onera.{fmt}" for fmt in _FORMATS}

_NUM_ZONES = 2
_NUM_VARS = 18

# Per-zone (nodes, elements, nodes_per_cell) — FluidVolume FEBrick, WingSurface FEQuad.
_Z1_NODES, _Z1_ELEMS, _Z1_NPC = 46417, 43008, 8
_Z2_NODES, _Z2_ELEMS, _Z2_NPC = 1453, 1408, 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_mat(path: Path, *, squeeze: bool = True) -> dict[str, Any]:
    """Load a ``.mat`` file as MATLAB structs (``mat_struct`` objects)."""
    return sio.loadmat(str(path), struct_as_record=False, squeeze_me=squeeze)


def _struct(container: dict[str, Any], name: str) -> Any:
    """Return struct *name*, unwrapping the 1x1 object array left by squeeze=False."""
    obj = container[name]
    if isinstance(obj, np.ndarray):
        obj = obj.reshape(-1)[0]
    return obj


def _write_synthetic(path: Path) -> None:
    """Write a small 2-zone ASCII DAT exercising passive and shared variables.

    Zone 1 (``Z1``) has all three variables active.  Zone 2 (``Z2``) shares
    variable 1 from zone 1, marks variable 2 passive, and keeps variable 3
    active.  Each active array has four values.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    p = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    p2 = np.array([20.0, 21.0, 22.0, 23.0], dtype=np.float64)
    with tecio.open(str(path), "w", title="syn", variables=["x", "y", "p"]) as w:
        w.write_ijk_zone(data=[x, y, p], title="Z1")
        w.write_ijk_zone(
            data=[p2],
            title="Z2",
            passive_vars=[False, True, False],
            var_sharing=[1, 0, 0],
        )


# ===========================================================================
# tec2mat
# ===========================================================================


class TestTec2mat:
    """Tests for tec2mat — converts a Tecplot file to a MATLAB .mat file."""

    # -- File naming and basic output ---------------------------------------

    def test_default_output_name(self, onera_path: Path, tmp_path: Path) -> None:
        """With no -o, output is <stem>.mat next to the input file."""
        src = tmp_path / onera_path.name
        shutil.copy(onera_path, src)
        assert tec2mat_main([str(src)]) == 0
        assert (tmp_path / "Onera.mat").exists()

    def test_explicit_output_path(self, onera_path: Path, tmp_path: Path) -> None:
        """-o writes the .mat to the requested path."""
        dst = tmp_path / "out.mat"
        assert tec2mat_main(["-o", str(dst), str(onera_path)]) == 0
        assert dst.exists()
        assert dst.stat().st_size > 0

    # -- info struct --------------------------------------------------------

    def test_info_struct(self, onera_path: Path, tmp_path: Path) -> None:
        """The info struct mirrors the dataset title, counts, and variable names."""
        dst = tmp_path / "out.mat"
        assert tec2mat_main(["-o", str(dst), str(onera_path)]) == 0
        info = _struct(_load_mat(dst), "info")
        assert int(info.num_zones) == _NUM_ZONES
        assert int(info.num_vars) == _NUM_VARS
        names = [str(v) for v in np.atleast_1d(info.var_names)]
        with tecio.open(str(onera_path), "r") as r:
            assert str(info.title) == r.title
            assert str(info.file_type) == r.file_type.name
            assert names == r.variables

    # -- zone structs -------------------------------------------------------

    def test_zone_structs_present(self, onera_path: Path, tmp_path: Path) -> None:
        """One struct per zone is emitted, 1-based, with no extras."""
        dst = tmp_path / "out.mat"
        tec2mat_main(["-o", str(dst), str(onera_path)])
        d = _load_mat(dst)
        assert "zone_1" in d
        assert "zone_2" in d
        assert "zone_3" not in d

    def test_zone_metadata(self, onera_path: Path, tmp_path: Path) -> None:
        """Zone titles, types, and node/element counts are preserved."""
        dst = tmp_path / "out.mat"
        tec2mat_main(["-o", str(dst), str(onera_path)])
        d = _load_mat(dst)
        z1 = _struct(d, "zone_1")
        z2 = _struct(d, "zone_2")
        assert str(z1.title) == "FluidVolume"
        assert str(z1.zone_type) == "FEBRICK"
        assert int(z1.num_nodes) == _Z1_NODES
        assert int(z1.num_elements) == _Z1_ELEMS
        assert str(z2.title) == "WingSurface"
        assert str(z2.zone_type) == "FEQUADRILATERAL"
        assert int(z2.num_nodes) == _Z2_NODES
        assert int(z2.num_elements) == _Z2_ELEMS

    def test_node_map_for_fe_zones(self, onera_path: Path, tmp_path: Path) -> None:
        """Both FE zones carry a 1-based (num_elements x nodes_per_cell) node map."""
        dst = tmp_path / "out.mat"
        tec2mat_main(["-o", str(dst), str(onera_path)])
        d = _load_mat(dst)
        z1 = _struct(d, "zone_1")
        z2 = _struct(d, "zone_2")
        assert z1.node_map.shape == (_Z1_ELEMS, _Z1_NPC)
        assert z2.node_map.shape == (_Z2_ELEMS, _Z2_NPC)
        assert int(z1.node_map.min()) >= 1
        assert int(z2.node_map.min()) >= 1

    def test_variable_arrays_match_node_count(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Every active nodal variable has one value per node."""
        dst = tmp_path / "out.mat"
        tec2mat_main(["-o", str(dst), str(onera_path)])
        z1 = _struct(_load_mat(dst), "zone_1")
        for k in range(1, _NUM_VARS + 1):
            assert getattr(z1, f"var_{k}").size == _Z1_NODES

    def test_metadata_arrays_active_nodal(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Onera variables are all active, nodal, and unshared."""
        dst = tmp_path / "out.mat"
        tec2mat_main(["-o", str(dst), str(onera_path)])
        z1 = _struct(_load_mat(dst), "zone_1")
        assert [str(s) for s in np.atleast_1d(z1.var_status)] == ["active"] * _NUM_VARS
        assert [str(s) for s in np.atleast_1d(z1.var_locations)] == [
            "NODAL"
        ] * _NUM_VARS
        assert np.array_equal(np.atleast_1d(z1.var_shared_from), [0] * _NUM_VARS)

    # -- dtype preservation -------------------------------------------------

    def test_dtype_preserved(self, onera_path: Path, tmp_path: Path) -> None:
        """Each stored array keeps the reader's on-disk dtype (DAT=double, PLT/SZL=single)."""  # noqa: E501
        dst = tmp_path / "out.mat"
        tec2mat_main(["-o", str(dst), str(onera_path)])
        z1 = _struct(_load_mat(dst), "zone_1")
        with tecio.open(str(onera_path), "r") as r:
            zone0 = r.zone[0]
            for k in range(1, _NUM_VARS + 1):
                vals = zone0.variable[k - 1].values
                assert vals is not None
                assert getattr(z1, f"var_{k}").dtype == vals.dtype

    # -- options: --compress -----------------------------------------------

    def test_compress_reduces_size(self, tmp_path: Path) -> None:
        """-c yields a smaller file than the uncompressed conversion."""
        plain = tmp_path / "plain.mat"
        comp = tmp_path / "comp.mat"
        assert tec2mat_main(["-o", str(plain), str(_ONERA["szplt"])]) == 0
        assert tec2mat_main(["-c", "-o", str(comp), str(_ONERA["szplt"])]) == 0
        assert comp.stat().st_size < plain.stat().st_size

    def test_compress_roundtrips(self, tmp_path: Path) -> None:
        """A compressed file still loads with the expected structure."""
        dst = tmp_path / "comp.mat"
        assert tec2mat_main(["--compress", "-o", str(dst), str(_ONERA["szplt"])]) == 0
        info = _struct(_load_mat(dst), "info")
        assert int(info.num_zones) == _NUM_ZONES

    # -- options: --oned-as -------------------------------------------------

    def test_oned_as_column_default(self, tmp_path: Path) -> None:
        """1-D FE variable arrays default to N x 1 column vectors."""
        dst = tmp_path / "out.mat"
        assert tec2mat_main(["-o", str(dst), str(_ONERA["szplt"])]) == 0
        z1 = _struct(_load_mat(dst, squeeze=False), "zone_1")
        assert z1.var_1.shape == (_Z1_NODES, 1)

    def test_oned_as_row(self, tmp_path: Path) -> None:
        """--oned-as row stores 1-D FE variable arrays as 1 x N row vectors."""
        dst = tmp_path / "out.mat"
        ret = tec2mat_main(["--oned-as", "row", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        z1 = _struct(_load_mat(dst, squeeze=False), "zone_1")
        assert z1.var_1.shape == (1, _Z1_NODES)

    # -- passive / shared / active (synthetic data) -------------------------

    def test_synthetic_zone1_all_active(self, tmp_path: Path) -> None:
        """In the synthetic file, zone 1 has three active variables with data."""
        src = tmp_path / "syn.dat"
        _write_synthetic(src)
        dst = tmp_path / "syn.mat"
        assert tec2mat_main(["-o", str(dst), str(src)]) == 0
        z1 = _struct(_load_mat(dst), "zone_1")
        assert [str(s) for s in np.atleast_1d(z1.var_status)] == [
            "active",
            "active",
            "active",
        ]
        for k in (1, 2, 3):
            assert getattr(z1, f"var_{k}").size == 4

    def test_synthetic_passive_and_shared(self, tmp_path: Path) -> None:
        """Zone 2: var 1 shared, var 2 passive (both empty), var 3 active."""
        src = tmp_path / "syn.dat"
        _write_synthetic(src)
        dst = tmp_path / "syn.mat"
        assert tec2mat_main(["-o", str(dst), str(src)]) == 0
        z2 = _struct(_load_mat(dst), "zone_2")
        assert [str(s) for s in np.atleast_1d(z2.var_status)] == [
            "shared",
            "passive",
            "active",
        ]
        # Shared and passive variables carry no data (MATLAB []).
        assert np.size(z2.var_1) == 0
        assert np.size(z2.var_2) == 0
        # The active variable retains its data.
        assert np.size(z2.var_3) == 4
        # var 1 is shared from zone 1 (1-based); the others are not shared.
        assert np.array_equal(np.atleast_1d(z2.var_shared_from), [1, 0, 0])

    # -- error and overwrite paths ------------------------------------------

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """A non-existent source file returns exit code 1."""
        assert tec2mat_main([str(tmp_path / "ghost.dat")]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """An existing output without --force returns exit code 1."""
        dst = tmp_path / "out.mat"
        dst.touch()
        assert tec2mat_main(["-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "out.mat"
        dst.touch()
        ret = tec2mat_main(["-f", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert dst.stat().st_size > 0


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
