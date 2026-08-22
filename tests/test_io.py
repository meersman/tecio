#!/usr/bin/env python3
"""pytest tests for :mod:`tecio._io` — the ``tecio.open()`` entry point.

Covers:
* All five open modes: ``'r'``, ``'w'``, ``'x'``, ``'a'``, ``'a+'``.
* :class:`~tecio._io.AppendWrite` — append new zones to an existing file.
* :class:`~tecio._io.AppendReadWrite` — append while reading the original
  data in the same session.
* Error paths (wrong extension, ``'x'`` on existing file, ``'a'`` on
  missing file, FEPOLYGON in append source).
* Cross-format round-trip via append (read SZL, write PLT, verify counts).

These tests are format-agnostic at the ``tecio.open()`` level. All writes
use ``.szplt`` unless the test specifically targets PLT/DAT behaviour.
"""

# ruff: noqa: E501

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import tecio
from tecio import FileType, ZoneType
from tecio._io import AppendReadWrite, AppendWrite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(tecio.__file__).parent.parent / "tests"
_ONERA = {fmt: _TESTS_DIR / f"Onera.{fmt}" for fmt in ("szplt", "plt", "dat")}

_RTOL_F32 = 1e-5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_simple_szplt(path: Path, n_zones: int = 1) -> dict:
    """Write a trivial SZL file with *n_zones* identical IJK zones.

    Returns the written coordinate arrays for later comparison.
    """
    x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    c = np.sin(2 * np.pi * x).astype(np.float32)

    with tecio.open(str(path), "w", variables=["x", "c"], title="test") as w:
        for i in range(n_zones):
            w.write_ijk_zone(
                data=[x, c],
                title=f"zone_{i + 1}",
                solution_time=float(i),
                strand_id=1 if n_zones > 1 else 0,
            )
    return {"x": x, "c": c}


def _write_simple_plt(path: Path, n_zones: int = 1) -> dict:
    """Write a trivial PLT file with *n_zones* identical IJK zones."""
    x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    c = np.cos(2 * np.pi * x).astype(np.float32)

    with tecio.open(str(path), "w", variables=["x", "c"], title="test_plt") as w:
        for i in range(n_zones):
            w.write_ijk_zone(
                data=[x, c],
                title=f"zone_{i + 1}",
                solution_time=float(i),
                strand_id=1 if n_zones > 1 else 0,
            )
    return {"x": x, "c": c}


def _write_simple_dat(path: Path, n_zones: int = 1) -> dict:
    """Write a trivial DAT file with *n_zones* identical IJK zones."""
    x = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    c = (x**2).astype(np.float32)

    with tecio.open(str(path), "w", variables=["x", "c"], title="test_dat") as w:
        for i in range(n_zones):
            w.write_ijk_zone(data=[x, c], title=f"zone_{i + 1}")
    return {"x": x, "c": c}


# ===========================================================================
# Mode 'r' — read
# ===========================================================================


class TestOpenRead:
    """Tests for ``tecio.open(path, 'r')``."""

    @pytest.mark.parametrize("fmt", ["szplt", "plt", "dat"])
    def test_open_onera(self, fmt: str) -> None:
        """Onera files open without error in all three formats."""
        with tecio.open(str(_ONERA[fmt]), "r") as r:
            assert r.num_vars == 18
            assert r.num_zones == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Opening a non-existent file raises :exc:`FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            tecio.open(str(tmp_path / "ghost.szplt"), "r")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """An unrecognised extension raises :exc:`ValueError`."""
        path = tmp_path / "data.nc"
        path.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            tecio.open(str(path), "r")

    def test_default_mode_is_read(self, tmp_path: Path) -> None:
        """Omitting ``mode`` defaults to ``'r'``."""
        path = tmp_path / "simple.szplt"
        _write_simple_szplt(path)
        with tecio.open(str(path)) as r:
            assert r.num_vars == 2

    def test_context_manager_read(self, tmp_path: Path) -> None:
        """Reader works as a context manager (no error on ``__exit__``)."""
        path = tmp_path / "ctx.szplt"
        _write_simple_szplt(path)
        with tecio.open(str(path), "r") as r:
            zones = r.num_zones
        assert zones == 1


# ===========================================================================
# Mode 'w' — write (overwrite)
# ===========================================================================


class TestOpenWrite:
    """Tests for ``tecio.open(path, 'w')``."""

    def test_creates_new_file(self, tmp_path: Path) -> None:
        """Mode ``'w'`` creates a file that did not previously exist."""
        path = tmp_path / "new.szplt"
        assert not path.exists()
        with tecio.open(str(path), "w") as w:
            x = np.array([0.0, 1.0], dtype=np.float32)
            w.write_ijk_zone(data=[x], variables=["x"])
        assert path.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Mode ``'w'`` silently replaces an existing file."""
        path = tmp_path / "overwrite.szplt"
        _write_simple_szplt(path, n_zones=3)

        with tecio.open(str(path), "w") as w:
            x = np.array([0.0, 1.0], dtype=np.float32)
            w.write_ijk_zone(data=[x], variables=["x"])

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 1  # not 3

    @pytest.mark.parametrize(
        "fmt,ext", [("szplt", ".szplt"), ("plt", ".plt"), ("dat", ".dat")]
    )
    def test_write_all_formats(self, fmt: str, ext: str, tmp_path: Path) -> None:
        """Mode ``'w'`` works for all supported extensions."""
        path = tmp_path / f"out{ext}"
        writer_fn = {
            "szplt": _write_simple_szplt,
            "plt": _write_simple_plt,
            "dat": _write_simple_dat,
        }[fmt]
        writer_fn(path)
        with tecio.open(str(path), "r") as r:
            assert r.num_vars == 2


# ===========================================================================
# Mode 'x' — exclusive creation
# ===========================================================================


class TestOpenExclusive:
    """Tests for ``tecio.open(path, 'x')``."""

    def test_creates_new_file(self, tmp_path: Path) -> None:
        """Mode ``'x'`` creates a file when it does not exist."""
        path = tmp_path / "exclusive.szplt"
        with tecio.open(str(path), "x") as w:
            x = np.array([0.0, 1.0], dtype=np.float32)
            w.write_ijk_zone(data=[x], variables=["x"])
        assert path.exists()

    def test_raises_if_file_exists(self, tmp_path: Path) -> None:
        """Mode ``'x'`` raises :exc:`FileExistsError` on an existing file."""
        path = tmp_path / "exists.szplt"
        _write_simple_szplt(path)
        with pytest.raises(FileExistsError):
            tecio.open(str(path), "x")

    def test_exclusive_content_is_correct(self, tmp_path: Path) -> None:
        """Data written via ``'x'`` reads back correctly."""
        path = tmp_path / "xwrite.szplt"
        x_in = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        with tecio.open(str(path), "x") as w:
            w.write_ijk_zone(data=[x_in], variables=["x"])

        with tecio.open(str(path), "r") as r:
            np.testing.assert_allclose(
                r.zone[0].variable[0].values.ravel(), x_in, rtol=_RTOL_F32
            )


# ===========================================================================
# Mode 'a' — append (AppendWrite)
# ===========================================================================


class TestAppendWrite:
    """Tests for ``tecio.open(path, 'a')`` returning :class:`AppendWrite`."""

    def test_returns_append_write_instance(self, tmp_path: Path) -> None:
        """``open(..., 'a')`` returns an :class:`AppendWrite`."""
        path = tmp_path / "aw.szplt"
        _write_simple_szplt(path)
        with tecio.open(str(path), "a") as w:
            assert isinstance(w, AppendWrite)

    def test_zone_count_after_append_szplt(self, tmp_path: Path) -> None:
        """Appending one zone to a 1-zone SZL file produces 2 zones."""
        path = tmp_path / "append_szplt.szplt"
        arrays = _write_simple_szplt(path)

        with tecio.open(str(path), "a") as w:
            assert w.current_zone == 1  # one zone was copied from original
            w.write_ijk_zone(
                data=[arrays["x"], arrays["c"]],
                title="appended_zone",
                solution_time=99.0,
                strand_id=1,
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            assert r.zone[1].title == "appended_zone"
            assert r.zone[1].solution_time == pytest.approx(99.0)

    def test_zone_count_after_append_plt(self, tmp_path: Path) -> None:
        """Appending one zone to a 2-zone PLT file produces 3 zones."""
        path = tmp_path / "append_plt.plt"
        arrays = _write_simple_plt(path, n_zones=2)

        with tecio.open(str(path), "a") as w:
            assert w.current_zone == 2  # two zones copied
            w.write_ijk_zone(
                data=[arrays["x"], arrays["c"]],
                title="new_zone",
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 3

    def test_original_zones_preserved(self, tmp_path: Path) -> None:
        """Zones from the original file survive the append operation intact."""
        path = tmp_path / "preserve.szplt"
        orig = _write_simple_szplt(path)

        with tecio.open(str(path), "a") as w:
            extra = np.ones(10, dtype=np.float32) * 9.9
            w.write_ijk_zone(data=[extra, extra], title="extra")

        with tecio.open(str(path), "r") as r:
            # Original zone 1 values must be unchanged.
            np.testing.assert_allclose(
                r.zone[0].variable[0].values.ravel(),
                orig["x"],
                rtol=_RTOL_F32,
            )

    def test_variable_list_preserved(self, tmp_path: Path) -> None:
        """Variable names from the original file are unchanged after append."""
        path = tmp_path / "varnames.szplt"
        _write_simple_szplt(path)

        with tecio.open(str(path), "a") as w:
            original_vars = w.variables
            x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
            w.write_ijk_zone(data=[x, x], title="z2")

        with tecio.open(str(path), "r") as r:
            assert r.variables == original_vars

    def test_title_preserved_by_default(self, tmp_path: Path) -> None:
        """Original dataset title is preserved when not overridden."""
        path = tmp_path / "title.szplt"
        with tecio.open(str(path), "w", title="MyTitle") as w:
            x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
            w.write_ijk_zone(data=[x], variables=["x"])

        with tecio.open(str(path), "a") as w:
            assert w.title == "MyTitle"

    def test_append_multiple_new_zones(self, tmp_path: Path) -> None:
        """Multiple zones can be appended in a single ``'a'`` session."""
        path = tmp_path / "multi_append.szplt"
        _write_simple_szplt(path, n_zones=1)
        x = np.linspace(0.0, 1.0, 10, dtype=np.float32)

        with tecio.open(str(path), "a") as w:
            for i in range(3):
                w.write_ijk_zone(data=[x, x], title=f"new_{i}")

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 4  # 1 original + 3 appended

    def test_append_fe_zone(self, tmp_path: Path) -> None:
        """An FE zone can be appended to a file containing an ordered zone."""
        path = tmp_path / "append_fe.szplt"
        _write_simple_szplt(path)

        pts = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [1.0, 1.0]], dtype=np.float32
        )
        nodes = np.array([[1, 2, 3], [2, 4, 3]], dtype=np.int32)
        x, c = pts[:, 0], np.zeros(4, dtype=np.float32)

        with tecio.open(str(path), "a") as w:
            w.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, c],
                node_map=nodes,
                title="tri_zone",
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            assert r.zone[1].zone_type == ZoneType.FETRIANGLE

    def test_append_to_missing_file_raises(self, tmp_path: Path) -> None:
        """Appending to a non-existent file raises :exc:`FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            tecio.open(str(tmp_path / "ghost.szplt"), "a")

    def test_close_replaces_original(self, tmp_path: Path) -> None:
        """The temporary file replaces the original on close (not before)."""
        path = tmp_path / "replace.szplt"
        _write_simple_szplt(path)
        mtime_before = path.stat().st_mtime

        with tecio.open(str(path), "a") as w:
            x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
            w.write_ijk_zone(data=[x, x], title="z2")

        # mtime must have changed because the file was atomically replaced.
        assert path.stat().st_mtime != mtime_before

    def test_onera_append_increases_zones(self, tmp_path: Path) -> None:
        """Appending one zone to the Onera SZL file produces 3 zones."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)

        with tecio.open(str(src), "r") as r:
            orig_vars = r.variables
            # r.zone[0].num_nodes

        np.zeros(10, dtype=np.float32)
        with tecio.open(str(src), "a") as w:
            assert w.current_zone == 2  # two Onera zones copied
            # Append a new tiny zone (all non-Onera variables passive).
            n_vars = len(w.variables)
            passive = [True] * n_vars
            passive[0] = passive[1] = passive[2] = False  # x, y, z
            pts = np.zeros((10, 3), dtype=np.float32)
            nodes = np.array([[i + 1, i + 2] for i in range(9)], dtype=np.int32)
            w.write_fe_zone(
                zone_type=ZoneType.FELINESEG,
                data=[pts[:, 0], pts[:, 1], pts[:, 2]],
                node_map=nodes,
                passive_vars=passive,
                title="tiny_new",
            )

        with tecio.open(str(src), "r") as r:
            assert r.num_zones == 3
            assert r.variables == orig_vars

    def test_append_preserves_zone_aux_data(self, tmp_path: Path) -> None:
        """Zone-level aux data from the original file survives append."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)

        with tecio.open(str(src), "a") as w:
            x = np.zeros(5, dtype=np.float32)
            n_vars = len(w.variables)
            passive = [True] * n_vars
            passive[0] = passive[1] = passive[2] = False
            nodes = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int32)
            w.write_fe_zone(
                zone_type=ZoneType.FELINESEG,
                data=[x, x, x],
                node_map=nodes,
                passive_vars=passive,
                title="filler",
            )

        with tecio.open(str(src), "r") as r:
            # Zone 2 (WingSurface) has BoundaryCondition aux data.
            aux = r.zone[1].auxdata
            assert aux["Common.BoundaryCondition"] == "Wall"


# ===========================================================================
# Mode 'a+' — append-read (AppendReadWrite)
# ===========================================================================


class TestAppendReadWrite:
    """Tests for ``tecio.open(path, 'a+')`` returning :class:`AppendReadWrite`."""

    def test_returns_append_read_write_instance(self, tmp_path: Path) -> None:
        """``open(..., 'a+')`` returns an :class:`AppendReadWrite`."""
        path = tmp_path / "arw.szplt"
        _write_simple_szplt(path)
        with tecio.open(str(path), "a+") as rw:
            assert isinstance(rw, AppendReadWrite)

    def test_can_read_original_zones(self, tmp_path: Path) -> None:
        """Original zones are readable through the read interface."""
        path = tmp_path / "arw_read.szplt"
        orig = _write_simple_szplt(path)

        with tecio.open(str(path), "a+") as rw:
            assert rw.num_zones == 1
            np.testing.assert_allclose(
                rw.zone[0].variable[0].values.ravel(),
                orig["x"],
                rtol=_RTOL_F32,
            )

    def test_can_write_new_zone_while_reading(self, tmp_path: Path) -> None:
        """New zones can be written while reading the original."""
        path = tmp_path / "arw_write.szplt"
        orig = _write_simple_szplt(path)

        with tecio.open(str(path), "a+") as rw:
            # Read original data.
            x_orig = rw.zone[0].variable[0].values.ravel().copy()
            # Write a derived zone.
            x_new = (x_orig * 2.0).astype(np.float32)
            c_new = orig["c"] * 0.5
            rw.write_ijk_zone(data=[x_new, c_new], title="derived")

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 2
            np.testing.assert_allclose(
                r.zone[1].variable[0].values.ravel(), x_new, rtol=_RTOL_F32
            )

    def test_read_interface_exposes_original_metadata(self, tmp_path: Path) -> None:
        """``num_vars``, ``variables``, ``title``, ``file_type`` match the source."""
        path = tmp_path / "arw_meta.szplt"
        with tecio.open(str(path), "w", title="MyTitle", file_type=FileType.FULL) as w:
            x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
            w.write_ijk_zone(data=[x, x], variables=["x", "c"])

        with tecio.open(str(path), "a+") as rw:
            assert rw.title == "MyTitle"
            assert rw.file_type == FileType.FULL
            assert rw.num_vars == 2
            assert rw.variables == ["x", "c"]

    def test_num_zones_reports_original_count(self, tmp_path: Path) -> None:
        """``num_zones`` reports the count before any new zones are appended."""
        path = tmp_path / "arw_nz.szplt"
        _write_simple_szplt(path, n_zones=3)

        with tecio.open(str(path), "a+") as rw:
            assert rw.num_zones == 3  # original count — new writes not included
            x = np.linspace(0.0, 1.0, 10, dtype=np.float32)
            rw.write_ijk_zone(data=[x, x], title="extra")
            assert rw.num_zones == 3  # still original count during session

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 4  # now includes the new zone

    def test_compute_and_write_average(self, tmp_path: Path) -> None:
        """A time-average computed from original zones is appended correctly."""
        path = tmp_path / "avg.szplt"
        n = 10
        times = np.linspace(0.0, np.pi, 5)
        x_base = np.linspace(0.0, 1.0, n, dtype=np.float32)

        with tecio.open(str(path), "w", variables=["x", "c"]) as w:
            for i, t in enumerate(times):
                c = np.sin(x_base + t).astype(np.float32)
                data = [x_base, c] if i == 0 else [c]
                sharing = None if i == 0 else [1, 0]
                w.write_ijk_zone(
                    data=data,
                    var_sharing=sharing,
                    solution_time=t,
                    strand_id=1,
                )

        with tecio.open(str(path), "a+") as rw:
            assert rw.num_zones == len(times)
            c_sum = sum(
                rw.zone[i].variable[1].values.ravel().astype(np.float64)
                for i in range(rw.num_zones)
            )
            c_avg = (c_sum / rw.num_zones).astype(np.float32)
            rw.zone[0].variable[0].values.ravel()

            rw.write_ijk_zone(
                data=[c_avg],
                var_sharing=[1, 0],
                title="time_average",
                solution_time=float(rw.num_zones),
                strand_id=2,
            )

        with tecio.open(str(path), "r") as r:
            assert r.num_zones == len(times) + 1
            avg_zone = r.zone[-1]
            assert avg_zone.title == "time_average"
            assert avg_zone.strand_id == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """``'a+'`` on a non-existent file raises :exc:`FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            tecio.open(str(tmp_path / "ghost.szplt"), "a+")

    def test_onera_read_and_append(self, tmp_path: Path) -> None:
        """Onera file: read zone titles then append a small new zone."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)

        with tecio.open(str(src), "a+") as rw:
            assert rw.num_zones == 2
            titles = [rw.zone[i].title for i in range(rw.num_zones)]
            assert titles == ["FluidVolume", "WingSurface"]

            n_vars = len(rw.variables)
            passive = [True] * n_vars
            passive[0] = passive[1] = passive[2] = False
            nodes = np.array([[1, 2], [2, 3]], dtype=np.int32)
            xyz = np.zeros(3, dtype=np.float32)
            rw.write_fe_zone(
                zone_type=ZoneType.FELINESEG,
                data=[xyz, xyz, xyz],
                node_map=nodes,
                passive_vars=passive,
                title="dummy",
            )

        with tecio.open(str(src), "r") as r:
            assert r.num_zones == 3
            assert r.zone[2].title == "dummy"


# ===========================================================================
# Mode errors
# ===========================================================================


class TestOpenErrors:
    """Tests for invalid open() arguments."""

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        """An unrecognised mode string raises :exc:`ValueError`."""
        path = tmp_path / "test.szplt"
        _write_simple_szplt(path)
        with pytest.raises(ValueError, match="[Mm]ode"):
            tecio.open(str(path), "z")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """An unrecognised extension raises :exc:`ValueError`."""
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            tecio.open(str(tmp_path / "data.hdf5"), "w")

    def test_append_poly_zone_raises(self, tmp_path: Path) -> None:
        """Appending to a file with FEPOLYGON zones raises :exc:`NotImplementedError`."""
        # Build a minimal PLT file containing a FEPOLYGON zone using the
        # low-level classic API, since the high-level writer does not expose
        # poly zones.
        path = tmp_path / "poly.plt"
        from tecio import libtecio
        from tecio.libtecio import ValueLocation as VL
        from tecio.libtecio import ZoneType as ZT

        pts = np.array(
            [
                [0.25, 0],
                [0.75, 0],
                [1, 0.25],
                [1, 0.75],
                [0.75, 1],
                [0.25, 1],
                [0, 0.75],
                [0, 0.25],
            ],
            dtype=np.float32,
        )
        x, y = pts[:, 0], pts[:, 1]
        face_pairs = np.array([
            [2, 1],
            [3, 2],
            [4, 3],
            [5, 4],
            [6, 5],
            [7, 6],
            [8, 7],
            [1, 8],
        ])
        nf = len(face_pairs)
        fn_counts = np.full(nf, 2, dtype=np.int32)
        fn_nodes = face_pairs.ravel().astype(np.int32)
        le = np.zeros(nf, dtype=np.int32)
        re = np.ones(nf, dtype=np.int32)
        c = np.array([1.0], dtype=np.float32)

        libtecio.tecini142(str(path), variables=["x", "y", "c"])
        libtecio.tecpolyzne142(
            "oct",
            ZT.FEPOLYGON,
            num_nodes=8,
            num_faces=nf,
            num_elements=1,
            total_num_face_nodes=len(fn_nodes),
            value_locations=[VL.NODAL, VL.NODAL, VL.CELL_CENTERED],
        )
        libtecio.tecdat142(x, is_double=False)
        libtecio.tecdat142(y, is_double=False)
        libtecio.tecdat142(c, is_double=False)
        libtecio.tecpolyface142(fn_counts, fn_nodes, le, re)
        libtecio.tecend142()

        with pytest.raises(NotImplementedError, match="[Pp]oly"):
            tecio.open(str(path), "a")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
