#!/usr/bin/env python3
"""pytest test suite for all tecio CLI tools.

Onera test file contents (confirmed via tecdump):
    Zones  : 2
        Zone 1 — FluidVolume   FEBRICK        46 417 nodes / 43 008 elements
        Zone 2 — WingSurface   FEQUADRILATERAL  1 453 nodes /  1 408 elements
    Variables : 18
        1 x, 2 y, 3 z, 4 Density, 5 Momentum U, 6 Momentum V, 7 Momentum W,
        8 Energy, 9 SA Turbulent Eddy Viscosity, 10 Pressure, 11 Temperature,
        12 Pressure_Coefficient, 13 Mach, 14 Laminar_Viscosity,
        15 Skin_Friction_Coefficient, 16 Heat_Flux, 17 Y_Plus, 18 Eddy_Viscosity
    Both zones: static (strand 0, solution_time 0.0), all variables NODAL FLOAT.
    Zone 2 has zone-level aux data.

Design notes:
    - All ``main()`` functions return int.  All error-path assertions use
      ``assert ret == 1`` — no ``pytest.raises(SystemExit)`` needed.
      (Requires the tecscale and tecmerge patches that replace sys.exit calls
      in helper functions with return-None / raise-FileNotFoundError.)
    - File-writing tests use pytest's ``tmp_path`` fixture; the ``tests/``
      directory is never modified.
    - ``tecstats -csv`` auto-names the CSV next to the input, so those tests
      copy the source file into ``tmp_path`` first.
    - The ``onera_path`` fixture is parametrised over all three formats;
      single-format tests reference ``_ONERA["szplt"]`` directly.
    - Both Onera zones are FE, so ``tecslice`` IJK flags produce verbatim
      copies with a warning.  Tests verify the warning and zone count.

Test files required in ``tests/``:
    Onera.szplt  Onera.plt  Onera.dat
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import tecio
from tecio.cli.tecdump import main as tecdump_main
from tecio.cli.tecextract import main as tecextract_main
from tecio.cli.tecfix import main as tecfix_main
from tecio.cli.tecmerge import main as tecmerge_main
from tecio.cli.teconvert import main as teconvert_main
from tecio.cli.tecscale import main as tecscale_main
from tecio.cli.tecslice import main as tecslice_main
from tecio.cli.tecstats import main as tecstats_main

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_DIR = Path(tecio.__file__).parent.parent / "tests"
_FORMATS = ["szplt", "plt", "dat"]
_ONERA: dict[str, Path] = {fmt: _TEST_DIR / f"Onera.{fmt}" for fmt in _FORMATS}

_NUM_ZONES = 2
_NUM_VARS = 18


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_readable(path: Path) -> bool:
    """Return True if *path* can be opened as a valid Tecplot file."""
    try:
        with tecio.open(str(path), "r") as r:
            return r.num_vars > 0
    except Exception:
        return False


def _copy_onera(src_fmt: str, dst_dir: Path, name: str) -> Path:
    """Copy an Onera file into *dst_dir* under *name*, preserving extension."""
    src = _ONERA[src_fmt]
    dst = dst_dir / f"{name}{src.suffix}"
    shutil.copy(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def meta() -> dict:
    """Dataset metadata from Onera.szplt, read once per test session."""
    with tecio.open(str(_ONERA["szplt"]), "r") as r:
        return {
            "num_zones": r.num_zones,
            "num_vars": r.num_vars,
            "variables": r.variables,
        }


@pytest.fixture(params=_FORMATS)
def onera_path(request) -> Path:
    """Yield the Onera input file for each supported format."""
    return _ONERA[request.param]


# ===========================================================================
# tecdump
# ===========================================================================


class TestTecdump:
    """Tests for tecdump — prints Tecplot file contents to stdout."""

    def test_basic_output(self, onera_path: Path, capsys) -> None:
        """File header fields appear in stdout; return code is 0."""
        ret = tecdump_main([str(onera_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "File Type" in out
        assert "Dataset Title" in out
        assert "Num Vars" in out
        assert "Num Zones" in out

    def test_zone_record_present(self, onera_path: Path, capsys) -> None:
        """Zone titles FluidVolume and WingSurface appear in the output."""
        ret = tecdump_main([str(onera_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "FluidVolume" in out
        assert "WingSurface" in out

    def test_ignore_zones_flag(self, onera_path: Path, capsys) -> None:
        """--ignore-zones suppresses zone records."""
        ret = tecdump_main(["--ignore-zones", str(onera_path)])
        assert ret == 0
        assert "Zone Record" not in capsys.readouterr().out

    def test_ignore_vars_flag(self, onera_path: Path, capsys) -> None:
        """--ignore-vars suppresses variable value output."""
        ret = tecdump_main(["--ignore-vars", str(onera_path)])
        assert ret == 0
        assert "Values" not in capsys.readouterr().out

    def test_zone_filter(self, onera_path: Path) -> None:
        """-zone 1 limits output to zone 1 without error."""
        assert tecdump_main(["-zone", "1", str(onera_path)]) == 0

    def test_variable_filter(self, onera_path: Path) -> None:
        """-variable 1 limits output to variable 1 without error."""
        assert tecdump_main(["-variable", "1", str(onera_path)]) == 0

    def test_maxvals(self, onera_path: Path) -> None:
        """-maxvals changes the array truncation threshold without error."""
        assert tecdump_main(["-maxvals", "5", str(onera_path)]) == 0


# ===========================================================================
# teconvert
# ===========================================================================


class TestTeconvert:
    """Tests for teconvert — converts between SZL, PLT, and DAT."""

    @pytest.mark.parametrize("src_fmt,dst_flag,dst_ext", [
        ("szplt", "-dat",   ".dat"),
        ("szplt", "-plt",   ".plt"),
        ("plt",   "-dat",   ".dat"),
        ("plt",   "-szplt", ".szplt"),
        ("dat",   "-szplt", ".szplt"),
        ("dat",   "-plt",   ".plt"),
    ])
    def test_convert_between_formats(
        self, src_fmt: str, dst_flag: str, dst_ext: str, tmp_path: Path
    ) -> None:
        """Convert between every supported format pair; output must be readable."""
        dst = tmp_path / f"out{dst_ext}"
        ret = teconvert_main([
            dst_flag, "--force", "-o", str(dst), str(_ONERA[src_fmt])
        ])
        assert ret == 0
        assert dst.exists()
        assert dst.stat().st_size > 0
        assert _is_readable(dst)

    @pytest.mark.parametrize("fmt,flag", [
        ("szplt", "-szplt"),
        ("plt",   "-plt"),
        ("dat",   "-dat"),
    ])
    def test_same_format_noop(self, fmt: str, flag: str) -> None:
        """Same-format conversion warns and returns 0."""
        ret = teconvert_main([flag, str(_ONERA[fmt])])
        assert ret == 0

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert teconvert_main(["-dat", str(tmp_path / "ghost.szplt")]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "out.dat"
        dst.touch()
        assert teconvert_main(["-dat", "-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "out.dat"
        dst.touch()
        ret = teconvert_main(["-dat", "--force", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert dst.stat().st_size > 0


# ===========================================================================
# tecextract
# ===========================================================================


class TestTecextract:
    """Tests for tecextract — extracts a subset of zones and/or variables."""

    def test_extract_all(self, onera_path: Path, tmp_path: Path) -> None:
        """No filter produces a readable verbatim copy."""
        dst = tmp_path / "out.szplt"
        assert tecextract_main(["-o", str(dst), "--force", str(onera_path)]) == 0
        assert _is_readable(dst)

    def test_extract_zone_1_only(self, onera_path: Path, tmp_path: Path) -> None:
        """``-zones 1`` extracts FluidVolume; output has 1 zone."""
        dst = tmp_path / "out.szplt"
        ret = tecextract_main(["-zones", "1", "-o", str(dst), str(onera_path)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 1

    def test_extract_zone_2_only(self, onera_path: Path, tmp_path: Path) -> None:
        """``-zones 2`` extracts WingSurface; output has 1 zone."""
        dst = tmp_path / "out.szplt"
        ret = tecextract_main(["-zones", "2", "-o", str(dst), str(onera_path)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 1

    def test_extract_variable_1_reduces_count(self, tmp_path: Path) -> None:
        """Output is written to DAT format because the SZL C library requires
        variables 1, 2, and 3 (x, y, z) to be present to compute its
        subzone layout — writing a single-variable SZL file is not supported.
        """
        dst = tmp_path / "out.dat"
        ret = tecextract_main([
            "-variables", "1", "-o", str(dst), str(_ONERA["szplt"])
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_vars == 1
            assert r.variables == ["x"]
            
    def test_extract_variable_1_szplt_requires_xyz(self, tmp_path: Path) -> None:
        """Extracting a single variable to SZL fails — the format requires
        variables 1, 2, and 3 (x, y, z) to compute its subzone layout.
        """
        dst = tmp_path / "out.szplt"
        ret = tecextract_main([
            "-variables", "1", "-o", str(dst), str(_ONERA["szplt"])
        ])
        assert ret == 1
    
    def test_extract_multiple_variables(self, tmp_path: Path) -> None:
        """``-variables 1,2,3`` produces a file with exactly 3 variables."""
        dst = tmp_path / "out.szplt"
        ret = tecextract_main([
            "-variables", "1,2,3", "-o", str(dst), str(_ONERA["szplt"])
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_vars == 3
            assert r.variables == ["x", "y", "z"]

    def test_extract_zone_and_variable(self, tmp_path: Path) -> None:
        """Combined zone + variable filter applies both restrictions."""
        dst = tmp_path / "out.szplt"
        ret = tecextract_main([
            "-zones", "2", "-variables", "10",
            "-o", str(dst), str(_ONERA["szplt"]),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == 1
            assert r.num_vars == 1
            assert r.variables == ["Pressure"]

    def test_output_format_controlled_by_extension(self, tmp_path: Path) -> None:
        """Writing to .dat produces a readable ASCII output."""
        dst = tmp_path / "out.dat"
        assert tecextract_main(
            ["-o", str(dst), "--force", str(_ONERA["szplt"])]
        ) == 0
        assert _is_readable(dst)

    def test_zone_out_of_range_returns_1(self, tmp_path: Path) -> None:
        """Zone index beyond num_zones (2) returns exit code 1."""
        assert tecextract_main([
            "-zones", "99", "-o", str(tmp_path / "out.szplt"),
            str(_ONERA["szplt"]),
        ]) == 1

    def test_variable_out_of_range_returns_1(self, tmp_path: Path) -> None:
        """Variable index beyond num_vars (18) returns exit code 1."""
        assert tecextract_main([
            "-variables", "99", "-o", str(tmp_path / "out.szplt"),
            str(_ONERA["szplt"]),
        ]) == 1

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert tecextract_main([
            "-o", str(tmp_path / "out.szplt"),
            str(tmp_path / "ghost.szplt"),
        ]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "out.szplt"
        dst.touch()
        assert tecextract_main(["-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "out.szplt"
        dst.touch()
        ret = tecextract_main(["--force", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert dst.stat().st_size > 0


# ===========================================================================
# tecfix
# ===========================================================================


class TestTecfix:
    """Tests for tecfix — marks NaN / Inf variable arrays as passive."""

    def test_clean_file_produces_copy(
        self, onera_path: Path, tmp_path: Path, capsys
    ) -> None:
        """A file with no NaN/Inf yields a clean copy and a 'clean' message."""
        dst = tmp_path / "fixed.szplt"
        ret = tecfix_main(["-o", str(dst), "--force", str(onera_path)])
        assert ret == 0
        assert _is_readable(dst)
        assert "clean" in capsys.readouterr().out.lower()

    def test_clean_copy_preserves_structure(self, tmp_path: Path) -> None:
        """Fixed file has the same zone and variable counts as the source."""
        dst = tmp_path / "fixed.szplt"
        tecfix_main(["--force", "-o", str(dst), str(_ONERA["szplt"])])
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES
            assert r.num_vars == _NUM_VARS

    def test_dry_run_does_not_write(self, onera_path: Path, tmp_path: Path) -> None:
        """--dry-run scans for issues without creating an output file."""
        dst = tmp_path / "fixed.szplt"
        assert tecfix_main(["--dry-run", str(onera_path)]) == 0
        assert not dst.exists()

    def test_dry_run_clean_message(self, onera_path: Path, capsys) -> None:
        """--dry-run on a clean file reports it is clean."""
        tecfix_main(["--dry-run", str(onera_path)])
        assert "clean" in capsys.readouterr().out.lower()

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert tecfix_main([str(tmp_path / "ghost.szplt")]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "fixed.szplt"
        dst.touch()
        assert tecfix_main(["-o", str(dst), str(_ONERA["szplt"])]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "fixed.szplt"
        dst.touch()
        ret = tecfix_main(["--force", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert dst.stat().st_size > 0


# ===========================================================================
# tecmerge
# ===========================================================================


class TestTecmerge:
    """Tests for tecmerge — merges zones from multiple files into one.

    Design note on deduplication:
        tecmerge._expand_inputs deduplicates resolved paths, so passing the
        same path twice is equivalent to passing it once — intended behaviour
        for glob patterns.  Tests that need N distinct "files" therefore copy
        the source into tmp_path under different names so each resolves to a
        unique path.
    """

    def test_merge_two_distinct_files_doubles_zones(self, tmp_path: Path) -> None:
        """Merging two distinct copies produces a file with 4 zones (2 × 2)."""
        src1 = _copy_onera("szplt", tmp_path, "a")
        src2 = _copy_onera("dat",   tmp_path, "b")
        dst = tmp_path / "merged.szplt"
        ret = tecmerge_main(["-o", str(dst), str(src1), str(src2)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES * 2
            assert r.num_vars == _NUM_VARS

    def test_merge_mixed_formats(self, tmp_path: Path) -> None:
        """SZL + DAT inputs are merged into a single readable output."""
        dst = tmp_path / "merged.szplt"
        ret = tecmerge_main([
            "-o", str(dst), str(_ONERA["szplt"]), str(_ONERA["dat"])
        ])
        assert ret == 0
        assert _is_readable(dst)

    def test_merge_three_formats(self, tmp_path: Path) -> None:
        """Three distinct copies produce 6 zones (3 × 2)."""
        src1 = _copy_onera("szplt", tmp_path, "a")
        src2 = _copy_onera("dat",   tmp_path, "b")
        src3 = _copy_onera("szplt", tmp_path, "c")
        dst = tmp_path / "merged.szplt"
        ret = tecmerge_main(["-o", str(dst), str(src1), str(src2), str(src3)])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES * 3

    def test_assign_time_strands_with_delta(self, tmp_path: Path) -> None:
        """``-delta`` assigns evenly-spaced solution times per input file."""
        src1 = _copy_onera("szplt", tmp_path, "t0")
        src2 = _copy_onera("dat",   tmp_path, "t1")
        dst = tmp_path / "transient.szplt"
        ret = tecmerge_main([
            "--assign-time-strands",
            "-start", "0.0", "-delta", "1.0", "-strand", "1",
            "-o", str(dst), str(src1), str(src2),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            # Zones 0..N-1 from file 0 → time 0.0; zones N..2N-1 from file 1 → 1.0
            assert r.zone[0].solution_time == pytest.approx(0.0)
            assert r.zone[_NUM_ZONES].solution_time == pytest.approx(1.0)

    def test_assign_time_strands_with_end(self, tmp_path: Path) -> None:
        """-end computes the step size automatically from start/end/N."""
        src1 = _copy_onera("szplt", tmp_path, "t0")
        src2 = _copy_onera("dat",   tmp_path, "t1")
        src3 = _copy_onera("szplt", tmp_path, "t2")
        dst = tmp_path / "transient_end.szplt"
        ret = tecmerge_main([
            "--assign-time-strands",
            "-start", "0.0", "-end", "4.0", "-strand", "1",
            "-o", str(dst), str(src1), str(src2), str(src3),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.zone[0].solution_time == pytest.approx(0.0)
            assert r.zone[_NUM_ZONES * 2].solution_time == pytest.approx(4.0)

    def test_missing_start_with_assign_ts_returns_1(self, tmp_path: Path) -> None:
        """--assign-time-strands without -start returns exit code 1."""
        assert tecmerge_main([
            "--assign-time-strands", "-delta", "1.0",
            "-o", str(tmp_path / "out.szplt"), str(_ONERA["szplt"]),
        ]) == 1

    def test_missing_delta_and_end_returns_1(self, tmp_path: Path) -> None:
        """--assign-time-strands without -delta or -end returns exit code 1."""
        assert tecmerge_main([
            "--assign-time-strands", "-start", "0.0",
            "-o", str(tmp_path / "out.szplt"), str(_ONERA["szplt"]),
        ]) == 1

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent input file returns exit code 1."""
        assert tecmerge_main([
            "-o", str(tmp_path / "out.szplt"),
            str(tmp_path / "ghost.szplt"),
        ]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "merged.szplt"
        dst.touch()
        assert tecmerge_main([
            "-o", str(dst),
            str(_ONERA["szplt"]), str(_ONERA["dat"]),
        ]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "merged.szplt"
        dst.touch()
        ret = tecmerge_main([
            "--force", "-o", str(dst),
            str(_ONERA["szplt"]), str(_ONERA["dat"]),
        ])
        assert ret == 0
        assert dst.stat().st_size > 0


# ===========================================================================
# tecscale
# ===========================================================================


class TestTecscale:
    """Tests for tecscale — scales and/or offsets a variable."""

    def test_scale_by_index(self, onera_path: Path, tmp_path: Path) -> None:
        """Scale variable 1 (x) by a constant factor; output is readable."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale_main([
            "-variable", "1", "-scale", "2.0",
            "-o", str(dst), str(onera_path),
        ])
        assert ret == 0
        assert _is_readable(dst)

    def test_scale_by_name(self, onera_path: Path, tmp_path: Path) -> None:
        """Scale the Pressure variable identified by name."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale_main([
            "-variable", "Pressure", "-scale", "1e-3",
            "-o", str(dst), str(onera_path),
        ])
        assert ret == 0
        assert _is_readable(dst)

    def test_scale_with_offset(self, onera_path: Path, tmp_path: Path) -> None:
        """Applying both scale and offset returns exit code 0."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale_main([
            "-variable", "Temperature", "-scale", "1.0", "-offset", "-273.15",
            "-o", str(dst), str(onera_path),
        ])
        assert ret == 0
        assert dst.exists()

    def test_scale_single_zone(self, onera_path: Path, tmp_path: Path) -> None:
        """-zone restricts scaling to zone 1 only."""
        dst = tmp_path / "scaled.szplt"
        ret = tecscale_main([
            "-variable", "Density", "-scale", "1000.0", "-zone", "1",
            "-o", str(dst), str(onera_path),
        ])
        assert ret == 0
        assert dst.exists()

    def test_scale_values_correct(self, tmp_path: Path) -> None:
        """Scaled values in zone 1 equal original × scale_factor."""
        src = _ONERA["szplt"]
        dst = tmp_path / "scaled.szplt"
        scale = 2.0
        tecscale_main([
            "-variable", "1", "-scale", str(scale),
            "--force", "-o", str(dst), str(src),
        ])
        with tecio.open(str(src), "r") as r_orig:
            orig = r_orig.zone[0].variable[0].values.ravel().astype(np.float64)
        with tecio.open(str(dst), "r") as r_scaled:
            scaled = r_scaled.zone[0].variable[0].values.ravel().astype(np.float64)
        np.testing.assert_allclose(scaled, orig * scale, rtol=1e-5)

    def test_unscaled_zone_unchanged(self, tmp_path: Path) -> None:
        """When -zone 1 is active, zone 2 values are identical to source."""
        src = _ONERA["szplt"]
        dst = tmp_path / "scaled.szplt"
        tecscale_main([
            "-variable", "Pressure", "-scale", "999.0", "-zone", "1",
            "--force", "-o", str(dst), str(src),
        ])
        with tecio.open(str(src), "r") as r_src:
            orig = r_src.zone[1].variable[9].values.ravel().astype(np.float64)
        with tecio.open(str(dst), "r") as r_dst:
            copy = r_dst.zone[1].variable[9].values.ravel().astype(np.float64)
        np.testing.assert_allclose(copy, orig, rtol=1e-5)

    def test_invalid_variable_name_returns_1(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Unknown variable name returns exit code 1."""
        assert tecscale_main([
            "-variable", "DOES_NOT_EXIST_XYZ",
            "-o", str(tmp_path / "scaled.szplt"), str(onera_path),
        ]) == 1

    def test_variable_index_out_of_range_returns_1(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Variable index beyond num_vars (18) returns exit code 1."""
        assert tecscale_main([
            "-variable", "99",
            "-o", str(tmp_path / "scaled.szplt"), str(onera_path),
        ]) == 1

    def test_zone_out_of_range_returns_1(
        self, onera_path: Path, tmp_path: Path
    ) -> None:
        """Zone index beyond num_zones (2) returns exit code 1."""
        assert tecscale_main([
            "-variable", "1", "-zone", "99",
            "-o", str(tmp_path / "scaled.szplt"), str(onera_path),
        ]) == 1

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert tecscale_main([
            "-variable", "1",
            "-o", str(tmp_path / "out.szplt"),
            str(tmp_path / "ghost.szplt"),
        ]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "out.szplt"
        dst.touch()
        assert tecscale_main([
            "-variable", "1", "-o", str(dst), str(_ONERA["szplt"])
        ]) == 1


# ===========================================================================
# tecslice
# ===========================================================================


class TestTecslice:
    """Tests for tecslice — slices structured zones along IJK or time.

    Both Onera zones are FE (FEBRICK and FEQUADRILATERAL), so IJK slice flags
    produce verbatim copies with a warning to stderr.  Tests verify that
    behaviour explicitly.
    """

    def test_ijk_slice_on_fe_copies_verbatim(
        self, onera_path: Path, tmp_path: Path, capsys
    ) -> None:
        """IJK ::2 on FE data returns 0; each zone is copied verbatim."""
        dst = tmp_path / "sliced.szplt"
        ret = tecslice_main(["-i", "::2", "-o", str(dst), str(onera_path)])
        assert ret == 0
        assert dst.exists()
        assert "unstructured" in capsys.readouterr().err.lower()

    def test_ijk_slice_preserves_all_zones(self, tmp_path: Path) -> None:
        """Verbatim copy retains the full zone and variable counts."""
        dst = tmp_path / "sliced.szplt"
        tecslice_main(["-i", "::2", "-j", "::2", "-o", str(dst), str(_ONERA["szplt"])])
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES
            assert r.num_vars == _NUM_VARS

    def test_no_flags_warns_and_copies(
        self, onera_path: Path, tmp_path: Path, capsys
    ) -> None:
        """No slice flags prints a warning to stderr and returns 0."""
        dst = tmp_path / "copy.szplt"
        ret = tecslice_main(["-o", str(dst), str(onera_path)])
        assert ret == 0
        assert dst.exists()
        assert "no slice" in capsys.readouterr().err.lower()

    def test_time_filter_on_static_keeps_all_zones(self, tmp_path: Path) -> None:
        """Static zones (strand 0) are always kept regardless of time filter."""
        dst = tmp_path / "sliced.szplt"
        ret = tecslice_main(["-t", "0.5:1.0", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES

    def test_strand_filter_on_static_keeps_all_zones(self, tmp_path: Path) -> None:
        """--strand-id on a static file keeps all zones (strand 0 is immune)."""
        dst = tmp_path / "sliced.szplt"
        ret = tecslice_main([
            "--strand-id", "1", "-t", "0.0:10.0",
            "-o", str(dst), str(_ONERA["szplt"]),
        ])
        assert ret == 0
        with tecio.open(str(dst), "r") as r:
            assert r.num_zones == _NUM_ZONES

    def test_output_format_controlled_by_extension(self, tmp_path: Path) -> None:
        """Output extension controls the file format."""
        dst = tmp_path / "sliced.dat"
        ret = tecslice_main(["-i", "::2", "-o", str(dst), str(_ONERA["szplt"])])
        assert ret == 0
        assert _is_readable(dst)

    def test_invalid_time_skip_returns_1(self, tmp_path: Path) -> None:
        """Negative time skip (::-1) returns exit code 1."""
        assert tecslice_main([
            "-t", "::-1",
            "-o", str(tmp_path / "out.szplt"),
            str(_ONERA["szplt"]),
        ]) == 1

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert tecslice_main([
            "-i", "::2",
            "-o", str(tmp_path / "sliced.szplt"),
            str(tmp_path / "ghost.szplt"),
        ]) == 1

    def test_existing_output_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing output without --force returns exit code 1."""
        dst = tmp_path / "sliced.szplt"
        dst.touch()
        assert tecslice_main([
            "-i", "::2", "-o", str(dst), str(_ONERA["szplt"])
        ]) == 1

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output file."""
        dst = tmp_path / "sliced.szplt"
        dst.touch()
        ret = tecslice_main([
            "--force", "-i", "::2", "-o", str(dst), str(_ONERA["szplt"])
        ])
        assert ret == 0
        assert dst.stat().st_size > 0


# ===========================================================================
# tecstats
# ===========================================================================


class TestTecstats:
    """Tests for tecstats — prints and optionally exports variable statistics."""

    def test_basic_console_output(self, onera_path: Path, capsys) -> None:
        """Statistics table is printed with the expected headers; returns 0."""
        ret = tecstats_main([str(onera_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Zone" in out
        assert "Zone Title" in out
        assert "Min" in out
        assert "Max" in out

    def test_zone_titles_in_output(self, onera_path: Path, capsys) -> None:
        """Zone titles FluidVolume and WingSurface appear in the table."""
        tecstats_main([str(onera_path)])
        out = capsys.readouterr().out
        assert "FluidVolume" in out
        assert "WingSurface" in out

    def test_zone_filter(self, onera_path: Path) -> None:
        """-zone 1 runs without error."""
        assert tecstats_main(["-zone", "1", str(onera_path)]) == 0

    def test_variable_filter(self, onera_path: Path) -> None:
        """-variable 10 (Pressure) runs without error."""
        assert tecstats_main(["-variable", "10", str(onera_path)]) == 0

    def test_zone_and_variable_filter(self, onera_path: Path) -> None:
        """Combined -zone 2 -variable 10 runs without error."""
        assert tecstats_main(["-zone", "2", "-variable", "10", str(onera_path)]) == 0

    def test_csv_created_with_stats_suffix(self, tmp_path: Path) -> None:
        """-csv creates <stem>_stats.csv next to the input file."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        assert tecstats_main(["-csv", str(src)]) == 0
        assert (tmp_path / "Onera_stats.csv").exists()

    def test_csv_zone_suffix(self, tmp_path: Path) -> None:
        """-csv -zone N produces <stem>_zone_N_stats.csv."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        assert tecstats_main(["-csv", "-zone", "1", str(src)]) == 0
        assert (tmp_path / "Onera_zone_1_stats.csv").exists()

    def test_csv_variable_suffix(self, tmp_path: Path) -> None:
        """-csv -variable N produces <stem>_var_N_stats.csv."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        assert tecstats_main(["-csv", "-variable", "10", str(src)]) == 0
        assert (tmp_path / "Onera_var_10_stats.csv").exists()

    def test_csv_zone_and_variable_suffix(self, tmp_path: Path) -> None:
        """Combined filters produce <stem>_zone_N_var_M_stats.csv."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        assert tecstats_main(["-csv", "-zone", "2", "-variable", "10", str(src)]) == 0
        assert (tmp_path / "Onera_zone_2_var_10_stats.csv").exists()

    def test_csv_header_and_data_rows(self, tmp_path: Path) -> None:
        """CSV has the correct header and 36 data rows (2 zones × 18 vars)."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        tecstats_main(["-csv", str(src)])
        lines = (tmp_path / "Onera_stats.csv").read_text(encoding="utf-8").splitlines()
        expected_header = (
            "zone_num,zone_title,var_num,var_name,"
            "min,max,mean,std,location,note"
        )
        assert lines[0] == expected_header
        assert len(lines) == _NUM_ZONES * _NUM_VARS + 1  # 36 data rows + header

    def test_csv_zone_title_in_rows(self, tmp_path: Path) -> None:
        """Zone title column contains the expected zone names."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        tecstats_main(["-csv", str(src)])
        content = (tmp_path / "Onera_stats.csv").read_text(encoding="utf-8")
        assert "FluidVolume" in content
        assert "WingSurface" in content

    def test_csv_existing_no_force_returns_1(self, tmp_path: Path) -> None:
        """Existing CSV without --force returns exit code 1."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        (tmp_path / "Onera_stats.csv").touch()
        assert tecstats_main(["-csv", str(src)]) == 1

    def test_csv_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force replaces an existing CSV with fresh content."""
        src = tmp_path / "Onera.szplt"
        shutil.copy(_ONERA["szplt"], src)
        stale = tmp_path / "Onera_stats.csv"
        stale.write_text("stale", encoding="utf-8")
        ret = tecstats_main(["-csv", "--force", str(src)])
        assert ret == 0
        assert stale.read_text(encoding="utf-8") != "stale"

    def test_missing_input_returns_1(self, tmp_path: Path) -> None:
        """Non-existent source file returns exit code 1."""
        assert tecstats_main([str(tmp_path / "ghost.szplt")]) == 1


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
