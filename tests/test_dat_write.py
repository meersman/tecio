#!/usr/bin/env python3
"""pytest tests for :class:`tecio.dat.Write` — DAT ASCII writer.

Pattern per test:
    1. Create data with specific dtypes
    2. Write file via tecio.open(..., "w")
    3. Read back via tecio.open(..., "r")
    4. Assert on metadata and values

DAT format notes
----------------
* All values are written as ASCII text and read back as float64 regardless
  of the input numpy dtype.  ``variable.data_type`` always returns
  ``DataType.DOUBLE`` — no dtype assertions are made in these tests.
* Value precision is controlled by ``sig_digits`` (default 9), giving
  roughly 1e-8 relative tolerance; tests use rtol=1e-7 for safety margin.
* Node maps are written as 1-based integers in ASCII and read back exactly.
* Integer arrays (int32, int16) are preserved exactly through ASCII since
  their values fit comfortably within the default 9 significant digits.

Run directly:

    $ python tests/test_dat_write.py -v

Keep output files for Tecplot 360 inspection:

    $ python tests/test_dat_write.py -v --keep-files
"""

# ruff: noqa: E501, SIM117

import sys
from collections.abc import Callable

import numpy as np
import pytest
from create_test_data import (
    create_FE_brick,
    create_FE_lineseg,
    create_FE_prism,
    create_FE_pyramid,
    create_FE_quad,
    create_FE_tet,
    create_FE_tri,
    create_FE_two_bricks,
    create_ordered,
    scalar_field,
)

import tecio
from tecio.libtecio import (
    DataPacking,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
)

# DAT reads everything back as float64; tolerance allows for 9-digit ASCII formatting
_RTOL_DAT = 1e-7


# ===========================================================================
# Ordered (IJK) zone tests
# ===========================================================================


class TestWriteIJKZone:
    """Tests for write_ijk_zone targeting DAT ASCII output."""

    def test_write_ijk_3d_mixed_input_dtypes(self, output_path: Callable) -> None:
        """3-D zone written with float32, float64, and int32 input arrays.

        Demonstrates:
        - Basic ``write_ijk_zone`` call for DAT: identical API to SZL/PLT
        - All input dtypes are written as ASCII text; all read back as float64
        - Value tolerance: default ``sig_digits=9`` gives ~1e-8 relative error;
          tests use rtol=1e-7 for a comfortable safety margin
        - No ``variable.data_type`` assertions — DAT always reports DOUBLE
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        z = z.astype(np.float32)
        c = scalar_field(x, y, z).astype(np.float64)

        path = output_path("test_dat_write_ijk_3d.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_ijk_zone(
                data=[x, y, z, c],
                variables=["x", "y", "z", "c"],
                title="zone_3d",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 1
            assert r.num_vars == 4
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.ORDERED
            assert zone.dimensions == (i, j, k)
            np.testing.assert_allclose(zone.variable[0].values.ravel(), x.ravel(), rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[1].values.ravel(), y.ravel(), rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[3].values.ravel(), c.ravel(), rtol=_RTOL_DAT)

    def test_write_ijk_int_arrays_preserve_values(self, output_path: Callable) -> None:
        """int32 and int16 inputs round-trip through DAT ASCII without loss.

        Demonstrates:
        - Integer arrays (int32, int16) are written as decimal text and read
          back as float64; values are preserved exactly because 9 significant
          digits can represent all integers up to 10^9 exactly
        - Cast patterns:
          - int32: ``np.arange(n) * 100`` → values 0, 100, 200, ...
          - int16: ``np.arange(n) * 10``  → values 0, 10, 20, ...
        - Assertion: ``assert_allclose`` against the float64-cast original
          (exact match for these small integers)
        - Implication: DAT is suitable for integer fields when exact ASCII
          representation is required; use SZL for compact binary int storage
        """
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c_i32 = np.arange(n, dtype=np.int32) * 100
        c_i16 = np.arange(n, dtype=np.int16) * 10

        path = output_path("test_dat_write_ijk_int.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_ijk_zone(data=[x, c_i32, c_i16], variables=["x", "c_i32", "c_i16"])

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            np.testing.assert_allclose(
                zone.variable[1].values.ravel(), c_i32.astype(np.float64)
            )
            np.testing.assert_allclose(
                zone.variable[2].values.ravel(), c_i16.astype(np.float64)
            )

    def test_write_ijk_cell_centered(self, output_path: Callable) -> None:
        """3-D zone with cell-centered float64 scalar in DAT format.

        Demonstrates:
        - ``value_locations=[..., CELL_CENTERED]``: same API as SZL/PLT
        - Cell-centered array shape: ``(imax-1, jmax-1, kmax-1)``
        - DAT writes cell-centered data in BLOCK format (one full variable
          block per variable) with the VARLOCATION header keyword
        - Values verified within DAT ASCII precision (rtol=1e-7)
        """
        i, j, k = 4, 5, 3
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        cc = np.random.default_rng(42).random((i - 1, j - 1, k - 1)).astype(np.float64)

        path = output_path("test_dat_write_ijk_cc.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_ijk_zone(
                data=[x, y, z, cc],
                variables=["x", "y", "z", "cc"],
                value_locations=[
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.CELL_CENTERED,
                ],
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            cc_var = zone.variable[3]
            assert cc_var.value_location == ValueLocation.CELL_CENTERED
            assert cc_var.values.size == (i - 1) * (j - 1) * (k - 1)
            np.testing.assert_allclose(cc_var.values.ravel(), cc.ravel(), rtol=_RTOL_DAT)

    def test_write_ijk_high_precision_sig_digits(self, output_path: Callable) -> None:
        """sig_digits=17 gives full float64 round-trip fidelity.

        Demonstrates:
        - ``sig_digits`` parameter on ``tecio.open(..., "w", sig_digits=17)``:
          controls the number of significant digits in the ASCII output
        - Default ``sig_digits=9`` gives ~1e-8 relative precision
        - ``sig_digits=17`` gives the maximum float64 precision (~1e-15);
          use this when the DAT file must preserve the exact solver values
          (e.g. for regression testing or data archival)
        - Tradeoff: larger files and slower read/write at higher precision
        """
        x = np.array([1.23456789012345678, 2.34567890123456789], dtype=np.float64)

        path = output_path("test_dat_write_ijk_sigdigits.dat")
        with tecio.open(str(path), "w", sig_digits=17) as datfile:
            datfile.write_ijk_zone(data=[x], variables=["x"])

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            np.testing.assert_allclose(
                r.zone[0].variable[0].values.ravel(), x, rtol=1e-15
            )

    def test_write_ijk_unsteady(self, output_path: Callable) -> None:
        """Multiple ordered zones with solution time — verified on read-back.

        Demonstrates:
        - Transient dataset in DAT: same ``strand_id`` and ``solution_time``
          API as SZL/PLT; written as STRANDID and SOLUTIONTIME in the zone header
        - DAT does not support variable sharing (ASCII format has no sharing
          mechanism); every zone must write all its variable arrays independently
        - Solution times verified with ``pytest.approx`` after ASCII round-trip
        """
        i, j = 10, 8
        x, y, z = create_ordered((i, j, 1))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        solution_times = [0.0, 0.5, 1.0, 1.5, 2.0]

        path = output_path("test_dat_write_ijk_unsteady.dat")
        with tecio.open(str(path), "w", variables=["x", "y", "c"]) as datfile:
            for t in solution_times:
                c = scalar_field(x + t, y + t).astype(np.float64)
                datfile.write_ijk_zone(data=[x, y, c], strand_id=1, solution_time=t)

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == len(solution_times)
            for i, t in enumerate(solution_times):
                assert r.zone[i].solution_time == pytest.approx(t)

    def test_write_ijk_passive_variable(self, output_path: Callable) -> None:
        """Passive variable in DAT format.

        Demonstrates:
        - ``passive_vars=[False, True, False]``: DAT writes PASSIVEVARLIST
          in the zone header; the passive variable has no data block
        - Verified with ``variable.is_passive() == True`` on read-back
        - Same API as SZL and PLT; works for all supported formats
        """
        x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)
        path = output_path("test_dat_write_ijk_passive.dat")

        with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as datfile:
            datfile.write_ijk_zone(data=[x, c], passive_vars=[False, True, False])

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zone[0].variable[1].is_passive()

    def test_write_ijk_aux_data(self, output_path: Callable) -> None:
        """Dataset and zone auxiliary data in DAT format.

        Demonstrates:
        - ``add_auxdataset_dict``: written as ``DATASETAUXDATA name="value"``
          lines at the file level (before the first ZONE keyword)
        - ``aux={"key": "value"}`` in ``write_ijk_zone``: written as
          ``AUXDATA key="value"`` inside the zone header
        - DAT auxiliary data is human-readable ASCII; both levels verified
          with exact string comparison on read-back
        """
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        dataset_aux = {"Solver": "TestCode", "Mach": "0.72"}
        zone_aux = {"MeshType": "structured", "Author": "pytest"}
        path = output_path("test_dat_write_ijk_aux.dat")

        with tecio.open(str(path), "w") as datfile:
            datfile.add_auxdataset_dict(dataset_aux)
            datfile.write_ijk_zone(data=[x], variables=["x"], aux=zone_aux)

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            for k, v in dataset_aux.items():
                assert r.auxdata[k] == v
            for k, v in zone_aux.items():
                assert r.zone[0].auxdata[k] == v

    def test_write_ijk_file_type_solution(self, output_path: Callable) -> None:
        """FileType.SOLUTION in DAT format.

        Demonstrates:
        - ``file_type=FileType.SOLUTION``: written as ``FILETYPE = SOLUTION``
          in the DAT file header; signals solution-only data (no grid)
        - Useful when the grid is stored in a separate FILETYPE=GRID file
        - FileType.SOLUTION verified to survive DAT round-trip exactly
        """
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        path = output_path("test_dat_write_ijk_solution.dat")

        with tecio.open(str(path), "w", file_type=FileType.SOLUTION) as datfile:
            datfile.write_ijk_zone(data=[x], variables=["x"])

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.file_type == FileType.SOLUTION

    def test_write_ijk_point_basic(self, output_path: Callable) -> None:
        """1-D ordered zone written and read back in POINT format.

        Demonstrates:
        - ``datapacking=DataPacking.POINT`` on ``write_ijk_zone``: emits one row of all
          variable values per node instead of one full variable block per variable
        - Data layout is equivalent to CSV with a Tecplot header; useful for
          third-party tools that expect row-major ASCII data
        - Values survive the POINT ASCII round-trip within DAT precision
        - Zone metadata (type, dimensions) are identical to a BLOCK-packed zone
        """
        n = 20
        x = np.linspace(0.0, 2 * np.pi, n, dtype=np.float64)
        y = np.sin(x).astype(np.float64)
        c = np.cos(x).astype(np.float64)

        path = output_path("test_dat_write_ijk_point_basic.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_ijk_zone(
                data=[x, y, c],
                variables=["x", "sin_x", "cos_x"],
                title="point_1d",
                datapacking=DataPacking.POINT,
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 1
            assert r.num_vars == 3
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.ORDERED
            assert zone.dimensions == (n, 1, 1)
            np.testing.assert_allclose(zone.variable[0].values.ravel(), x, rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[1].values.ravel(), y, rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[2].values.ravel(), c, rtol=_RTOL_DAT)

    def test_write_ijk_point_3d(self, output_path: Callable) -> None:
        """3-D IJK zone in POINT format — values and dimensions verified.

        Demonstrates:
        - POINT packing is valid for multi-dimensional ordered zones; Tecplot
          360 reads the rows in I-J-K (Fortran-column-major) index order
        - Coordinates and a scalar field survive the round-trip within
          DAT ASCII precision (rtol=1e-7)
        - Reshaped (I, J, K) arrays are ravelled column-major before writing
          to match the expected node ordering
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))

        path = output_path("test_dat_write_ijk_point_3d.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_ijk_zone(
                data=[x, y, z],
                variables=["x", "y", "z"],
                title="point_3d",
                datapacking=DataPacking.POINT,
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.dimensions == (i, j, k)
            np.testing.assert_allclose(zone.variable[0].values.ravel(), x.ravel(), rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[1].values.ravel(), y.ravel(), rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[2].values.ravel(), z.ravel(), rtol=_RTOL_DAT)

    def test_write_ijk_point_matches_block(self, output_path: Callable) -> None:
        """POINT and BLOCK produce identical read-back values for the same data.

        Demonstrates:
        - The two packing modes are semantically equivalent; the only
          difference is the on-disk layout
        - Verifies that the POINT reader correctly transposes rows into
          per-variable arrays matching the BLOCK output
        """
        i, j = 6, 7
        x, y, _ = create_ordered((i, j, 1))
        c = scalar_field(x.squeeze(-1), y.squeeze(-1)).astype(np.float64)
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        path_block = output_path("test_dat_ijk_packing_block.dat")
        path_point = output_path("test_dat_ijk_packing_point.dat")

        for path, packing in ((path_block, "BLOCK"), (path_point, "POINT")):
            with tecio.open(str(path), "w") as datfile:
                datfile.write_ijk_zone(
                    data=[x, y, c],
                    variables=["x", "y", "c"],
                    datapacking=packing,
                )

        with tecio.open(str(path_block), "r") as rb, tecio.open(str(path_point), "r") as rp:
            for vi in range(3):
                np.testing.assert_allclose(
                    rb.zone[0].variable[vi].values.ravel(),
                    rp.zone[0].variable[vi].values.ravel(),
                    rtol=_RTOL_DAT,
                )

    def test_write_ijk_point_cell_centered(self, output_path: Callable) -> None:
        """POINT packing with a cell-centred variable — nodal rows then CC rows.

        Demonstrates:
        - Mixed nodal + CC variables in POINT format: the spec places nodal
          variable rows first (one row per node) followed by a separate CC
          section (one row per cell)
        - ``VARLOCATION`` keyword is still emitted in the zone header
        - CC values survive the two-section POINT round-trip correctly
        """
        i, j, k = 4, 5, 3
        x, y, z = create_ordered((i, j, k))
        cc = np.random.default_rng(7).random((i - 1, j - 1, k - 1)).astype(np.float64)

        path = output_path("test_dat_write_ijk_point_cc.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_ijk_zone(
                data=[x, y, z, cc],
                variables=["x", "y", "z", "cc"],
                value_locations=[
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.CELL_CENTERED,
                ],
                datapacking=DataPacking.POINT,
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            cc_var = zone.variable[3]
            assert cc_var.value_location == ValueLocation.CELL_CENTERED
            assert cc_var.values.size == (i - 1) * (j - 1) * (k - 1)
            np.testing.assert_allclose(cc_var.values.ravel(), cc.ravel(), rtol=_RTOL_DAT)

    def test_write_ijk_point_invalid_datapacking_raises(
        self, output_path: Callable
    ) -> None:
        """Unrecognised datapacking string raises ValueError immediately."""
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        path = output_path("test_dat_write_ijk_bad_packing.dat")

        with pytest.raises(ValueError, match="datapacking"):
            with tecio.open(str(path), "w") as datfile:
                datfile.write_ijk_zone(data=[x], variables=["x"], datapacking="CSV")

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_write_ijk_var_count_mismatch_raises(self, output_path: Callable) -> None:
        """Fewer arrays than active variables raises ValueError."""
        x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        y = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        path = output_path("test_dat_write_ijk_mismatch.dat")

        with pytest.raises(ValueError, match="[Ee]xpected"):
            with tecio.open(str(path), "w", variables=["x", "y", "c"]) as datfile:
                datfile.write_ijk_zone(data=[x, y])

    def test_write_ijk_shape_mismatch_raises(self, output_path: Callable) -> None:
        """Inconsistent array shapes raise ValueError."""
        i, j, k = 4, 5, 1
        x, y, _ = create_ordered((i, j, k))
        x = x.squeeze(-1)
        y_bad = y.squeeze(-1)[:-1, :]
        path = output_path("test_dat_write_ijk_shape.dat")

        with pytest.raises(ValueError), tecio.open(str(path), "w") as datfile:
            datfile.write_ijk_zone(data=[x, y_bad], variables=["x", "y"])


# ===========================================================================
# Finite-element zone tests
# ===========================================================================


class TestWriteFEZone:
    """Tests for write_fe_zone targeting DAT ASCII output."""

    def test_write_fe_lineseg(self, output_path: Callable) -> None:
        """FELINESEG — float32 x/y, float64 scalar. Node map preserved exactly.

        Demonstrates:
        - Basic FE write to DAT: same ``write_fe_zone`` API as SZL/PLT
        - DAT writes connectivity as ASCII integers (1-based); read back
          as int64 via the DAT reader
        - Node map verified with ``assert_array_equal``
        - All float arrays read back as float64; tolerance is rtol=1e-7
        """
        x, y, nodes = create_FE_lineseg()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)

        path = output_path("test_dat_write_fe_lineseg.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FELINESEG,
                data=[x, y, c],
                node_map=nodes,
                variables=["x", "y", "c"],
                title="FE_LineSeg",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.FELINESEG
            assert zone.num_nodes == len(x)
            assert zone.num_elements == len(nodes)
            np.testing.assert_allclose(zone.variable[0].values, x, rtol=_RTOL_DAT)
            np.testing.assert_array_equal(zone.node_map, nodes.astype(np.int64))

    def test_write_fe_tri(self, output_path: Callable) -> None:
        """FETRIANGLE — float32 x, float64 y, int32 scalar.

        Demonstrates:
        - Mixed input dtypes in an FE zone; all become float64 in DAT
        - int32 scalar: ``(scalar_field() * 1000).astype(np.int32)``; values
          are preserved exactly through 9-digit ASCII formatting
        - Assertion pattern: compare int32 values against their float64 cast
        """
        x, y, nodes = create_FE_tri()
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        c = (scalar_field(x, y) * 1000).astype(np.int32)

        path = output_path("test_dat_write_fe_tri.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, y, c],
                node_map=nodes,
                variables=["x", "y", "c"],
                title="FE_Tri",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.FETRIANGLE
            assert zone.num_nodes == 4
            assert zone.num_elements == 2
            np.testing.assert_allclose(zone.variable[1].values, y, rtol=_RTOL_DAT)
            np.testing.assert_allclose(
                zone.variable[2].values, c.astype(np.float64)
            )

    def test_write_fe_quad(self, output_path: Callable) -> None:
        """FEQUADRILATERAL — float64 x/y, float32 scalar.

        Demonstrates:
        - DOUBLE coordinates with FLOAT scalar in DAT; both become float64
          on read-back — DAT does not preserve individual input dtypes
        - Values verified within DAT ASCII precision
        """
        x, y, nodes = create_FE_quad()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        c = scalar_field(x, y).astype(np.float32)

        path = output_path("test_dat_write_fe_quad.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FEQUADRILATERAL,
                data=[x, y, c],
                node_map=nodes,
                variables=["x", "y", "c"],
                title="FE_Quad",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.FEQUADRILATERAL
            np.testing.assert_allclose(zone.variable[0].values, x, rtol=_RTOL_DAT)

    def test_write_fe_tet(self, output_path: Callable) -> None:
        """FETETRAHEDRON — float32 x/y/z, float64 scalar. Node map verified.

        Demonstrates:
        - 3-D FETETRAHEDRON in DAT: same call as SZL/PLT
        - Standard CFD precision: float32 grid, float64 solution
        - Node map ASCII round-trip: written as integers, read back as int64
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_dat_write_fe_tet.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FETETRAHEDRON,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Tet",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.FETETRAHEDRON
            assert zone.num_nodes == 5
            assert zone.num_elements == 2
            np.testing.assert_array_equal(zone.node_map, nodes.astype(np.int64))

    def test_write_fe_pyramid(self, output_path: Callable) -> None:
        """Degenerate FEBRICK pyramid — float32 all variables.

        Demonstrates:
        - Pyramid as collapsed FEBRICK in DAT: same node_map convention as
          SZL/PLT; DAT writes the 8-node connectivity as integers
        - All float32 input; all read back as float64 in DAT
        """
        x, y, z, nodes = create_FE_pyramid()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = scalar_field(x, y).astype(np.float32)

        path = output_path("test_dat_write_fe_pyramid.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Pyramid",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zone[0].zone_type == ZoneType.FEBRICK

    def test_write_fe_prism(self, output_path: Callable) -> None:
        """Degenerate FEBRICK prism — float64 all variables.

        Demonstrates:
        - Triangular prism as FEBRICK in DAT: repeated edge nodes in node_map
        - All float64; DAT ASCII output at maximum default precision (9 sig fig)
        """
        x, y, z, nodes = create_FE_prism()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        z = z.astype(np.float64)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_dat_write_fe_prism.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Prism",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zone[0].zone_type == ZoneType.FEBRICK

    def test_write_fe_brick(self, output_path: Callable) -> None:
        """FEBRICK — float32 x/y, float64 z/scalar.

        Demonstrates:
        - Standard 8-node hex in DAT ASCII format
        - Mixed float32/float64 input; float64 z verified within rtol=1e-7
          (dominated by DAT's 9-digit ASCII precision, not float64 precision)
        """
        x, y, z, faces, nodes = create_FE_brick()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float64)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_dat_write_fe_brick.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Brick",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.num_nodes == 8
            assert zone.num_elements == 1
            np.testing.assert_allclose(zone.variable[2].values, z, rtol=_RTOL_DAT)

    def test_write_fe_face_neighbors(self, output_path: Callable) -> None:
        """Two FEBRICK cells with face-neighbor connectivity and cell-centered variable.

        Demonstrates:
        - Face neighbors in DAT: ``face_neighbors`` and ``face_nbr_mode``
          work identically to SZL/PLT at the high-level API
        - Cell-centered float64 variable: ``value_locations=[..., CELL_CENTERED]``
        - DAT writes VARLOCATION keyword in the zone header for CC variables
        - CC values verified within DAT ASCII precision
        """
        x, y, z, nodes, face_neighbors = create_FE_two_bricks()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = np.array([1.5, 2.5], dtype=np.float64)

        path = output_path("test_dat_write_fe_face_neighbors.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_2Bricks",
                value_locations=[
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.CELL_CENTERED,
                ],
                face_neighbors=face_neighbors,
                face_nbr_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.num_elements == 2
            cc_var = zone.variable[3]
            assert cc_var.value_location == ValueLocation.CELL_CENTERED
            np.testing.assert_allclose(cc_var.values, c, rtol=_RTOL_DAT)

    def test_write_fe_unsteady(self, output_path: Callable) -> None:
        """100 FETETRAHEDRON zones with strand ID and solution time.

        Demonstrates:
        - Transient FE dataset in DAT: ``strand_id`` and ``solution_time``
          written as STRANDID and SOLUTIONTIME in each zone header
        - DAT does not support variable sharing; each zone writes all arrays
        - Solution times verified after ASCII round-trip with pytest.approx
        - ``aux={}`` added per zone for demonstrating zone-level metadata
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 100)

        path = output_path("test_dat_write_fe_unsteady.dat")
        with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as datfile:
            for step, t in enumerate(solution_times):
                c = np.sin(x + t).astype(np.float64)
                datfile.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title=f"zone_t{step + 1}",
                    strand_id=1,
                    solution_time=float(t),
                    aux={"MeshType": "unstructured", "Author": "test_dat_write"},
                )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 100
            assert r.zone[0].solution_time == pytest.approx(0.0)
            assert r.zone[99].solution_time == pytest.approx(solution_times[-1])

    def test_write_fe_point_tri(self, output_path: Callable) -> None:
        """FETRIANGLE in POINT format — nodal rows verified on read-back.

        Demonstrates:
        - ``datapacking=DataPacking.POINT`` on ``write_fe_zone``: one row per node
          containing all nodal variable values
        - Connectivity block always follows the data section regardless of
          packing mode
        - Node map, coordinates, and scalar field all survive the round-trip
        """
        x, y, nodes = create_FE_tri()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_dat_write_fe_point_tri.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, y, c],
                node_map=nodes,
                variables=["x", "y", "c"],
                title="FE_Tri_Point",
                datapacking=DataPacking.POINT,
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.FETRIANGLE
            assert zone.num_nodes == 4
            assert zone.num_elements == 2
            np.testing.assert_allclose(zone.variable[0].values, x, rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[1].values, y, rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[2].values, c, rtol=_RTOL_DAT)
            np.testing.assert_array_equal(zone.node_map, nodes.astype(np.int64))

    def test_write_fe_point_tet(self, output_path: Callable) -> None:
        """FETETRAHEDRON in POINT format — float32 coords, float64 scalar.

        Demonstrates:
        - POINT packing with a 3-D FE zone; the common CFD pattern of
          float32 grid + float64 solution works identically in POINT mode
        - Scalar values verified within DAT ASCII precision
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_dat_write_fe_point_tet.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FETETRAHEDRON,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Tet_Point",
                datapacking=DataPacking.POINT,
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.FETETRAHEDRON
            assert zone.num_nodes == 5
            assert zone.num_elements == 2
            np.testing.assert_allclose(zone.variable[2].values, z, rtol=_RTOL_DAT)
            np.testing.assert_allclose(zone.variable[3].values, c, rtol=_RTOL_DAT)

    def test_write_fe_point_matches_block(self, output_path: Callable) -> None:
        """POINT and BLOCK produce identical read-back values for an FE zone.

        Demonstrates:
        - Packing mode does not affect the values returned by the reader;
          only the on-disk layout differs
        - FE connectivity is written and read back identically in both modes
        """
        x, y, nodes = create_FE_quad()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        c = scalar_field(x, y).astype(np.float64)

        path_block = output_path("test_dat_fe_packing_block.dat")
        path_point = output_path("test_dat_fe_packing_point.dat")

        for path, packing in ((path_block, "BLOCK"), (path_point, "POINT")):
            with tecio.open(str(path), "w") as datfile:
                datfile.write_fe_zone(
                    zone_type=ZoneType.FEQUADRILATERAL,
                    data=[x, y, c],
                    node_map=nodes,
                    variables=["x", "y", "c"],
                    datapacking=packing,
                )

        with tecio.open(str(path_block), "r") as rb, tecio.open(str(path_point), "r") as rp:
            for vi in range(3):
                np.testing.assert_allclose(
                    rb.zone[0].variable[vi].values.ravel(),
                    rp.zone[0].variable[vi].values.ravel(),
                    rtol=_RTOL_DAT,
                )
            np.testing.assert_array_equal(
                rb.zone[0].node_map, rp.zone[0].node_map
            )

    def test_write_fe_point_cell_centered(self, output_path: Callable) -> None:
        """FE zone in POINT format with a cell-centred variable.

        Demonstrates:
        - Mixed nodal + CC in POINT mode for an FE zone: nodal rows first,
          then one CC row per element
        - ``VARLOCATION`` is still emitted in the zone header
        - CC values survive the two-section POINT round-trip
        """
        x, y, z, nodes, _ = create_FE_two_bricks()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        z = z.astype(np.float64)
        c = np.array([1.5, 2.5], dtype=np.float64)

        path = output_path("test_dat_write_fe_point_cc.dat")
        with tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_2Bricks_Point",
                value_locations=[
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.CELL_CENTERED,
                ],
                datapacking=DataPacking.POINT,
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            cc_var = zone.variable[3]
            assert cc_var.value_location == ValueLocation.CELL_CENTERED
            np.testing.assert_allclose(cc_var.values, c, rtol=_RTOL_DAT)

    def test_write_fe_point_invalid_datapacking_raises(
        self, output_path: Callable
    ) -> None:
        """Unrecognised datapacking string raises ValueError immediately."""
        x, y, nodes = create_FE_tri()
        path = output_path("test_dat_write_fe_bad_packing.dat")

        with pytest.raises(ValueError, match="datapacking"):
            with tecio.open(str(path), "w") as datfile:
                datfile.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x, y],
                    node_map=nodes,
                    variables=["x", "y"],
                    datapacking="ROWS",
                )

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_write_fe_var_count_mismatch_raises(self, output_path: Callable) -> None:
        """Too few data arrays raises ValueError."""
        x, y, nodes = create_FE_tri()
        path = output_path("test_dat_write_fe_mismatch.dat")

        with pytest.raises(ValueError, match="[Ee]xpected"):
            with tecio.open(str(path), "w", variables=["x", "y", "c"]) as datfile:
                datfile.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x, y],
                    node_map=nodes,
                )

    def test_write_fe_array_length_mismatch_raises(self, output_path: Callable) -> None:
        """Nodal array shorter than num_nodes raises ValueError."""
        x, y, nodes = create_FE_tri()
        path = output_path("test_dat_write_fe_len.dat")

        with pytest.raises(ValueError), tecio.open(str(path), "w") as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x[:-1], y],
                node_map=nodes,
                variables=["x", "y"],
            )

    def test_write_fe_unsupported_zone_type_raises(self, output_path: Callable) -> None:
        """FEPOLYGON raises NotImplementedError.

        Demonstrates:
        - Poly zones are not supported by the DAT writer (no face-based
          connectivity format in the ASCII spec implemented here)
        - Use the low-level ``libtecio.tecpolyzne142`` + ``tecpolyface142``
          path for polygon zones in PLT format
        """
        x, y, nodes = create_FE_tri()
        path = output_path("test_dat_write_fe_poly.dat")

        with pytest.raises(NotImplementedError):
            with tecio.open(str(path), "w") as datfile:
                datfile.write_fe_zone(
                    zone_type=ZoneType.FEPOLYGON,
                    data=[x, y],
                    node_map=nodes,
                    variables=["x", "y"],
                )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
