#!/usr/bin/env python3
"""pytest tests for :class:`tecio.szl.Write` — SZL high-level writer.

Pattern per test:
    1. Create data with specific dtypes
    2. Write file via tecio.open(..., "w")
    3. Read back via tecio.open(..., "r")
    4. Assert on metadata and values

SZL preserves the on-disk DataType exactly, so dtype round-trips are
verified with ``variable.data_type`` assertions on read-back.

Run directly:

    $ python tests/test_szl_write.py -v

Keep output files for Tecplot 360 inspection:

    $ python tests/test_szl_write.py -v --keep-files
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import tecio
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
from tecio.libtecio import DataType, FaceNeighborMode, FileType, ValueLocation, ZoneType

_RTOL_F32 = 1e-5
_RTOL_F64 = 1e-10


# ===========================================================================
# Ordered (IJK) zone tests
# ===========================================================================


class TestWriteIJKZone:
    """Tests for write_ijk_zone."""

    def test_write_ijk_3d_mixed_dtypes(self, output_path: Callable) -> None:
        """3-D ordered zone with mixed float32 and float64 variables.

        Demonstrates:
        - Basic ``write_ijk_zone`` call structure: ``data``, ``variables``,
          ``title``
        - Mixed-precision zone: float32 x/z, float64 y/scalar in one write call
        - SZL infers DataType from the numpy array dtype automatically
        - Zone dimensions ``(imax, jmax, kmax)`` are inferred from the first
          nodal array shape — no explicit dimension arguments needed
        - Dtype verification: ``variable.data_type`` reads back FLOAT or DOUBLE
          matching the input array dtype exactly
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)                              # → FLOAT
        y = y.astype(np.float64)                              # → DOUBLE
        z = z.astype(np.float32)                              # → FLOAT
        c = scalar_field(x, y, z).astype(np.float64)          # → DOUBLE

        path = output_path("test_szl_write_ijk_3d.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_ijk_zone(
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
            assert zone.variable[0].data_type == DataType.FLOAT
            assert zone.variable[1].data_type == DataType.DOUBLE
            assert zone.variable[2].data_type == DataType.FLOAT
            assert zone.variable[3].data_type == DataType.DOUBLE
            np.testing.assert_allclose(zone.variable[0].values.ravel(), x.ravel(), rtol=_RTOL_F32)
            np.testing.assert_allclose(zone.variable[1].values.ravel(), y.ravel(), rtol=_RTOL_F64)

    def test_write_ijk_cell_centered(self, output_path: Callable) -> None:
        """3-D zone with nodal coordinates and a cell-centered scalar.

        Demonstrates:
        - ``value_locations=[NODAL, NODAL, NODAL, CELL_CENTERED]``: per-variable
          location list; length must equal the number of data arrays
        - Cell-centered array shape must be ``(imax-1, jmax-1, kmax-1)``; the
          writer infers the zone dimensions from the nodal arrays and validates
          the cell-centered shape automatically
        - Read-back: ``variable.value_location == ValueLocation.CELL_CENTERED``
        - Typical use: pressure, temperature, or other solution fields stored
          at element centres rather than nodes
        """
        i, j, k = 4, 5, 3
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        cc = np.random.default_rng(42).random((i - 1, j - 1, k - 1)).astype(np.float64)

        path = output_path("test_szl_write_ijk_cc.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_ijk_zone(
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
            assert cc_var.data_type == DataType.DOUBLE
            assert cc_var.values.size == (i - 1) * (j - 1) * (k - 1)
            np.testing.assert_allclose(cc_var.values.ravel(), cc.ravel(), rtol=_RTOL_F64)

    def test_write_ijk_int32_variable(self, output_path: Callable) -> None:
        """Ordered zone with an INT32 variable — round-trip verified exactly.

        Demonstrates:
        - INT32 array: cast a float field with ``(field * scale).astype(np.int32)``
        - SZL stores INT32 exactly on disk with no floating-point rounding
        - Round-trip assertion: ``assert_array_equal`` (not ``assert_allclose``)
          because integer types have zero round-trip error
        - Useful for zone indices, iteration counts, material IDs, or any
          discrete integer quantity
        """
        i, j, k = 5, 6, 1
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        c = (scalar_field(x, y) * 1000).astype(np.int32)

        path = output_path("test_szl_write_ijk_int32.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_ijk_zone(
                data=[x, y, c],
                variables=["x", "y", "c"],
                title="int32_zone",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.variable[2].data_type == DataType.INT32
            np.testing.assert_array_equal(zone.variable[2].values.ravel(), c.ravel())

    def test_write_ijk_int16_and_byte_variables(self, output_path: Callable) -> None:
        """Ordered zone with INT16 and BYTE variables — ultra-compact storage.

        Demonstrates:
        - INT16 cast pattern: ``(field * 100).astype(np.int16)``
          → range [-32768, 32767]; useful for quality flags or compact indices
        - BYTE (uint8) cast pattern: ``((field + 1.0) * 127).astype(np.uint8)``
          → maps [-1, 1] float range to [0, 254]; useful for color indices
          or per-node boolean masks
        - Both types stored exactly in SZL; verified with ``assert_array_equal``
        - File size benefit: INT16 uses 2 bytes/value, BYTE uses 1 byte/value
          vs 4 bytes/value for FLOAT
        """
        n = 10
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c_i16 = (np.sin(2 * np.pi * x) * 100).astype(np.int16)
        c_u8 = ((np.sin(2 * np.pi * x) + 1.0) * 127).astype(np.uint8)

        path = output_path("test_szl_write_ijk_i16_u8.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_ijk_zone(
                data=[x, c_i16, c_u8],
                variables=["x", "c_i16", "c_u8"],
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.variable[1].data_type == DataType.INT16
            assert zone.variable[2].data_type == DataType.BYTE
            np.testing.assert_array_equal(zone.variable[1].values.ravel(), c_i16)
            np.testing.assert_array_equal(zone.variable[2].values.ravel(), c_u8)

    def test_write_ijk_unsteady_with_sharing(self, output_path: Callable) -> None:
        """100-zone transient dataset with variable sharing for grid coordinates.

        Demonstrates:
        - ``strand_id`` and ``solution_time``: required for transient animation
          in Tecplot 360; zones with the same strand_id animate together
        - Variable sharing: ``var_sharing=[1, 1, 1, 0]`` means variables 1-3
          (x, y, z) are shared from zone 1; only variable 4 (c) is written
          for zones 2-100; eliminates storing the grid N times
        - First zone: ``var_sharing=None`` writes all variables independently
        - Subsequent zones: supply only the non-shared arrays in ``data``
        - ``aux={"Step": str(n)}``: per-zone metadata embedded alongside data
        - File size benefit: grid is stored once regardless of zone count
        """
        i, j, k = 20, 15, 10
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 100)

        path = output_path("test_szl_write_ijk_unsteady.szplt")
        with tecio.open(str(path), "w") as szlfile:
            for n, t in enumerate(solution_times):
                c = scalar_field(x + t, y + t, z).astype(np.float64)
                szlfile.write_ijk_zone(
                    variables=["x", "y", "z", "c"],
                    data=[x, y, z, c] if n == 0 else [c],
                    var_sharing=None if n == 0 else [1, 1, 1, 0],
                    strand_id=1,
                    solution_time=float(t),
                    aux={"Step": str(n)},
                )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 100
            assert r.zone[0].solution_time == pytest.approx(0.0)
            assert r.zone[99].solution_time == pytest.approx(solution_times[-1])
            assert r.zone[50].strand_id == 1
            assert r.zone[1].variable[0].shared_zone is not None

    def test_write_ijk_passive_variable(self, output_path: Callable) -> None:
        """Passive variable in an ordered zone.

        Demonstrates:
        - ``passive_vars=[False, True, False]``: marks "unused" as passive;
          length must equal the total number of dataset variables (not just
          the number of data arrays supplied)
        - Passive variables occupy no storage in the file but are still listed
          in the dataset variable index
        - Supply only the non-passive arrays in ``data``; the writer skips
          passive slots automatically
        - Typical use: a dataset-wide variable that is absent from some zones
          (e.g. turbulence fields absent from a laminar-flow zone)
        """
        n = 8
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)

        path = output_path("test_szl_write_ijk_passive.szplt")
        with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as szlfile:
            szlfile.write_ijk_zone(
                data=[x, c],
                passive_vars=[False, True, False],
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zone[0].variable[1].is_passive()

    def test_write_ijk_dataset_and_zone_aux(self, output_path: Callable) -> None:
        """Dataset-level and zone-level auxiliary data.

        Demonstrates:
        - ``add_auxdataset_dict({"key": "value", ...})``: buffers dataset-level
          metadata; flushed automatically before the first zone is written
        - ``aux={"key": "value"}`` in ``write_ijk_zone``: zone-level metadata
          written alongside that zone's data
        - Common uses: solver name, run conditions, mesh info, boundary labels
        - Both levels survive the SZL round-trip exactly
        """
        x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        dataset_aux = {"Solver": "TestCode", "Mach": "0.72"}
        zone_aux = {"MeshType": "structured", "Author": "pytest"}

        path = output_path("test_szl_write_ijk_aux.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.add_auxdataset_dict(dataset_aux)
            szlfile.write_ijk_zone(data=[x], variables=["x"], aux=zone_aux)

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            for k, v in dataset_aux.items():
                assert r.auxdata[k] == v
            for k, v in zone_aux.items():
                assert r.zone[0].auxdata[k] == v

    def test_write_ijk_file_type_grid(self, output_path: Callable) -> None:
        """FileType.GRID — grid-only file, no solution data.

        Demonstrates:
        - ``file_type=FileType.GRID``: written in the ``tecio.open`` call;
          signals that the file contains only mesh coordinates
        - Paired with a ``FileType.SOLUTION`` file, this avoids duplicating
          the grid in every time-step solution file
        - ``FileType`` is verified to survive the SZL round-trip exactly
        - Options: ``FileType.FULL`` (default), ``FileType.GRID``,
          ``FileType.SOLUTION``
        """
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        path = output_path("test_szl_write_ijk_grid.szplt")

        with tecio.open(str(path), "w", file_type=FileType.GRID) as szlfile:
            szlfile.write_ijk_zone(data=[x], variables=["x"])

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.file_type == FileType.GRID

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_write_ijk_var_count_mismatch_raises(self, output_path: Callable) -> None:
        """Fewer data arrays than active variables raises ValueError.

        Demonstrates:
        - Validation: the writer counts active (non-passive, non-shared)
          variables from the dataset definition and compares to ``len(data)``
        - Error message includes "Expected N arrays, got M" for diagnosis
        """
        x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        y = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        path = output_path("test_szl_write_ijk_mismatch.szplt")

        with pytest.raises(ValueError, match="[Ee]xpected"):
            with tecio.open(str(path), "w", variables=["x", "y", "c"]) as szlfile:
                szlfile.write_ijk_zone(data=[x, y])  # missing c

    def test_write_ijk_shape_mismatch_raises(self, output_path: Callable) -> None:
        """Nodal arrays with inconsistent shapes raise ValueError.

        Demonstrates:
        - Shape validation: all nodal arrays must share the same shape;
          cell-centered arrays must match (nodal_shape - 1) per dimension
        - Raises before any data is written, keeping the file clean
        """
        i, j, k = 4, 5, 1
        x, y, z = create_ordered((i, j, k))
        x = x.squeeze(-1)
        y_bad = y.squeeze(-1)[:-1, :]
        path = output_path("test_szl_write_ijk_shape.szplt")

        with pytest.raises(ValueError):
            with tecio.open(str(path), "w") as szlfile:
                szlfile.write_ijk_zone(data=[x, y_bad], variables=["x", "y"])


# ===========================================================================
# Finite-element zone tests
# ===========================================================================


class TestWriteFEZone:
    """Tests for write_fe_zone."""

    def test_write_fe_lineseg(self, output_path: Callable) -> None:
        """FELINESEG — two-node line segment elements.

        Demonstrates:
        - Basic ``write_fe_zone`` call structure: ``zone_type``, ``data``,
          ``node_map``, ``variables``, ``title``
        - ``ZoneType.FELINESEG``: 2 nodes per element; for 1-D line meshes
          or structural beam elements
        - ``node_map`` shape: ``(num_elements, nodes_per_cell)`` with 1-based
          node indices; the writer infers ``num_nodes = node_map.max()``
        - FLOAT x/y coordinates, DOUBLE scalar field
        - Node map verified with ``assert_array_equal`` — stored exactly
        """
        x, y, nodes = create_FE_lineseg()
        x = x.astype(np.float32)                           # FLOAT
        y = y.astype(np.float32)                           # FLOAT
        c = np.sin(2 * np.pi * x).astype(np.float64)      # DOUBLE

        path = output_path("test_szl_write_fe_lineseg.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_fe_zone(
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
            assert zone.variable[0].data_type == DataType.FLOAT
            assert zone.variable[2].data_type == DataType.DOUBLE
            np.testing.assert_allclose(zone.variable[0].values, x, rtol=_RTOL_F32)
            np.testing.assert_array_equal(zone.node_map, nodes.astype(np.int64))

    def test_write_fe_tri(self, output_path: Callable) -> None:
        """FETRIANGLE — three-node triangular elements.

        Demonstrates:
        - ``ZoneType.FETRIANGLE``: 3 nodes per element; surface triangulation
        - FLOAT x, DOUBLE y, INT32 scalar — three different types in one zone
        - INT32 cast: ``(scalar_field() * 1000).astype(np.int32)``; represents
          a quantity scaled to integer precision (e.g. surface cell quality index)
        - SZL stores all three types independently; verified with mixed
          ``assert_allclose`` (floats) and ``assert_array_equal`` (integers)
        """
        x, y, nodes = create_FE_tri()
        x = x.astype(np.float32)                                   # FLOAT
        y = y.astype(np.float64)                                   # DOUBLE
        c = (scalar_field(x, y) * 1000).astype(np.int32)          # INT32

        path = output_path("test_szl_write_fe_tri.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_fe_zone(
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
            assert zone.variable[0].data_type == DataType.FLOAT
            assert zone.variable[1].data_type == DataType.DOUBLE
            assert zone.variable[2].data_type == DataType.INT32
            np.testing.assert_allclose(zone.variable[0].values, x, rtol=_RTOL_F32)
            np.testing.assert_allclose(zone.variable[1].values, y, rtol=_RTOL_F64)
            np.testing.assert_array_equal(zone.variable[2].values, c)

    def test_write_fe_quad(self, output_path: Callable) -> None:
        """FEQUADRILATERAL — four-node quadrilateral elements.

        Demonstrates:
        - ``ZoneType.FEQUADRILATERAL``: 4 nodes per element; structured-like
          surface mesh or shell elements
        - DOUBLE x/y (float64 coordinates) with FLOAT scalar — reverse of the
          typical CFD convention; shows that precision is set per-variable
        - node_map shape: ``(num_elements, 4)``
        """
        x, y, nodes = create_FE_quad()
        x = x.astype(np.float64)                            # DOUBLE
        y = y.astype(np.float64)                            # DOUBLE
        c = scalar_field(x, y).astype(np.float32)           # FLOAT

        path = output_path("test_szl_write_fe_quad.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_fe_zone(
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
            assert zone.variable[0].data_type == DataType.DOUBLE
            assert zone.variable[2].data_type == DataType.FLOAT
            np.testing.assert_allclose(zone.variable[0].values, x, rtol=_RTOL_F64)

    def test_write_fe_tet(self, output_path: Callable) -> None:
        """FETETRAHEDRON — four-node tetrahedral elements (3-D volume mesh).

        Demonstrates:
        - ``ZoneType.FETETRAHEDRON``: 4 nodes per element; standard 3-D FE
          volume mesh for CFD, FEA, and structural analysis
        - Three coordinate arrays (x, y, z) plus one solution variable
        - FLOAT for spatial coords, DOUBLE for solution — typical CFD pattern:
          compact grid storage, full precision for computed fields
        - node_map shape: ``(num_elements, 4)``; verified with
          ``assert_array_equal`` to confirm 1-based indices survived intact
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)                            # FLOAT
        y = y.astype(np.float32)                            # FLOAT
        z = z.astype(np.float32)                            # FLOAT
        c = scalar_field(x, y).astype(np.float64)           # DOUBLE

        path = output_path("test_szl_write_fe_tet.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_fe_zone(
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
            assert zone.variable[3].data_type == DataType.DOUBLE
            np.testing.assert_array_equal(zone.node_map, nodes.astype(np.int64))

    def test_write_fe_pyramid(self, output_path: Callable) -> None:
        """Pyramid element as a degenerate FEBRICK — INT16 scalar.

        Demonstrates:
        - Pyramid representation: use ``ZoneType.FEBRICK`` with nodes 5-8
          all referencing the single apex node (collapsed brick convention)
        - ``create_FE_pyramid`` returns a ``(1, 8)`` node map with nodes 5-8
          all set to the apex index
        - INT16 scalar: ``(field * 100).astype(np.int16)``; compact storage
          for quantities with limited dynamic range (range [-32768, 32767])
        - SZL stores INT16 exactly; verified with ``assert_array_equal``
        """
        x, y, z, nodes = create_FE_pyramid()
        x = x.astype(np.float32)                                    # FLOAT
        y = y.astype(np.float64)                                    # DOUBLE
        z = z.astype(np.float32)                                    # FLOAT
        c = (scalar_field(x, y) * 100).astype(np.int16)            # INT16

        path = output_path("test_szl_write_fe_pyramid.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Pyramid",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.FEBRICK
            assert zone.variable[3].data_type == DataType.INT16
            np.testing.assert_array_equal(zone.variable[3].values, c)

    def test_write_fe_prism(self, output_path: Callable) -> None:
        """Triangular prism as a degenerate FEBRICK — BYTE (uint8) scalar.

        Demonstrates:
        - Prism representation: ``ZoneType.FEBRICK`` with repeated edge nodes
          (bottom tri 1,2,3,3 / top tri 4,5,6,6); ``create_FE_prism`` handles
          this automatically
        - BYTE (uint8) scalar: ``((field + 1.0) * 127).astype(np.uint8)``
          maps the [-1, 1] float range to [0, 254]; 1 byte per value
        - SZL stores BYTE exactly; verified with ``assert_array_equal``
        - Useful for: visualization colour indices, per-node binary masks,
          material tags limited to 256 categories
        """
        x, y, z, nodes = create_FE_prism()
        x = x.astype(np.float32)                                         # FLOAT
        y = y.astype(np.float64)                                         # DOUBLE
        z = z.astype(np.float64)                                         # DOUBLE
        c = ((scalar_field(x, y) + 1.0) * 127).astype(np.uint8)         # BYTE

        path = output_path("test_szl_write_fe_prism.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Prism",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.variable[3].data_type == DataType.BYTE
            np.testing.assert_array_equal(zone.variable[3].values, c)

    def test_write_fe_brick(self, output_path: Callable) -> None:
        """FEBRICK — standard 8-node hexahedral elements.

        Demonstrates:
        - ``ZoneType.FEBRICK``: 8 nodes per element; standard 3-D structured-
          equivalent unstructured hex mesh
        - FLOAT x/y, DOUBLE z, INT32 scalar — four variables at three different
          precisions in the same zone
        - node_map shape: ``(num_elements, 8)``; uses standard Tecplot hex
          ordering (four bottom nodes, four top nodes in matching order)
        - Value verification: float64 z and int32 c both round-trip exactly
        """
        x, y, z, faces, nodes = create_FE_brick()
        x = x.astype(np.float32)                               # FLOAT
        y = y.astype(np.float32)                               # FLOAT
        z = z.astype(np.float64)                               # DOUBLE
        c = (scalar_field(x, y) * 1000).astype(np.int32)      # INT32

        path = output_path("test_szl_write_fe_brick.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_fe_zone(
                zone_type=ZoneType.FEBRICK,
                data=[x, y, z, c],
                node_map=nodes,
                variables=["x", "y", "z", "c"],
                title="FE_Brick",
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.zone_type == ZoneType.FEBRICK
            assert zone.num_nodes == 8
            assert zone.num_elements == 1
            assert zone.variable[2].data_type == DataType.DOUBLE
            assert zone.variable[3].data_type == DataType.INT32
            np.testing.assert_allclose(zone.variable[2].values, z, rtol=_RTOL_F64)
            np.testing.assert_array_equal(zone.variable[3].values, c)

    def test_write_fe_face_neighbors(self, output_path: Callable) -> None:
        """Two FEBRICK cells with face-neighbor connectivity.

        Demonstrates:
        - ``face_neighbors``: array of (cell, face, neighbor_cell) triplets;
          the writer sets ``num_face_cons`` in the zone header automatically
        - ``face_nbr_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE``: each face has
          at most one neighbor within the same zone
        - Cell-centered INT16 scalar: ``value_locations=[..., CELL_CENTERED]``
          combined with an array of length ``num_elements``; one value per cell
        - Face neighbors enable smooth shading at element boundaries in Tecplot
          360 without duplicating shared-face nodes
        """
        x, y, z, nodes, face_neighbors = create_FE_two_bricks()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = np.array([100, 200], dtype=np.int16)       # CELL_CENTERED INT16

        path = output_path("test_szl_write_fe_face_neighbors.szplt")
        with tecio.open(str(path), "w") as szlfile:
            szlfile.write_fe_zone(
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
            assert zone.num_nodes == len(x)
            assert zone.num_elements == 2
            cc_var = zone.variable[3]
            assert cc_var.value_location == ValueLocation.CELL_CENTERED
            assert cc_var.data_type == DataType.INT16
            np.testing.assert_array_equal(cc_var.values, c)

    def test_write_fe_passive_variable(self, output_path: Callable) -> None:
        """Passive variable in an FE zone.

        Demonstrates:
        - ``passive_vars`` in the FE zone context — identical behaviour to
          ordered zones; works for all zone types
        - The dataset declares three variables; zone supplies only two arrays
          with the middle one marked passive
        - Verified with ``variable.is_passive() == True`` on read-back
        """
        x, y, nodes = create_FE_tri()
        x = x.astype(np.float32)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_szl_write_fe_passive.szplt")
        with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as szlfile:
            szlfile.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, c],
                node_map=nodes,
                passive_vars=[False, True, False],
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zone[0].variable[1].is_passive()

    def test_write_fe_unsteady(self, output_path: Callable) -> None:
        """100 FETETRAHEDRON zones with strand ID and solution time.

        Demonstrates:
        - Transient FE dataset: ``strand_id`` and ``solution_time`` per zone
        - Unlike ordered zones, FE zones cannot share coordinates across zones
          (node maps would differ), so each zone writes all variables
        - All zones share the same node_map topology (fixed mesh, moving solution)
        - Consistent ``strand_id=1`` groups all zones into one animation in
          Tecplot 360's time animation controls
        - Zone-level aux data via ``aux={"MeshType": ..., "Author": ...}``
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 100)

        path = output_path("test_szl_write_fe_unsteady.szplt")
        with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as szlfile:
            for step, t in enumerate(solution_times):
                c = np.sin(x + t).astype(np.float64)
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title=f"zone_t{step + 1}",
                    strand_id=1,
                    solution_time=float(t),
                    aux={"MeshType": "unstructured", "Author": "test_szl_write"},
                )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 100
            assert r.zone[0].solution_time == pytest.approx(0.0)
            assert r.zone[99].solution_time == pytest.approx(solution_times[-1])
            assert r.zone[0].strand_id == 1

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_write_fe_var_count_mismatch_raises(self, output_path: Callable) -> None:
        """Too few data arrays for active variables raises ValueError.

        Demonstrates:
        - Same validation as ordered zones: writer counts active variables
          and compares to ``len(data)``
        - Error message includes "Expected N arrays, got M"
        """
        x, y, nodes = create_FE_tri()
        path = output_path("test_szl_write_fe_mismatch.szplt")

        with pytest.raises(ValueError, match="[Ee]xpected"):
            with tecio.open(str(path), "w", variables=["x", "y", "c"]) as szlfile:
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x, y],
                    node_map=nodes,
                )

    def test_write_fe_array_length_mismatch_raises(self, output_path: Callable) -> None:
        """Nodal array shorter than num_nodes raises ValueError.

        Demonstrates:
        - Writer computes ``num_nodes = node_map.max()`` and validates that
          every nodal array has exactly that many values
        - Raises before any data is written to disk
        """
        x, y, nodes = create_FE_tri()
        path = output_path("test_szl_write_fe_len.szplt")

        with pytest.raises(ValueError):
            with tecio.open(str(path), "w") as szlfile:
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x[:-1], y],
                    node_map=nodes,
                    variables=["x", "y"],
                )

    def test_write_fe_unsupported_zone_type_raises(self, output_path: Callable) -> None:
        """FEPOLYGON raises NotImplementedError.

        Demonstrates:
        - Poly zones (FEPOLYGON, FEPOLYHEDRON) require face-based connectivity
          that ``write_fe_zone`` does not support
        - Use the low-level ``libtecio.tecpolyzne142`` + ``tecpolyface142`` path
          for polygon/polyhedral zones (see test_libtecio.py)
        """
        x, y, nodes = create_FE_tri()
        path = output_path("test_szl_write_fe_poly.szplt")

        with pytest.raises(NotImplementedError):
            with tecio.open(str(path), "w") as szlfile:
                szlfile.write_fe_zone(
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
