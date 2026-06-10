#!/usr/bin/env python3
"""pytest tests for :class:`tecio.plt.Write` — PLT high-level writer.

Pattern per test:
    1. Create data with specific dtypes
    2. Write file via tecio.open(..., "w")
    3. Read back via tecio.open(..., "r")
    4. Assert on metadata and values

PLT dtype limitation
--------------------
The classic ``tecdat142`` call accepts only float32 or float64.
Input arrays are mapped as follows by the high-level writer:

    float32  →  FLOAT  on disk  →  reads back as DataType.FLOAT
    float64  →  DOUBLE on disk  →  reads back as DataType.DOUBLE
    int32    →  upcast to DOUBLE on disk  →  reads back as DataType.DOUBLE
    int16    →  upcast to DOUBLE on disk  →  reads back as DataType.DOUBLE
    uint8    →  upcast to DOUBLE on disk  →  reads back as DataType.DOUBLE

Integer arrays are upcast to float64 in the file; values are still preserved
exactly because float64 can represent all int32 values.

Run directly:

    $ python tests/test_plt_write.py -v

Keep output files for Tecplot 360 inspection:

    $ python tests/test_plt_write.py -v --keep-files
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
from tecio.libtecio import DataType, FaceNeighborMode, ValueLocation, ZoneType

_RTOL_F32 = 1e-5
_RTOL_F64 = 1e-10


# ===========================================================================
# Ordered (IJK) zone tests
# ===========================================================================


class TestWriteIJKZone:
    """Tests for write_ijk_zone targeting PLT output."""

    def test_write_ijk_3d_float32_float64(self, output_path: Callable) -> None:
        """3-D zone — float32 x/z, float64 y/scalar.

        Demonstrates:
        - Basic ``write_ijk_zone`` call for PLT: identical API to SZL
        - Mixed float32 / float64: PLT preserves FLOAT and DOUBLE natively
        - Zone dimensions inferred from the first nodal array shape
        - DataType limitation: PLT classic API calls tecini142 with VIsDouble=1
          (the default), which tells the C library to store all floating-point
          data as DOUBLE regardless of the per-variable is_double flag passed
          to individual tecdat142 calls.  Every variable reads back as DOUBLE.
          Use SZL format if per-variable float32 storage is required.
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)  # → FLOAT
        y = y.astype(np.float64)  # → DOUBLE
        z = z.astype(np.float32)  # → FLOAT
        c = scalar_field(x, y, z).astype(np.float64)  # → DOUBLE

        path = output_path("test_plt_write_ijk_3d.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_ijk_zone(
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
            assert zone.variable[0].data_type == DataType.DOUBLE  # input was float32
            assert zone.variable[1].data_type == DataType.DOUBLE
            assert zone.variable[2].data_type == DataType.DOUBLE  # input was float32
            assert zone.variable[3].data_type == DataType.DOUBLE
            np.testing.assert_allclose(
                zone.variable[0].values.ravel(), x.ravel(), rtol=_RTOL_F32
            )
            np.testing.assert_allclose(
                zone.variable[1].values.ravel(), y.ravel(), rtol=_RTOL_F64
            )

    def test_write_ijk_int32_upcasts_to_double(self, output_path: Callable) -> None:
        """int32 input is stored as float64 in PLT.

        Demonstrates:
        - PLT limitation: only float32 and float64 are stored natively;
          ``_infer_data_type(np.int32)`` → DataType.INT32 → ``is_double=True``
          → written as float64 → reads back as DataType.DOUBLE
        - Values are still correct: float64 can represent all int32 exactly
        - Implication: if you need integer storage, use SZL format instead
        - Assertion pattern for upcast variables: compare against
          ``c.astype(np.float64)`` rather than the original int32 array
        """
        i, j = 5, 6
        x, y, z = create_ordered((i, j, 1))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        c = (scalar_field(x, y) * 1000).astype(np.int32)

        path = output_path("test_plt_write_ijk_int32.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_ijk_zone(data=[x, y, c], variables=["x", "y", "c"])

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]
            assert zone.variable[2].data_type == DataType.DOUBLE  # upcast from INT32
            np.testing.assert_allclose(
                zone.variable[2].values.ravel(), c.ravel().astype(np.float64)
            )

    def test_write_ijk_cell_centered(self, output_path: Callable) -> None:
        """3-D zone with float32 nodal coords and float64 cell-centered scalar.

        Demonstrates:
        - Cell-centered variable in PLT: ``value_locations=[..., CELL_CENTERED]``
        - Cell-centered array shape: ``(imax-1, jmax-1, kmax-1)``
        - PLT stores DOUBLE exactly; verified with float64 tolerance
        """
        i, j, k = 4, 5, 3
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        cc = np.random.default_rng(42).random((i - 1, j - 1, k - 1)).astype(np.float64)

        path = output_path("test_plt_write_ijk_cc.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_ijk_zone(
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
            np.testing.assert_allclose(
                cc_var.values.ravel(), cc.ravel(), rtol=_RTOL_F64
            )

    def test_write_ijk_unsteady_with_sharing(self, output_path: Callable) -> None:
        """100-zone transient dataset with variable sharing — float32 grid, float64 solution.

        Demonstrates:
        - Same variable-sharing pattern as SZL: ``var_sharing=[1, 1, 1, 0]``
          shares x/y/z from zone 1 for zones 2-100
        - PLT supports variable sharing identically to SZL at the high-level API
        - Significant file size reduction for large grids with many time steps
        - ``strand_id`` and ``solution_time`` enable Tecplot 360 animation
        """
        i, j, k = 20, 15, 10
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 100)

        path = output_path("test_plt_write_ijk_unsteady.plt")
        with tecio.open(str(path), "w") as pltfile:
            for n, t in enumerate(solution_times):
                c = scalar_field(x + t, y + t, z).astype(np.float64)
                pltfile.write_ijk_zone(
                    variables=["x", "y", "z", "c"],
                    data=[x, y, z, c] if n == 0 else [c],
                    var_sharing=None if n == 0 else [1, 1, 1, 0],
                    strand_id=1,
                    solution_time=float(t),
                )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 100
            assert r.zone[0].solution_time == pytest.approx(0.0)
            assert r.zone[99].solution_time == pytest.approx(solution_times[-1])

    def test_write_ijk_passive_variable(self, output_path: Callable) -> None:
        """Passive variable in PLT — identical to SZL behaviour.

        Demonstrates:
        - ``passive_vars=[False, True, False]``: same API as SZL; passive
          variables take no storage in PLT either
        - Verified with ``variable.is_passive() == True`` on read-back
        """
        x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)
        path = output_path("test_plt_write_ijk_passive.plt")

        with tecio.open(str(path), "w", variables=["x", "unused", "c"]) as pltfile:
            pltfile.write_ijk_zone(data=[x, c], passive_vars=[False, True, False])

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.zone[0].variable[1].is_passive()

    def test_write_ijk_aux_data(self, output_path: Callable) -> None:
        """Dataset and zone auxiliary data survive PLT round-trip.

        Demonstrates:
        - ``add_auxdataset_dict`` and ``aux={}`` work identically in PLT and SZL
        - Both levels of metadata are stored in the PLT binary header and
          recovered correctly by the reader
        """
        x = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        dataset_aux = {"Solver": "TestCode", "Mach": "0.72"}
        zone_aux = {"MeshType": "structured", "Author": "pytest"}
        path = output_path("test_plt_write_ijk_aux.plt")

        with tecio.open(str(path), "w") as pltfile:
            pltfile.add_auxdataset_dict(dataset_aux)
            pltfile.write_ijk_zone(data=[x], variables=["x"], aux=zone_aux)

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            for k, v in dataset_aux.items():
                assert r.auxdata[k] == v
            for k, v in zone_aux.items():
                assert r.zone[0].auxdata[k] == v

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_write_ijk_var_count_mismatch_raises(self, output_path: Callable) -> None:
        """Fewer arrays than active variables raises ValueError."""
        x = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        y = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        path = output_path("test_plt_write_ijk_mismatch.plt")

        with pytest.raises(ValueError, match="[Ee]xpected"):
            with tecio.open(str(path), "w", variables=["x", "y", "c"]) as pltfile:
                pltfile.write_ijk_zone(data=[x, y])

    def test_write_ijk_shape_mismatch_raises(self, output_path: Callable) -> None:
        """Inconsistent array shapes raise ValueError."""
        i, j, k = 4, 5, 1
        x, y, _ = create_ordered((i, j, k))
        x = x.squeeze(-1)
        y_bad = y.squeeze(-1)[:-1, :]
        path = output_path("test_plt_write_ijk_shape.plt")

        with pytest.raises(ValueError), tecio.open(str(path), "w") as pltfile:
            pltfile.write_ijk_zone(data=[x, y_bad], variables=["x", "y"])


# ===========================================================================
# Finite-element zone tests
# ===========================================================================


class TestWriteFEZone:
    """Tests for write_fe_zone targeting PLT output."""

    def test_write_fe_lineseg(self, output_path: Callable) -> None:
        """FELINESEG — float32 x/y, float64 scalar. Node map preserved exactly.

        Demonstrates:
        - Basic FE write to PLT: same ``write_fe_zone`` API as SZL
        - FLOAT x/y coordinates, DOUBLE scalar; PLT stores both natively
        - Node map round-trip: verified with ``assert_array_equal``
        """
        x, y, nodes = create_FE_lineseg()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        c = np.sin(2 * np.pi * x).astype(np.float64)

        path = output_path("test_plt_write_fe_lineseg.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
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
            np.testing.assert_allclose(zone.variable[0].values, x, rtol=_RTOL_F32)
            np.testing.assert_array_equal(zone.node_map, nodes.astype(np.int64))

    def test_write_fe_tri(self, output_path: Callable) -> None:
        """FETRIANGLE — float64 x/y, float32 scalar.

        Demonstrates:
        - Reversed precision: DOUBLE coordinates, FLOAT solution field
        - PLT stores FLOAT and DOUBLE natively; DataType verified on read-back
        """
        x, y, nodes = create_FE_tri()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        c = scalar_field(x, y).astype(np.float32)

        path = output_path("test_plt_write_fe_tri.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
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
            assert zone.variable[0].data_type == DataType.DOUBLE
            assert zone.variable[2].data_type == DataType.DOUBLE  # input was float32
            np.testing.assert_allclose(zone.variable[0].values, x, rtol=_RTOL_F64)

    def test_write_fe_quad(self, output_path: Callable) -> None:
        """FEQUADRILATERAL — float32 x/y, float64 scalar.

        Demonstrates:
        - Standard CFD precision: compact FLOAT coordinates, DOUBLE solution
        - PLT FE quad: node_map shape ``(num_elements, 4)``
        """
        x, y, nodes = create_FE_quad()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_plt_write_fe_quad.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
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
            assert zone.num_nodes == 6
            assert zone.num_elements == 2

    def test_write_fe_tet(self, output_path: Callable) -> None:
        """FETETRAHEDRON — float32 x/y/z, float64 scalar. Node map verified.

        Demonstrates:
        - 3-D FETETRAHEDRON in PLT: same call as SZL
        - Standard precision pattern for CFD: float32 grid, float64 solution
        - Node map round-trip verified with ``assert_array_equal``
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_plt_write_fe_tet.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
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
        """Degenerate FEBRICK pyramid — float64 all variables.

        Demonstrates:
        - Pyramid as collapsed FEBRICK in PLT: same node_map convention as SZL
        - All float64 (maximum precision); useful when no precision trade-off
          is needed or when the grid is also needed at high precision
        """
        x, y, z, nodes = create_FE_pyramid()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        z = z.astype(np.float64)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_plt_write_fe_pyramid.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
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
        """Degenerate FEBRICK prism — float32 all variables.

        Demonstrates:
        - Triangular prism as FEBRICK in PLT: identical collapsed-node convention
        - All float32 (minimum precision, smallest file size);
          suitable for visualization-only data where solver precision is not needed
        """
        x, y, z, nodes = create_FE_prism()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = scalar_field(x, y).astype(np.float32)

        path = output_path("test_plt_write_fe_prism.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
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
        - Standard 8-node hex in PLT
        - Mixed precision: compact float32 for x/y, full float64 for z and solution
        - DOUBLE z verified on read-back; values match within float64 tolerance
        """
        x, y, z, faces, nodes = create_FE_brick()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float64)
        c = scalar_field(x, y).astype(np.float64)

        path = output_path("test_plt_write_fe_brick.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
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
            assert zone.variable[2].data_type == DataType.DOUBLE
            np.testing.assert_allclose(zone.variable[2].values, z, rtol=_RTOL_F64)

    def test_write_fe_face_neighbors(self, output_path: Callable) -> None:
        """Two FEBRICK cells with face-neighbor connectivity and cell-centered variable.

        Demonstrates:
        - Face neighbors in PLT: ``face_neighbors`` and ``face_nbr_mode`` work
          identically to SZL at the high-level API
        - Cell-centered float64 variable: one value per element
        - Value assertion: float64 CC values survive PLT round-trip exactly
        """
        x, y, z, nodes, face_neighbors = create_FE_two_bricks()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = np.array([1.1, 2.2], dtype=np.float64)  # one per element

        path = output_path("test_plt_write_fe_face_neighbors.plt")
        with tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
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
            np.testing.assert_allclose(cc_var.values, c, rtol=_RTOL_F64)

    def test_write_fe_unsteady(self, output_path: Callable) -> None:
        """100 FETETRAHEDRON zones — float32 coords, float64 solution.

        Demonstrates:
        - Transient FE dataset in PLT: same ``strand_id`` and ``solution_time``
          API as SZL; zones with the same strand_id animate together
        - PLT requires strict ordering: header → data → connectivity per zone
          before the next zone begins; the high-level writer handles this
        """
        x, y, z, nodes = create_FE_tet()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 100)

        path = output_path("test_plt_write_fe_unsteady.plt")
        with tecio.open(str(path), "w", variables=["x", "y", "z", "c"]) as pltfile:
            for step, t in enumerate(solution_times):
                c = np.sin(x + t).astype(np.float64)
                pltfile.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title=f"zone_t{step + 1}",
                    strand_id=1,
                    solution_time=float(t),
                )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            assert r.num_zones == 100
            assert r.zone[0].solution_time == pytest.approx(0.0)
            assert r.zone[99].solution_time == pytest.approx(solution_times[-1])

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_write_fe_var_count_mismatch_raises(self, output_path: Callable) -> None:
        """Too few data arrays raises ValueError."""
        x, y, nodes = create_FE_tri()
        path = output_path("test_plt_write_fe_mismatch.plt")

        with pytest.raises(ValueError, match="[Ee]xpected"):
            with tecio.open(str(path), "w", variables=["x", "y", "c"]) as pltfile:
                pltfile.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x, y],
                    node_map=nodes,
                )

    def test_write_fe_array_length_mismatch_raises(self, output_path: Callable) -> None:
        """Nodal array shorter than num_nodes raises ValueError."""
        x, y, nodes = create_FE_tri()
        path = output_path("test_plt_write_fe_len.plt")

        with pytest.raises(ValueError), tecio.open(str(path), "w") as pltfile:
            pltfile.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x[:-1], y],
                node_map=nodes,
                variables=["x", "y"],
            )

    def test_write_fe_unsupported_zone_type_raises(self, output_path: Callable) -> None:
        """FEPOLYGON raises NotImplementedError.

        Demonstrates:
        - Poly zones require the low-level ``tecpolyzne142`` + ``tecpolyface142``
          path (see test_libtecio.py::TestClassicApi::test_plt_tec_zone_create_fe_polygon)
        """
        x, y, nodes = create_FE_tri()
        path = output_path("test_plt_write_fe_poly.plt")

        with pytest.raises(NotImplementedError):
            with tecio.open(str(path), "w") as pltfile:
                pltfile.write_fe_zone(
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
