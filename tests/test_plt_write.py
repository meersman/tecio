#!/usr/bin/env python3
r"""Tests for the :class:`plt.Write` higher-level writing API.

This test suite is a 1-for-1 mirror of ``test_szl_write.py``.  Every test
exercises the same geometric case and the same capability as its SZL
counterpart, but targets the PLT format via :class:`plt.Write` instead of
:class:`szl.Write`.

Key differences from the SZL tests:

* Output filenames end in ``.plt`` instead of ``.szplt``.
* The PLT format does not support per-variable ``DataType`` selection at zone
  creation time — all numeric data is stored as ``float32`` or ``float64``
  on disk regardless of the NumPy array dtype.
* Variable sharing across zones is supported by ``teczne142`` but sharing
  is more constrained than in SZL: the shared source zone must already have
  been written and the library writes data strictly in order.
* Only one PLT file may be active at a time; tests that write to different
  files within a single ``with`` block are therefore not possible with the
  classic API.

Capabilities exercised:

* dimensionality  (1-D, 2-D, 3-D)
* data types      (float32, float64, mixed)
* value locations (all-nodal, mixed nodal / cell-centred)
* unsteady options (strand ID and solution time)
* zone auxiliary data
* lazy-open path  (variables supplied on first zone-write call)
* eager-open path (variables supplied to ``Write.__init__``)
* context-manager close
* variable-count mismatch guard
* array-shape mismatch guard

Data-generation helpers are shared with ``test_libtecio`` so both suites
always exercise identical geometric cases.
"""

import numpy as np

from tecio import plt
from tecio.libtecio import FaceNeighborMode, ValueLocation, ZoneType

from test_libtecio import (
    _create_FE_brick,
    _create_FE_lineseg,
    _create_FE_prism,
    _create_FE_pyramid,
    _create_FE_quad,
    _create_FE_tet,
    _create_FE_tri,
    _create_FE_two_bricks,
    _create_ordered,
)


# ===========================================================================
# Local helpers
# ===========================================================================

def _scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Return a simple sin-cos scalar field over the supplied coordinate arrays."""
    if z is not None:
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1.0 + 0.1 * z)
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


# ===========================================================================
# IJK-ordered zone tests
# ===========================================================================

def test_write_ijk_3d() -> None:
    """Write a 3-D ordered zone (I, J, K all > 1).

    Demonstrates:
    - 3-D structured zone writing
    - Mixed nodal / cell-centred variables
    """
    try:
        i, j, k = 3, 4, 5
        x, y, z = _create_ordered((i, j, k))
        c = _scalar_field(x, y, z)

        # Cell-centred array: shape (I-1) x (J-1) x (K-1)
        cc = np.random.rand(i - 1, j - 1, k - 1)

        with plt.Write("test_plt_write_ijk_3d.plt", title="3D_test") as writer:
            writer.write_ijk_zone(
                data=[x, y, z, c],
                variables=["x", "y", "z", "c"],
                title="zone_3d",
            )
            writer.write_ijk_zone(
                data=[x, y, z, cc],
                title="zone_cc",
                value_locations=[
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.CELL_CENTERED,
                ],
            )
        print("PASS: test_write_ijk_3d")
    except Exception as e:
        print(f"FAIL: test_write_ijk_3d: {e}")


def test_write_ijk_unsteady() -> None:
    """Write multiple zones representing a transient solution (strand + time).

    Demonstrates:
    - Strand ID and solution time for unsteady data
    - Zone-level auxiliary data
    - Variable sharing across zones

    Note:
        In the PLT classic API, variable sharing (``var_sharing``) is passed
        as part of the zone header via ``teczne142``.  The first zone in a
        strand must supply all variable data; subsequent zones may share
        coordinate variables from zone 1 and supply only the changing field.
    """
    try:
        i, j, k = 100, 50, 20
        x, y, z = _create_ordered((i, j, k))

        solution_times = np.linspace(0.0, 2 * np.pi, 100)
        aux = {"MeshType": "structured", "Author": "test_plt_write"}

        with plt.Write("test_plt_write_ijk_unsteady.plt") as writer:
            for n, t in enumerate(solution_times):
                c = _scalar_field(x + t, y + t, z).astype(np.float32)
                if n == 0:
                    # First write: supply all variable arrays
                    writer.write_ijk_zone(
                        data=[x, y, z, c],
                        variables=["x", "y", "z", "c"],
                        strand_id=1,
                        solution_time=t,
                        aux=aux,
                    )
                else:
                    # Subsequent writes: share x, y, z from zone 1; write only c
                    writer.write_ijk_zone(
                        data=[c],
                        var_sharing=[1, 1, 1, 0],  # share x,y,z from zone 1
                        strand_id=1,
                        solution_time=t,
                        aux=aux,
                    )
        print("PASS: test_write_ijk_unsteady")
    except Exception as e:
        print(f"FAIL: test_write_ijk_unsteady: {e}")


# ---------------------------------------------------------------------------
# Exception-raising tests for invalid input data
# ---------------------------------------------------------------------------

def test_write_ijk_var_count_mismatch() -> None:
    """write_ijk_zone must raise ValueError when data count != active variable count.

    Demonstrates:
    - Variable count mismatch validation for the structured zone writer
    """
    try:
        i, j, k = 3, 3, 1
        x, y, _ = _create_ordered((i, j, k))

        with plt.Write(
            "test_plt_write_ijk_var_mismatch.plt",
            title="mismatch_test",
            variables=["x", "y", "c"],  # 3 variables declared
        ) as writer:
            writer.write_ijk_zone(
                data=[x, y],  # only 2 arrays supplied
                title="zone_bad",
            )
            print(
                "FAIL: test_write_ijk_var_count_mismatch: "
                "expected ValueError, got none"
            )
    except ValueError:
        print("PASS: test_write_ijk_var_count_mismatch")
    except Exception as e:
        print(f"FAIL: test_write_ijk_var_count_mismatch: unexpected exception: {e}")


def test_write_ijk_shape_mismatch() -> None:
    """write_ijk_zone must raise ValueError when two nodal arrays differ in shape.

    Demonstrates:
    - Array shape mismatch validation for the structured zone writer
    """
    try:
        i, j, k = 4, 5, 1
        x, y, _ = _create_ordered((i, j, k))
        x = x.squeeze(0)        # shape (j, i) = (5, 4)
        y_bad = y.squeeze(0)[:-1, :]  # shape (4, 4) — wrong

        with plt.Write(
            "test_plt_write_ijk_shape_mismatch.plt",
            title="shape_test",
        ) as writer:
            writer.write_ijk_zone(
                data=[x, y_bad],
                title="zone_bad",
                variables=["x", "y"],
            )
        print("FAIL: test_write_ijk_shape_mismatch: expected ValueError, got none")
    except ValueError:
        print("PASS: test_write_ijk_shape_mismatch")
    except Exception as e:
        print(f"FAIL: test_write_ijk_shape_mismatch: unexpected exception: {e}")


# ===========================================================================
# FE zone tests — one per zone type
# ===========================================================================

def test_write_fe_cells() -> None:
    """Write all FE cell shapes.

    Demonstrates:
    - All FE cell shapes (line seg, tri, quad, tet, pyramid, prism, brick)
    - Face-neighbour connectivity
    - Mixed nodal / cell-centred variables

    Note:
        Unlike the SZL writer, the PLT classic API writes all zones to a
        single file sequentially.  The ``with`` block therefore covers every
        zone shape.
    """
    offset = 2
    try:
        with plt.Write("test_plt_write_fe_cells.plt") as writer:

            # FE line segment
            try:
                x, y, nodes = _create_FE_lineseg()
                c = _scalar_field(x, y)
                writer.write_fe_zone(
                    zone_type=ZoneType.FELINESEG,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_LineSeg",
                    variables=["x", "y", "c"],
                )
                print("PASS: test_write_fe_lineseg")
            except Exception as e:
                print(f"FAIL: test_write_fe_lineseg: {e}")

            # FE triangle
            try:
                x, y, nodes = _create_FE_tri()
                c = _scalar_field(x, y)
                x = x + offset
                writer.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_Tri",
                    variables=["x", "y", "c"],
                )
                print("PASS: test_write_fe_tri")
            except Exception as e:
                print(f"FAIL: test_write_fe_tri: {e}")

            # FE quadrilateral
            try:
                x, y, nodes = _create_FE_quad()
                c = _scalar_field(x, y)
                x = x + 2 * offset
                writer.write_fe_zone(
                    zone_type=ZoneType.FEQUADRILATERAL,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_Quad",
                    variables=["x", "y", "c"],
                )
                print("PASS: test_write_fe_quad")
            except Exception as e:
                print(f"FAIL: test_write_fe_quad: {e}")

            # FE tetrahedron
            try:
                x, y, z, nodes = _create_FE_tet()
                c = _scalar_field(x, y)
                x = x + 3 * offset
                writer.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Tet",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_tet")
            except Exception as e:
                print(f"FAIL: test_write_fe_tet: {e}")

            # FE pyramid as degenerate FEBRICK
            try:
                x, y, z, nodes = _create_FE_pyramid()
                c = _scalar_field(x, y)
                x = x + 4 * offset
                writer.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Pyramid",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_pyramid")
            except Exception as e:
                print(f"FAIL: test_write_fe_pyramid: {e}")

            # FE triangular prism as degenerate FEBRICK
            try:
                x, y, z, nodes = _create_FE_prism()
                c = _scalar_field(x, y)
                x = x + 5 * offset
                writer.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Prism",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_prism")
            except Exception as e:
                print(f"FAIL: test_write_fe_prism: {e}")

            # FEBRICK
            try:
                x, y, z, _faces, nodes = _create_FE_brick()
                c = _scalar_field(x, y)
                x = x + 6 * offset
                writer.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Brick",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_brick")
            except Exception as e:
                print(f"FAIL: test_write_fe_brick: {e}")

            # Two adjacent FEBRICKs with explicit face-neighbour connections
            try:
                x, y, z, nodes, face_neighbors = _create_FE_two_bricks()
                c = np.array([1, 2])
                x = x + 7 * offset
                writer.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_2Bricks",
                    variables=["x", "y", "z", "c"],
                    value_locations=[
                        ValueLocation.NODAL,
                        ValueLocation.NODAL,
                        ValueLocation.NODAL,
                        ValueLocation.CELL_CENTERED,
                    ],
                    face_neighbors=face_neighbors,
                    face_nbr_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
                )
                print("PASS: test_write_fe_face_neighbors")
            except Exception as e:
                print(f"FAIL: test_write_fe_face_neighbors: {e}")

    except Exception as e:
        print(f"FAIL: test_write_fe_cells: {e}")


def test_write_fe_unsteady() -> None:
    """Write multiple FE zones with strand ID and solution time.

    Demonstrates:
    - Strand ID and solution time for unsteady FE data
    - Zone-level auxiliary data on every time step
    """
    try:
        x, y, z, nodes = _create_FE_tet()
        solution_times = np.linspace(0.0, 2 * np.pi, 100)

        with plt.Write(
            "test_plt_write_fe_unsteady.plt",
            title="fe_unsteady_test",
            variables=["x", "y", "z", "c"],
        ) as writer:
            for step, t in enumerate(solution_times):
                c = np.sin(x + t) * np.cos(y + t)
                x = x + np.random.rand() / 10
                y = y + np.random.rand() / 10
                z = z + np.random.rand() / 10
                writer.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title=f"zone_t{step + 1}",
                    strand_id=1,
                    solution_time=t,
                    aux={"MeshType": "unstructured", "Author": "test_plt_write"},
                )
        print("PASS: test_write_fe_unsteady")
    except Exception as e:
        print(f"FAIL: test_write_fe_unsteady: {e}")


# ---------------------------------------------------------------------------
# Exception-raising tests for invalid input data
# ---------------------------------------------------------------------------

def test_write_fe_var_count_mismatch() -> None:
    """write_fe_zone must raise ValueError when data count != active variable count."""
    try:
        x, y, nodes = _create_FE_tri()

        with plt.Write(
            "test_plt_write_fe_var_mismatch.plt",
            title="fe_mismatch_test",
            variables=["x", "y", "c"],  # 3 variables declared
        ) as writer:
            writer.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, y],  # only 2 arrays
                node_map=nodes,
                title="zone_bad",
            )
        print("FAIL: test_write_fe_var_count_mismatch: expected ValueError, got none")
    except ValueError:
        print("PASS: test_write_fe_var_count_mismatch")
    except Exception as e:
        print(f"FAIL: test_write_fe_var_count_mismatch: unexpected exception: {e}")


def test_write_fe_array_length_mismatch() -> None:
    """write_fe_zone must raise ValueError when a nodal array has the wrong length."""
    try:
        x, y, nodes = _create_FE_tri()  # 4 nodes
        x_short = x[:-1]               # 3 values — one too few

        with plt.Write(
            "test_plt_write_fe_len_mismatch.plt", title="fe_len_test"
        ) as writer:
            writer.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x_short, y],
                node_map=nodes,
                title="zone_bad",
                variables=["x", "y"],
            )
        print(
            "FAIL: test_write_fe_array_length_mismatch: expected ValueError, got none"
        )
    except ValueError:
        print("PASS: test_write_fe_array_length_mismatch")
    except Exception as e:
        print(
            f"FAIL: test_write_fe_array_length_mismatch: unexpected exception: {e}"
        )


def test_write_fe_unsupported_zone_type() -> None:
    """write_fe_zone must raise NotImplementedError for FEPOLYGON."""
    try:
        x, y, nodes = _create_FE_tri()

        with plt.Write(
            "test_plt_write_fe_polygon.plt", title="fe_polygon_test"
        ) as writer:
            writer.write_fe_zone(
                zone_type=ZoneType.FEPOLYGON,
                data=[x, y],
                node_map=nodes,
                title="zone_polygon",
                variables=["x", "y"],
            )
        print(
            "FAIL: test_write_fe_unsupported_zone_type: "
            "expected NotImplementedError, got none"
        )
    except NotImplementedError:
        print("PASS: test_write_fe_unsupported_zone_type")
    except Exception as e:
        print(
            f"FAIL: test_write_fe_unsupported_zone_type: unexpected exception: {e}"
        )


# ===========================================================================
# Run all tests
# ===========================================================================
if __name__ == "__main__":
    # Ordered zone tests
    test_write_ijk_3d()
    test_write_ijk_unsteady()
    # FE zone tests
    test_write_fe_cells()
    test_write_fe_unsteady()
    # Validation tests (PASS = expected exception raised)
    test_write_ijk_var_count_mismatch()
    test_write_ijk_shape_mismatch()
    test_write_fe_var_count_mismatch()
    test_write_fe_array_length_mismatch()
    test_write_fe_unsupported_zone_type()
