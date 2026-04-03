#!/usr/bin/env python3
r"""Tests for the :class:`szl.Write` higher-level writing API.

Data-generation helpers are imported directly from ``test_libtecio`` so both test suites
always exercise the same geometric cases.
"""

import numpy as np
import tecio
from tecio.libtecio import ValueLocation, ZoneType, FaceNeighborMode

from create_test_data import *

#=======================================================================================
# Local functions to create all supported data formats
#=======================================================================================

def _scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Return a simple sin-cos scalar field over the supplied coordinate arrays."""
    if z is not None:
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1.0 + 0.1 * z)
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


#=======================================================================================
# IJK-ordered zone tests
#=======================================================================================

def test_write_ijk_3d() -> None:
    """Write a 3-D ordered zone (I, J, K all > 1).

    Demonstrates:
    - 3D Structured zone writing
    - Mixed nodal / cell-centred variables
    - Variable sharing across zones
    """
    try:
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        c = _scalar_field(x, y, z)

        # Cell-centred: (I-1) x (J-1) x (K-1)
        cc = np.random.rand(i - 1, j - 1, k - 1)

        with tecio.open("test_szl_write_ijk_3d.szplt", "w") as szlfile:
            szlfile.write_ijk_zone(
                data=[x, y, z, c],
                variables=["x", "y", "z", "c"],
                title="zone_3d",
            )
            szlfile.write_ijk_zone(
                data=[cc],
                title="zone_cc",
                var_sharing=[1, 1, 1, 0],  # share x,y,z from first zone
                value_locations=[ValueLocation.CELL_CENTERED],
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
    """
    try:
        i, j, k = 100, 50, 20
        x, y, z = create_ordered((i, j, k))

        solution_times = np.linspace(0.0, 2 * np.pi, 100)
        aux = {"MeshType": "structured", "Author": "test_szl_write"}

        with tecio.open("test_szl_write_ijk_unsteady.szplt", "w") as szlfile:
            for n, t in enumerate(solution_times):
                c = _scalar_field(x + t, y + t, z).astype(np.float32)
                if n == 0:
                    # On first write, write all variables
                    szlfile.write_ijk_zone(
                        data=[x, y, z, c],
                        variables=["x", "y", "z", "c"],
                        strand_id=1,
                        solution_time=t,
                        aux=aux,
                    )
                else:
                    # On subsequent writes, only write the changing variable (c)
                    szlfile.write_ijk_zone(
                        data=[c],
                        var_sharing=[1, 1, 1, 0],  # share x,y,z from first zone
                        strand_id=1,
                        solution_time=t,
                        aux=aux,
                    )
        print("PASS: test_write_ijk_unsteady")
    except Exception as e:
        print(f"FAIL: test_write_ijk_unsteady: {e}")


#---------------------------------------------------------------------------------------
# Exception-raising tests for invalid input data
#---------------------------------------------------------------------------------------

def test_write_ijk_var_count_mismatch() -> None:
    """write_ijk_zone must raise when data length != variable count.

    Demonstrates:
    - Variable count mismatch validateion for structured zone writer
    """
    try:
        i, j, k = 3, 3, 1
        x, y, _ = create_ordered((i, j, k))

        with tecio.open(
            "test_szl_write_ijk_var_mismatch.szplt",
            "w",
            title="mismatch_test",
            variables=["x", "y", "c"],  # 3 variables declared
        ) as szlfile:
            szlfile.write_ijk_zone(
                data=[x, y],  # only 2 arrays supplied
                title="zone_bad",
            )
            print(
                "FAIL: test_write_ijk_var_count_mismatch: expected ValueError, got none"
            )
    except ValueError:
        print("PASS: test_write_ijk_var_count_mismatch")
    except Exception as e:
        print(f"FAIL: test_write_ijk_var_count_mismatch: unexpected exception: {e}")


def test_write_ijk_shape_mismatch() -> None:
    """write_ijk_zone must raise when two nodal arrays have different shapes.

    Demonstrates:
    - Array shape mismatch validation for structured zone writer
    """
    try:
        i, j, k = 4, 5, 1
        x, y, _ = create_ordered((i, j, k))
        x = x.squeeze(0)  # shape (j, i) = (5, 4)
        y_bad = y.squeeze(0)[:-1, :]  # shape (4, 4) — wrong

        with tecio.open(
            "test_szl_write_ijk_shape_mismatch.szplt",
            "w",
            title="shape_test",
        ) as szlfile:
            szlfile.write_ijk_zone(
                data=[x, y_bad],
                title="zone_bad",
                variables=["x", "y"],
            )
        print("FAIL: test_write_ijk_shape_mismatch: expected ValueError, got none")
    except ValueError:
        print("PASS: test_write_ijk_shape_mismatch")
    except Exception as e:
        print(f"FAIL: test_write_ijk_shape_mismatch: unexpected exception: {e}")


#=======================================================================================
# FE zone tests—  one per zone type
#=======================================================================================

def test_write_fe_cells() -> None:
    """Write all FE cell shapes.

    Demonstrates:
    - All FE cell shapes (line seg, tri, quad, tet, pyramid, prism, brick)
    - Face neighbor connectivity for FE cells
    - Passive variable support
    - Mixed nodal / cell-centered variables
    """
    # Shape offset to view all at once
    offset = 2
    try:
        with tecio.open("test_szl_write_fe_cells.szplt", "w") as szlfile:

            # Write a FE line segement cell shape
            try:
                x, y, nodes = create_FE_lineseg()
                c = _scalar_field(x, y)
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FELINESEG,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_Line_Segment",
                    variables=["x", "y", "z", "c"],
                    passive_vars=[False, False, True, False]
                )
                print("PASS: test_write_fe_lineseg")
            except Exception as e:
                print(f"FAIL: test_write_fe_lineseg: {e}")

            # Write a FE triangle
            try:
                x, y, nodes = create_FE_tri()
                c = _scalar_field(x, y)
                x = x + offset
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_Triangle",
                    variables=["x", "y", "z", "c"],
                    passive_vars=[False, False, True, False]
                )
                print("PASS: test_write_fe_tri")
            except Exception as e:
                print(f"FAIL: test_write_fe_tri: {e}")

            # Write a FE quadrilateral
            try:
                x, y, nodes = create_FE_quad()
                c = _scalar_field(x, y)
                x = x + 2*offset
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FEQUADRILATERAL,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_Quadrilateral",
                    variables=["x", "y", "z", "c"],
                    passive_vars=[False, False, True, False]
                )
                print("PASS: test_write_fe_quad")
            except Exception as e:
                print(f"FAIL: test_write_fe_quad: {e}")

            # Write a FE tetrahedron
            try:
                x, y, z, nodes = create_FE_tet()
                c = _scalar_field(x, y)
                x = x + 3*offset
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Tet",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_tet")
            except Exception as e:
                print(f"FAIL: test_write_fe_tet: {e}")

            # Write a FE pyramid as a degenerate FEBRICK
            try:
                x, y, z, nodes = create_FE_pyramid()
                c = _scalar_field(x, y)
                x = x + 4*offset
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Pyramid",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_pyramid")
            except Exception as e:
                print(f"FAIL: test_write_fe_pyramid: {e}")

            # Write a FE prism as a degenerate FEBRICK
            try:
                x, y, z, nodes = create_FE_prism()
                c = _scalar_field(x, y)
                x = x + 5*offset
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Prism",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_prism")
            except Exception as e:
                print(f"FAIL: test_write_fe_prism: {e}")

            # Write a FEBRICK
            try:
                x, y, z, _faces, nodes = create_FE_brick()
                c = _scalar_field(x, y)
                x = x + 6*offset
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Brick",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_brick")
            except Exception as e:
                print(f"FAIL: test_write_fe_brick: {e}")

            # Write two adjacent FEBRICK cells with explicit face-neighbor connections
            try:
                x, y, z, nodes, face_neighbors = create_FE_two_bricks()
                c = np.array([1,2])
                x = x + 7*offset
                szlfile.write_fe_zone(
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

    Demonstrtes:
    - Strand ID and solution time for unsteady data
    - Zone-level auxiliary data

    """
    try:
        x, y, z, nodes = create_FE_tet()
        solution_times = np.linspace(0.0, 2 * np.pi, 100)

        with tecio.open(
            "test_szl_write_fe_unsteady.szplt",
            "w",
            title="fe_unsteady_test",
            variables=["x", "y", "z", "c"],
        ) as szlfile:
            for step, t in enumerate(solution_times):
                c = (np.sin(x + t) * np.cos(y + t))
                x = x + np.random.rand()/10
                y = y + np.random.rand()/10
                z = z + np.random.rand()/10
                szlfile.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title=f"zone_t{step + 1}",
                    strand_id=1,
                    solution_time=t,
                    aux={"MeshType": "unstructured", "Author": "test_szl_write"},
                )
        print("PASS: test_write_fe_unsteady")
    except Exception as e:
        print(f"FAIL: test_write_fe_unsteady: {e}")


#---------------------------------------------------------------------------------------
# Exception-raising tests for invalid input data
#---------------------------------------------------------------------------------------

def test_write_fe_var_count_mismatch() -> None:
    """write_fe_zone must raise ValueError when data length != variable count."""
    try:
        x, y, nodes = create_FE_tri()

        with tecio.open(
            "test_szl_write_fe_var_mismatch.szplt",
            "w",
            title="fe_mismatch_test",
            variables=["x", "y", "c"],  # 3 variables declared
        ) as szlfile:
            szlfile.write_fe_zone(
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
    """write_fe_zone must raise ValueError when a nodal array is the wrong length."""
    try:
        x, y, nodes = create_FE_tri()  # 4 nodes
        x_short = x[:-1]  # 3 values — one too few

        with tecio.open(
                "test_szl_write_fe_len_mismatch.szplt", "w", title="fe_len_test"
        ) as szlfile:
            szlfile.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x_short, y],
                node_map=nodes,
                title="zone_bad",
                variables=["x", "y"],
            )
        print("FAIL: test_write_fe_array_length_mismatch: expected ValueError, got none")
    except ValueError:
        print("PASS: test_write_fe_array_length_mismatch")
    except Exception as e:
        print(f"FAIL: test_write_fe_array_length_mismatch: unexpected exception: {e}")


def test_write_fe_unsupported_zone_type() -> None:
    """write_fe_zone must raise NotImplementedError for FEPOLYGON."""
    try:
        x, y, nodes = create_FE_tri()

        with tecio.open("test_szl_write_fe_polygon.szplt", "w") as szlfile:
            szlfile.write_fe_zone(
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
        print(f"FAIL: test_write_fe_unsupported_zone_type: unexpected exception: {e}")


#=======================================================================================
# Run all tests
#=======================================================================================
if __name__ == "__main__":
    # Ordered zone tests
    test_write_ijk_3d()
    test_write_ijk_unsteady()
    # FE zone tests
    test_write_fe_cells()
    test_write_fe_unsteady()
    # Zone validation tests (pass = raise expected exception)
    test_write_ijk_var_count_mismatch()
    test_write_ijk_shape_mismatch()
    test_write_fe_var_count_mismatch()
    test_write_fe_array_length_mismatch()
    test_write_fe_unsupported_zone_type()
