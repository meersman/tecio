#!/usr/bin/env python3
r"""Tests for the :class:`dat.Write` higher-level writing API."""

import numpy as np
import tecio
from tecio.libtecio import FaceNeighborMode, ValueLocation, ZoneType

from create_test_data import *


#=======================================================================================
# IJK-ordered zone tests
#=======================================================================================

def test_write_ijk_3d() -> None:
    """Write a 3D ordered zone (I, J, K all > 1).

    Demonstrates:
    - 3D Structured zone writing
    - Mixed nodal / cell-centred variables
    - Variable sharing across zones
    """
    try:
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        c = scalar_field(x, y, z)

        # Cell-centered array: shape (I-1) x (J-1) x (K-1)
        cc = np.random.rand(i - 1, j - 1, k - 1)

        with tecio.open("test_dat_write_ijk_3d.dat", "w") as datfile:
            datfile.write_ijk_zone(
                data=[x, y, z, c],
                variables=["x", "y", "z", "c"],
                title="zone_3d",
            )
            datfile.write_ijk_zone(
                data=[cc],
                title="zone_cc",
                var_sharing=[1, 1, 1, 0],  # share x,y,z from first zone
                value_locations=[ValueLocation.CELL_CENTERED],
            )
        print("PASS: test_write_ijk_3d")
    except Exception as exc:
        print(f"FAIL: test_write_ijk_3d: {exc}")


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
        aux = {"MeshType": "structured", "Author": "test_dat_write"}

        with tecio.open("test_dat_write_ijk_unsteady.dat", "w") as datfile:
            for n, t in enumerate(solution_times):
                c = scalar_field(x + t, y + t, z).astype(np.float32)
                if n == 0:
                    # On first write: supply all variable arrays
                    datfile.write_ijk_zone(
                        data=[x, y, z, c],
                        variables=["x", "y", "z", "c"],
                        strand_id=1,
                        solution_time=t,
                        aux=aux,
                    )
                else:
                    # On subsequent writes, only write the changing variable (c)
                    datfile.write_ijk_zone(
                        data=[c],
                        var_sharing=[1, 1, 1, 0],  # share x,y,z from zone 1
                        strand_id=1,
                        solution_time=t,
                        aux=aux,
                    )
        print("PASS: test_write_ijk_unsteady")
    except Exception as exc:
        print(f"FAIL: test_write_ijk_unsteady: {exc}")


#---------------------------------------------------------------------------------------
# Exception-raising tests for invalid input data
#---------------------------------------------------------------------------------------

def test_write_ijk_var_count_mismatch() -> None:
    """write_ijk_zone must raise ValueError when data count != active variable count.

    Demonstrates:
    - Variable count mismatch validation for the structured zone writer
    """
    try:
        i, j, k = 3, 3, 1
        x, y, _ = create_ordered((i, j, k))

        with tecio.open(
            "test_dat_write_ijk_var_mismatch.dat",
            "w",
            title="mismatch_test",
            variables=["x", "y", "c"],  # 3 variables declared
        ) as datfile:
            datfile.write_ijk_zone(
                data=[x, y],  # only 2 arrays supplied
                title="zone_bad",
            )
            print(
                "FAIL: test_write_ijk_var_count_mismatch: expected ValueError, got none"
            )
    except ValueError:
        print("PASS: test_write_ijk_var_count_mismatch")
    except Exception as exc:
        print(f"FAIL: test_write_ijk_var_count_mismatch: unexpected exception: {exc}")


def test_write_ijk_shape_mismatch() -> None:
    """write_ijk_zone must raise ValueError when two nodal arrays differ in shape.

    Demonstrates:
    - Array shape mismatch validation for the structured zone writer
    """
    try:
        i, j, k = 4, 5, 1
        x, y, _ = create_ordered((i, j, k))
        x = x.squeeze(0)  # shape (j, i) = (5, 4)
        y_bad = y.squeeze(0)[:-1, :]  # shape (4, 4) — wrong

        with tecio.open(
            "test_dat_write_ijk_shape_mismatch.dat",
            "w",
            title="shape_test",
        ) as datfile:
            datfile.write_ijk_zone(
                data=[x, y_bad],
                title="zone_bad",
                variables=["x", "y"],
            )
        print("FAIL: test_write_ijk_shape_mismatch: expected ValueError, got none")
    except ValueError:
        print("PASS: test_write_ijk_shape_mismatch")
    except Exception as exc:
        print(f"FAIL: test_write_ijk_shape_mismatch: unexpected exception: {exc}")


#=======================================================================================
# FE zone tests — one per zone type
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
        with tecio.open("test_dat_write_fe_cells.dat", "w") as datfile:

            # Write a FE line segment
            try:
                x, y, nodes = create_FE_lineseg()
                c = scalar_field(x, y)
                datfile.write_fe_zone(
                    zone_type=ZoneType.FELINESEG,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_LineSeg",
                    variables=["x", "y", "z", "c"],
                    passive_vars=[False, False, True, False]
                )
                print("PASS: test_write_fe_lineseg")
            except Exception as exc:
                print(f"FAIL: test_write_fe_lineseg: {exc}")

            # Write a FE triangle
            try:
                x, y, nodes = create_FE_tri()
                c = scalar_field(x, y)
                x = x + offset
                datfile.write_fe_zone(
                    zone_type=ZoneType.FETRIANGLE,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_Tri",
                    variables=["x", "y", "z", "c"],
                    passive_vars=[False, False, True, False]
                )
                print("PASS: test_write_fe_tri")
            except Exception as exc:
                print(f"FAIL: test_write_fe_tri: {exc}")

            # Write a FE quadrilateral
            try:
                x, y, nodes = create_FE_quad()
                c = scalar_field(x, y)
                x = x + 2*offset
                datfile.write_fe_zone(
                    zone_type=ZoneType.FEQUADRILATERAL,
                    data=[x, y, c],
                    node_map=nodes,
                    title="FE_Quad",
                    variables=["x", "y", "z", "c"],
                    passive_vars=[False, False, True, False]
                )
                print("PASS: test_write_fe_quad")
            except Exception as exc:
                print(f"FAIL: test_write_fe_quad: {exc}")

            # Write a FE tetrahedron
            try:
                x, y, z, nodes = create_FE_tet()
                c = scalar_field(x, y)
                x = x + 3*offset
                datfile.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Tet",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_tet")
            except Exception as exc:
                print(f"FAIL: test_write_fe_tet: {exc}")

            # Write a FE pyramid as degenerate FEBRICK
            try:
                x, y, z, nodes = create_FE_pyramid()
                c = scalar_field(x, y)
                x = x + 4*offset
                datfile.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Pyramid",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_pyramid")
            except Exception as exc:
                print(f"FAIL: test_write_fe_pyramid: {exc}")

            # Write a FE prism as degenerate FEBRICK
            try:
                x, y, z, nodes = create_FE_prism()
                c = scalar_field(x, y)
                x = x + 5*offset
                datfile.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Prism",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_prism")
            except Exception as exc:
                print(f"FAIL: test_write_fe_prism: {exc}")

            # Write a FEBRICK
            try:
                x, y, z, _faces, nodes = create_FE_brick()
                c = scalar_field(x, y)
                x = x + 6*offset
                datfile.write_fe_zone(
                    zone_type=ZoneType.FEBRICK,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title="FE_Brick",
                    variables=["x", "y", "z", "c"],
                )
                print("PASS: test_write_fe_brick")
            except Exception as exc:
                print(f"FAIL: test_write_fe_brick: {exc}")

            # Write two adjacent FEBRICK cells with explicit face-neighbor connections
            try:
                x, y, z, nodes, face_neighbors = create_FE_two_bricks()
                c = np.array([1, 2])
                x = x + 7*offset
                datfile.write_fe_zone(
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
            except Exception as exc:
                print(f"FAIL: test_write_fe_face_neighbors: {exc}")

    except Exception as exc:
        print(f"FAIL: test_write_fe_cells: {exc}")


def test_write_fe_unsteady() -> None:
    """Write multiple FE zones with strand ID and solution time.

    Demonstrates:
    - Strand ID and solution time for unsteady data
    - Zone-level auxiliary data
    """
    try:
        x, y, z, nodes = create_FE_tet()
        solution_times = np.linspace(0.0, 2 * np.pi, 100)

        with tecio.open(
            "test_dat_write_fe_unsteady.dat",
            "w",
            title="fe_unsteady_test",
            variables=["x", "y", "z", "c"],
        ) as datfile:
            for step, t in enumerate(solution_times):
                c = np.sin(x + t) * np.cos(y + t)
                x = x + np.random.rand()/10
                y = y + np.random.rand()/10
                z = z + np.random.rand()/10
                datfile.write_fe_zone(
                    zone_type=ZoneType.FETETRAHEDRON,
                    data=[x, y, z, c],
                    node_map=nodes,
                    title=f"zone_t{step + 1}",
                    strand_id=1,
                    solution_time=t,
                    aux={"MeshType": "unstructured", "Author": "test_dat_write"},
                )
        print("PASS: test_write_fe_unsteady")
    except Exception as exc:
        print(f"FAIL: test_write_fe_unsteady: {exc}")


#---------------------------------------------------------------------------------------
# Exception-raising tests for invalid input data
#---------------------------------------------------------------------------------------

def test_write_fe_var_count_mismatch() -> None:
    """write_fe_zone must raise ValueError when data count != active variable count."""
    try:
        x, y, nodes = create_FE_tri()

        with tecio.open(
            "test_dat_write_fe_var_mismatch.dat",
            "w",
            title="fe_mismatch_test",
            variables=["x", "y", "c"],  # 3 variables declared
        ) as datfile:
            datfile.write_fe_zone(
                zone_type=ZoneType.FETRIANGLE,
                data=[x, y],  # only 2 arrays
                node_map=nodes,
                title="zone_bad",
            )
        print("FAIL: test_write_fe_var_count_mismatch: expected ValueError, got none")
    except ValueError:
        print("PASS: test_write_fe_var_count_mismatch")
    except Exception as exc:
        print(f"FAIL: test_write_fe_var_count_mismatch: unexpected exception: {exc}")


def test_write_fe_array_length_mismatch() -> None:
    """write_fe_zone must raise ValueError when a nodal array has the wrong length."""
    try:
        x, y, nodes = create_FE_tri()  # 4 nodes
        x_short = x[:-1]  # 3 values — one too few

        with tecio.open("test_dat_write_fe_len_mismatch.dat", "w") as datfile:
            datfile.write_fe_zone(
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
    except Exception as exc:
        print(f"FAIL: test_write_fe_array_length_mismatch: unexpected exception: {exc}")


def test_write_fe_unsupported_zone_type() -> None:
    """write_fe_zone must raise NotImplementedError for FEPOLYGON."""
    try:
        x, y, nodes = create_FE_tri()

        with tecio.open("test_dat_write_fe_polygon.dat", "w") as datfile:
            datfile.write_fe_zone(
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
    except Exception as exc:
        print(f"FAIL: test_write_fe_unsupported_zone_type: unexpected exception: {exc}")


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
