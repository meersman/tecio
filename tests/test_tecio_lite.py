#!/usr/bin/env python3
"""Lite test for libtecio functions directly."""
# ruff: noqa: F403, F405

import numpy as np
import numpy.typing as npt

import tecio
from tecio.libtecio import DataType, FaceNeighborMode, FileType, ValueLocation, ZoneType


def _create_ordered(
    ijk: tuple[int, int, int],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Create ordered coordinates."""
    x_ = np.linspace(0.0, ijk[0], ijk[0])
    y_ = np.linspace(0.0, ijk[1], ijk[1])
    z_ = np.linspace(0.0, ijk[2], ijk[2])
    x, y = np.meshgrid(x_, y_, indexing="xy")
    x = np.array([x] * ijk[2])
    y = np.array([y] * ijk[2])
    z = np.repeat(z_, ijk[0] * ijk[1])
    return x, y, z


def _create_FE_lineseg() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FELINESEG zone type."""
    # xy coords
    points = np.array([
        [0, 0],  # point A
        [1, 1],  # point B
        [2, 0],  # Point C
    ])
    x = points[:, 0]
    y = points[:, 1]
    # Nodemap
    nodes = np.array([
        [1, 2],  # line seg AB
        [2, 3],  # line seg BC
    ])
    return x, y, nodes


def _create_FE_tri() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FELTRIANGLE zone type."""
    points = np.array([
        [0, 0],
        [1, 0],
        [0.5, 1],
        [1.5, 0.5],
    ])
    x = points[:, 0]
    y = points[:, 1]
    # Nodemap
    nodes = np.array([
        [1, 2, 3],
        [2, 4, 3],
    ])
    return x, y, nodes


def _create_FE_quad() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FEQUADRILATERAL zone type."""
    # Can create polygon from the same two FE tri cells above
    points = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 2],
    ])
    x = points[:, 0]
    y = points[:, 1]
    # Nodemap
    nodes = np.array([
        [1, 2, 5, 4],
        [2, 3, 6, 5],
    ])
    return x, y, nodes


def _create_FE_polygon() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FEPOLYGON zone type."""
    points = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 2],
    ])
    x = points[:, 0]
    y = points[:, 1]
    # Nodemap
    faces = np.array([
        [2, 1],  # Face 1
        [1, 3],  # Face 2
        [3, 2],  # Face 3
        [3, 4],  # Face 4
        [4, 2],  # Face 5
    ])
    return x, y, faces


def _create_FE_tet() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for FETETRAHEDRON zone type.
    Two tetrahedra sharing a face.
    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 1
        [1.0, 0.0, 0.0],  # 2
        [0.5, 1.0, 0.0],  # 3
        [0.5, 0.5, 1.0],  # 4 - apex of tet 1
        [0.5, 0.5, -1.0],  # 5 - apex of tet 2
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Each row: base triangle (CCW from outside) + apex
    nodes = np.array([
        [1, 2, 3, 4],  # tet 1 - apex above
        [1, 3, 2, 5],  # tet 2 - apex below
    ])
    return x, y, z, nodes


def _create_FE_pyramid() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for a pyramid using FEBRICK zone type.
    Tecplot represents pyramids as degenerate bricks where nodes 5,6,7,8
    are all the apex node repeated.
    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 1 - base
        [1.0, 0.0, 0.0],  # 2 - base
        [1.0, 1.0, 0.0],  # 3 - base
        [0.0, 1.0, 0.0],  # 4 - base
        [0.5, 0.5, 1.0],  # 5 - apex
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Repeat apex node (5) for the top 4 nodes of the brick
    nodes = np.array([
        [1, 2, 3, 4, 5, 5, 5, 5],
    ])
    return x, y, z, nodes


def _create_FE_prism() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for a triangular prism using FEBRICK.
    Represented as degenerate brick: bottom tri nodes 1,2,3 paired with
    top tri nodes 4,5,6 — each tri edge node repeated to fill 8-node brick.
    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 1 - bottom tri
        [1.0, 0.0, 0.0],  # 2 - bottom tri
        [0.5, 1.0, 0.0],  # 3 - bottom tri
        [0.0, 0.0, 1.0],  # 4 - top tri
        [1.0, 0.0, 1.0],  # 5 - top tri
        [0.5, 1.0, 1.0],  # 6 - top tri
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Bottom tri: 1,2,3,3 / Top tri: 4,5,6,6
    nodes = np.array([
        [1, 2, 3, 3, 4, 5, 6, 6],
    ])
    return x, y, z, nodes


def _create_FE_brick() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for FEBRICK zone type"""
    points = np.array([
        [0, 3, 0],  # XYZ
        [3, 3, 0],  # XYZ
        [3, 1, 0],  # XYZ
        [0, 1, 0],  # XYZ
        [0, 3, 1],  # XYZ
        [3, 3, 1],  # XYZ
        [3, 1, 1],  # XYZ
        [0, 1, 1],  # XYZ
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Nodemap
    faces = np.array([
        [1, 2, 3, 4],  # Face 1
        [1, 4, 8, 5],  # Face 2
        [5, 8, 7, 6],  # Face 3
        [2, 6, 7, 3],  # Face 4
        [6, 2, 1, 5],  # Face 5
        [3, 7, 8, 4],  # Face 6
    ])
    nodes = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    return x, y, z, faces, nodes


def _create_FE_two_bricks() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Two bricks sharing a face, with explicit face neighbor connectivity.

    Node Map:
      8 --- 7 --- 12
     /|    /|    /|
    5 --- 6 --- 11|
    | 4 --|-3 --|-9
    |/    |/    |/
    1 --- 2 --- 10

    Notes:
    - For the first FEBRICK cell shown above the faces are defined as
    f1: n1-n5-n8-n4 left face (Imin)
    f2: n2-n3-n7-n6 right face (Imax)
    f3: n1-n2-n6-n6 front face (Jmin)
    f4: n3-n4-n8-n7 back face (Jmax)
    f5: n1-n2-n3-n4 bottom face (Kmin)
    f6: n5-n6-n7-n8 top face (Kmax)

    """
    points = np.array([
            [0, 0, 0],  # 1
            [1, 0, 0],  # 2
            [1, 1, 0],  # 3
            [0, 1, 0],  # 4
            [0, 0, 1],  # 5
            [1, 0, 1],  # 6
            [1, 1, 1],  # 7
            [0, 1, 1],  # 8
            [2, 0, 0],  # 9
            [2, 1, 0],  # 10
            [2, 0, 1],  # 11
            [2, 1, 1],  # 12
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    nodes = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],  # cell 1
            [2, 9, 10, 3, 6, 11, 12, 7],  # cell 2
    ])
    # 6 faces per cell: bottom, top, front, back, left, right
    # 0 = boundary, positive int = 1-based neighbor cell index
    face_neighbors = np.array([
            [1, 2, 2],  # cell 1: right face neighbors cell 2
            [2, 1, 1],  # cell 2: left face neighbors cell 1
        ])
    return x, y, z, nodes, face_neighbors


def test_tec_zone_create_ijk() -> None:
    """Setup and test the functionality of tec_zone_create_ijk using minimal input"""
    try:
        # Create samople structured data
        i, j, k = (3, 4, 5)
        x, y, z = _create_ordered((i, j, k))
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_create_ijk.szplt",
            "test_tecio",
            "x, y, z, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_ijk(
            handle,
            "test_ordered_ijk",
            i,
            j,
            k,
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel())
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_ijk")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_ijk - {e}")


def test_tec_zone_create_fe_lineseg() -> None:
    """Setup and test the functionality of tec_zone_create_fe using minimal input"""
    try:
        # Create sample FELINESEG data
        x, y, nodes = _create_FE_lineseg()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_lineseg.szplt",
            "test_tecio",
            "x, y, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_fe(
            handle,
            "test_fe_lineseg",
            ZoneType.FELINESEG,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, c.ravel())
        tecio.libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel())
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_lineseg")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_lineseg - {e}")


def test_tec_zone_create_fe_tri() -> None:
    """Setup and test the functionality of tec_zone_create_fe using minimal input"""
    try:
        # Create sample FETRIANGLE data
        x, y, nodes = _create_FE_tri()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_tri.szplt",
            "test_tecio",
            "x, y, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_fe(
            handle,
            "test_fe_triangle",
            ZoneType.FETRIANGLE,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, c.ravel())
        tecio.libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel())
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_tri")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_tri - {e}")


def test_tec_zone_create_fe_quad() -> None:
    """Setup and test the functionality of tec_zone_create_fe using minimal input"""
    try:
        # Create sample FEQUADRILATERAL data
        x, y, nodes = _create_FE_quad()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_quad.szplt",
            "test_tecio",
            "x, y, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_fe(
            handle,
            "test_fe_quad",
            ZoneType.FEQUADRILATERAL,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, c.ravel())
        tecio.libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel())
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_quad")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_quad - {e}")


def test_tec_zone_create_fe_tet() -> None:
    """Test FETETRAHEDRON zone type"""
    try:
        x, y, z, nodes = _create_FE_tet()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_tet.szplt",
            "test_tecio",
            "x, y, z, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_fe(
            handle,
            "test_fe_tet",
            ZoneType.FETETRAHEDRON,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel())
        tecio.libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel())
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_tet")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_tet - {e}")


def test_tec_zone_create_fe_pyramid() -> None:
    """Test pyramid as degenerate FEBRICK zone type"""
    try:
        x, y, z, nodes = _create_FE_pyramid()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_pyramid.szplt",
            "test_tecio",
            "x, y, z, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_fe(
            handle,
            "test_fe_pyramid",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel())
        tecio.libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel())
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_pyramid")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_pyramid - {e}")


def test_tec_zone_create_fe_prism() -> None:
    """Test triangular prism as degenerate FEBRICK zone type"""
    try:
        x, y, z, nodes = _create_FE_prism()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_prism.szplt",
            "test_tecio",
            "x, y, z, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_fe(
            handle,
            "test_fe_prism",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel())
        tecio.libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel())
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_prism")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_prism - {e}")


def test_tec_zone_create_fe_brick() -> None:
    """Setup and test the functionality of tec_zone_create_fe using minimal input"""
    try:
        # Create sample FEBRICK data
        x, y, z, faces, nodes = _create_FE_brick()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_brick.szplt",
            "test_tecio",
            "x, y, z, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_fe(
            handle,
            "test_fe_brick",
            ZoneType.FEBRICK,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel())
        tecio.libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel())
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_brick")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_brick - {e}")


def test_tec_zone_face_nbr_write_connections() -> None:
    """Test face neighbor connections with two adjacent bricks"""
    try:
        x, y, z, nodes, face_neighbors = _create_FE_two_bricks()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_zone_face_nbr.szplt",
            "test_tecio",
            "x, y, z, c",
            FileType.FULL,
        )
        zone_idx = tecio.libtecio.tec_zone_create_fe(
            handle,
            "test_two_bricks",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
            num_face_cons=len(face_neighbors),
            face_nbr_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
        )
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel())
        tecio.libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel())
        tecio.libtecio.tec_zone_face_nbr_write_connections32(
            handle, zone_idx, face_neighbors.ravel()
        )
        tecio.libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_face_nbr_write_connections")
    except Exception as e:
        print(f"FAIL: test_tec_zone_face_nbr_write_connections - {e}")


def test_tec_zone_aux_data_and_unsteady() -> None:
    """Test dataset, variable, and zone auxiliary data plus unsteady options"""
    try:
        i, j, k = (3, 4, 5)
        x, y, z = _create_ordered((i, j, k))
        c = np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

        handle = tecio.libtecio.tec_file_writer_open(
            "test_tec_aux_data_unsteady.szplt",
            "test_tecio",
            "x, y, z, c",
            FileType.FULL,
        )

        # Dataset-level aux data
        tecio.libtecio.tec_data_set_add_aux_data(handle, "Author", "test_tecio_lite")
        tecio.libtecio.tec_data_set_add_aux_data(handle, "CreatedBy", "libtecio")
        tecio.libtecio.tec_data_set_add_aux_data(handle, "Version", "1.0")

        zone_idx = tecio.libtecio.tec_zone_create_ijk(
            handle,
            "test_aux_unsteady",
            i, j, k,
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )

        # Variable-level aux data
        tecio.libtecio.tec_var_add_aux_data(handle, 1, "Units", "meters")
        tecio.libtecio.tec_var_add_aux_data(handle, 2, "Units", "meters")
        tecio.libtecio.tec_var_add_aux_data(handle, 3, "Units", "meters")
        tecio.libtecio.tec_var_add_aux_data(handle, 4, "Units", "dimensionless")
        tecio.libtecio.tec_var_add_aux_data(handle, 4, "Description", "sin*cos field")

        # Zone-level aux data
        tecio.libtecio.tec_zone_add_aux_data(handle, zone_idx, "MeshType", "structured")
        tecio.libtecio.tec_zone_add_aux_data(handle, zone_idx, "IJK", f"{i}x{j}x{k}")

        # Unsteady options
        tecio.libtecio.tec_zone_set_unsteady_options(
            handle,
            zone_idx,
            solution_time=np.pi,
            strand=10,
        )

        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.flatten())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.flatten())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.flatten())
        tecio.libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.flatten())

        tecio.libtecio.tec_file_writer_close(handle)
        print(f"PASS: test_tec_zone_aux_data_and_unsteady")
    except Exception as e:
        print(f"FAIL: test_tec_zone_aux_data_and_unsteady - {e}")


if __name__ == "__main__":
    test_tec_zone_create_ijk()
    test_tec_zone_create_fe_lineseg()
    test_tec_zone_create_fe_tri()
    test_tec_zone_create_fe_quad()
    test_tec_zone_create_fe_tet()
    test_tec_zone_create_fe_pyramid()
    test_tec_zone_create_fe_prism()
    test_tec_zone_create_fe_brick()
    test_tec_zone_face_nbr_write_connections()
    test_tec_zone_aux_data_and_unsteady()
