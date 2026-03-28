#!/usr/bin/env python3
"""Lite test for libtecio functions directly."""

# ruff: noqa: F403, F405

import numpy as np
import numpy.typing as npt

from tecio import libtecio
from tecio.libtecio import (
    DataType,
    FaceNeighborMode,
    FileFormat,
    FileType,
    ValueLocation,
    ZoneType,
)


#=======================================================================================
# Local functions to create all supported data formats
#=======================================================================================
def _create_ordered(
    ijk: tuple[int, ...],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Create ordered coordinates.

    Example layout for (3, 4, 1) - single XY plane viewed from above:

    j=4  10 -- 11 -- 12
         |     |     |
    j=3   7 ---8 --- 9
          |    |     |
    j=2   4 ---5 --- 6
          |    |     |
    j=1   1 ---2 --- 3
         i=1  i=2   i=3

    Node ordering: i (x) varies fastest, then j (y), then k (z).
    """
    x_ = np.linspace(0.0, ijk[0], ijk[0], dtype=np.float32)
    y_ = np.linspace(0.0, ijk[1], ijk[1], dtype=np.float32)
    z_ = np.linspace(0.0, ijk[2], ijk[2], dtype=np.float32)
    x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
    return x, y, z


def _create_FE_lineseg() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FELINESEG zone type.

    Node Map:
    [1]--<C1>--[2]--<C2>--[3]
    """
    # xy coords
    points = np.array([
        [0, 0],  # point A
        [0.5, 1],  # point B
        [1, 0],  # Point C
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
    r"""Create coordinates and nodemap for FELTRIANGLE zone type.

    Node Map: Two triangles sharing edge 2-3:
    3 --- 4  Cell 1: 1-2-3
    |\ C2 |  Cell 2: 2-4-3
    | \   |
    |C1 \ |
    1 --- 2
    """
    points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
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
    """Create coordinates and nodemap for FEQUADRILATERAL zone type.

    Node Map: Two quads sharing edge 2-5:
    4 --- 5 --- 6  Cell 1: 1-2-5-4
    | C1  | C2  |  Cell 2: 2-3-6-5
    1 --- 2 --- 3
    """
    # Can create polygon from the same two FE tri cells above
    points = np.array([
        [0, 0],
        [0.5, 0],
        [1, 0],
        [0, 0.3],
        [0.5, 0.6],
        [1, 1],
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
    r"""Create coordinates and nodemap for FETETRAHEDRON zone type.

    Node Map: Two tetrahedra sharing base face 1-2-3, apices above (4) and below (5):

        4        <- apex tet 1 (z > 0)
       /|\
      / | \
     /  |  \
    1---+---2   <- shared base triangle
     \  |  /
      \ | /
       \|/
        5        <- apex tet 2 (z < 0)

    Cell 1: [1, 2, 3, 4]
    Cell 2: [1, 3, 2, 5]
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

    Node: Tecplot represents pyramids as degenerate bricks where nodes 5,6,7,8
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

    Note: Represented as degenerate brick: bottom tri nodes 1,2,3 paired with
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
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for FEBRICK zone type."""
    points = np.array([
        [0, 0, 0],  # 1
        [1, 0, 0],  # 2
        [1, 1, 0],  # 3
        [0, 1, 0],  # 4
        [0, 0, 1],  # 5
        [1, 0, 1],  # 6
        [1, 1, 1],  # 7
        [0, 1, 1],  # 8
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
        [1, 2, 2],  # cell 1: right face neighbors cell 2 (cz1, fz, cz2)
        [2, 1, 1],  # cell 2: left face neighbors cell 1
    ])
    return x, y, z, nodes, face_neighbors


def _create_FE_mixed() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create a mixed FE zone with one tet, one pyramid, one prism, one hex.

    All cells share a common base region, each demonstrating a different
    3D element type. Returns x, y, z coordinates, concatenated node map,
    and num_nodes_per_element array (one entry per cell).

    Cell layout:
        Cell 1: Tet       (4 nodes)
        Cell 2: Pyramid   (5 nodes)
        Cell 3: Prism     (6 nodes)
        Cell 4: Hex       (8 nodes)
    """
    points = np.array(
        [
            # Base layer (z=0)
            [0.0, 0.0, 0.0],  #  1
            [1.0, 0.0, 0.0],  #  2
            [1.0, 1.0, 0.0],  #  3
            [0.0, 1.0, 0.0],  #  4
            [2.0, 0.0, 0.0],  #  5
            [2.0, 1.0, 0.0],  #  6
            [3.0, 0.0, 0.0],  #  7
            [3.0, 1.0, 0.0],  #  8
            [4.0, 0.0, 0.0],  #  9
            [4.0, 1.0, 0.0],  # 10
            # Mid layer (z=1)
            [2.0, 0.0, 1.0],  # 11
            [2.0, 1.0, 1.0],  # 12
            [3.0, 0.0, 1.0],  # 13
            [3.0, 1.0, 1.0],  # 14
            [4.0, 0.0, 1.0],  # 15
            [4.0, 1.0, 1.0],  # 16
            # Apex nodes
            [0.5, 0.5, 1.0],  # 17 - tet apex
            [1.5, 0.5, 1.0],  # 18 - pyramid apex
        ],
        dtype=np.float32,
    )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Cell 1: Tet - base triangle 1,2,3 + apex 17
    # Cell 2: Pyramid - base quad 2,5,6,3 + apex 18
    # Cell 3: Prism - bottom tri 5,7,6 + top tri 11,13,12
    # Cell 4: Hex - bottom quad 7,9,10,8 + top quad 13,15,16,14
    node_map = np.array(
        [
            1, 2, 3, 17,  # tet: 4 nodes
            2, 5, 6, 3, 18,  # pyramid: 5 nodes
            5, 7, 6, 11, 13, 12,  # prism: 6 nodes
            7, 9, 10, 8, 13, 15, 16, 14,  # hex: 8 nodes
        ],
        dtype=np.int32,
    )

    # One entry per cell: number of nodes in that cell
    num_nodes_per_element = np.array([4, 5, 6, 8], dtype=np.int32)

    return x, y, z, node_map, num_nodes_per_element


#=======================================================================================
# MVP style tests for each data type
#=======================================================================================

#---------------------------------------------------------------------------------------
# New API (SZL/.szplt):
# - tec_file_writer_open returns an explicit file handle that is passed to every
#   subsequent call, making the target file unambiguous
# - Multiple files can be written simultaneously by holding multiple handles at once
#   tec_zone_create_ijk / tec_zone_create_fe append a new zone record and return a
#   1-based zone index
# - Variable data, node maps, aux data, and unsteady options are written by referencing
#   the file handle + zone/variable index, so they can be written in any order after
#   zone creation
# - tec_file_writer_close finalizes and flushes the file
#---------------------------------------------------------------------------------------

def test_tec_zone_create_ijk() -> None:
    """Test ORDERED zone type."""
    try:
        # Create samople structured data
        i, j, k = (3, 4, 5)
        x, y, z = _create_ordered((i, j, k))
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_ijk.szplt",
            variables=["x", "y", "z", "c"],
        )
        zone_idx = libtecio.tec_zone_create_ijk(
            handle,
            "test_ordered_ijk",
            i,
            j,
            k,
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_ijk")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_ijk: {e}")


def test_tec_zone_create_fe_lineseg() -> None:
    """Test FELINESEG zone type."""
    try:
        # Create sample FELINESEG data
        x, y, nodes = _create_FE_lineseg()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_lineseg.szplt",
            variables=["x", "y","c"],
        )
        zone_idx = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_lineseg",
            ZoneType.FELINESEG,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_lineseg")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_lineseg: {e}")


def test_tec_zone_create_fe_tri() -> None:
    """Test FETRIANGLE zone type."""
    try:
        # Create sample FETRIANGLE data
        x, y, nodes = _create_FE_tri()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_tri.szplt",
            variables=["x", "y", "c"],
        )
        zone_idx = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_triangle",
            ZoneType.FETRIANGLE,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_tri")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_tri: {e}")


def test_tec_zone_create_fe_quad() -> None:
    """Test FEQUADRILATERAL zone type."""
    try:
        # Create sample FEQUADRILATERAL data
        x, y, nodes = _create_FE_quad()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_quad.szplt",
            variables=["x", "y", "c"],
        )
        zone_idx = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_quad",
            ZoneType.FEQUADRILATERAL,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_quad")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_quad: {e}")


def test_tec_zone_create_fe_tet() -> None:
    """Test FETETRAHEDRON zone type."""
    try:
        x, y, z, nodes = _create_FE_tet()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_tet.szplt",
            variables=["x", "y", "z", "c"],
        )
        zone_idx = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_tet",
            ZoneType.FETETRAHEDRON,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_tet")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_tet: {e}")


def test_tec_zone_create_fe_pyramid() -> None:
    """Test pyramid as degenerate FEBRICK zone type."""
    try:
        x, y, z, nodes = _create_FE_pyramid()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_pyramid.szplt",
            variables=["x", "y", "z", "c"],
        )
        zone_idx = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_pyramid",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_pyramid")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_pyramid: {e}")


def test_tec_zone_create_fe_prism() -> None:
    """Test triangular prism as degenerate FEBRICK zone type."""
    try:
        x, y, z, nodes = _create_FE_prism()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_prism.szplt",
            variables=["x", "y", "z", "c"],
        )
        zone_idx = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_prism",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_prism")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_prism: {e}")


def test_tec_zone_create_fe_brick() -> None:
    """Test FEBRICK zone type."""
    try:
        # Create sample FEBRICK data
        x, y, z, faces, nodes = _create_FE_brick()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_brick.szplt",
            variables=["x", "y", "z", "c"],
        )
        zone_idx = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_brick",
            ZoneType.FEBRICK,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_brick")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_brick: {e}")


def test_tec_zone_face_nbr_write_connections() -> None:
    """Test face neighbor connections with two adjacent bricks."""
    try:
        x, y, z, nodes, face_neighbors = _create_FE_two_bricks()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_face_nbr.szplt",
            variables=["x", "y", "z", "c"],
        )
        zone_idx = libtecio.tec_zone_create_fe(
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
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, zone_idx, nodes.ravel(order="F"))
        libtecio.tec_zone_face_nbr_write_connections32(
            handle, zone_idx, face_neighbors.ravel(order="F")
        )
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_face_nbr_write_connections")
    except Exception as e:
        print(f"FAIL: test_tec_zone_face_nbr_write_connections: {e}")


def test_tec_zone_aux_data_and_unsteady() -> None:
    """Test dataset, variable, and zone auxiliary data plus unsteady options."""
    try:
        i, j, k = (3, 4, 5)
        x, y, z = _create_ordered((i, j, k))
        solution_times = np.linspace(0.0, 2 * np.pi, 50)

        handle = libtecio.tec_file_writer_open(
            "test_tec_aux_data_unsteady.szplt",
            variables=["x", "y", "z", "c"],
        )

        # Dataset-level aux data
        libtecio.tec_data_set_add_aux_data(handle, "Author", "test_tecio_lite")
        libtecio.tec_data_set_add_aux_data(handle, "CreatedBy", "libtecio")
        libtecio.tec_data_set_add_aux_data(handle, "Version", "1.0")

        # Variable-level aux data
        libtecio.tec_var_add_aux_data(handle, 1, "Units", "meters")
        libtecio.tec_var_add_aux_data(handle, 2, "Units", "meters")
        libtecio.tec_var_add_aux_data(handle, 3, "Units", "meters")
        libtecio.tec_var_add_aux_data(handle, 4, "Units", "dimensionless")
        libtecio.tec_var_add_aux_data(handle, 4, "Description", "sin*cos field")

        for num, t in enumerate(solution_times):
            c = np.sin(2 * np.pi * x + t) * np.cos(2 * np.pi * y + t)

            zone_idx = libtecio.tec_zone_create_ijk(
                handle,
                f"test_aux_unsteady_{num + 1}",
                i,
                j,
                k,
                var_types=[DataType.FLOAT] * 4,
                value_locations=[ValueLocation.NODAL] * 4,
            )

            # Zone-level aux data
            libtecio.tec_zone_add_aux_data(handle, zone_idx, "MeshType", "structured")
            libtecio.tec_zone_add_aux_data(handle, zone_idx, "IJK", f"{i}x{j}x{k}")

            # Unsteady options - strand 1 links all zones into a time series
            libtecio.tec_zone_set_unsteady_options(
                handle,
                zone_idx,
                solution_time=t,
                strand=1,
            )

            libtecio.tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel(order="F"))
            libtecio.tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel(order="F"))
            libtecio.tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel(order="F"))
            libtecio.tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel(order="F"))

        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_aux_data_and_unsteady")
    except Exception as e:
        print(f"FAIL: test_tec_zone_aux_data_and_unsteady: {e}")


#---------------------------------------------------------------------------------------
# Classic API (PLT/.plt AND SZL/.szplt):
# - No handle is returned — the library maintains a single implicit global file context
# - Only one file can be active at a time; tecfil142 must be called to switch between
#   files if writing multiple files simultainously
# - tecini142 initializes the file and sets the global context; all subsequent calls
#   implicitly target this file
# - Zone records (teczne142), data (tecdat142), and node maps (tecnode142) must be
#   written strictly in order — each zone's header followed immediately by its data
#   before the next zone is declared
# - tecend142 finalizes and closes the active file
#---------------------------------------------------------------------------------------

def test_plt_tec_zone_create_ijk() -> None:
    """Test PLT ordered IJK zone using classic API."""
    try:
        i, j, k = (3, 4, 5)
        x, y, z = _create_ordered((i, j, k))
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_create_ijk.plt",
            variables=["x", "y", "z", "c"],
        )
        libtecio.teczne142(
            "test_ordered_ijk",
            ZoneType.ORDERED,
            i,
            j,
            k,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(z.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_ijk")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_create_ijk: {e}")


def test_plt_tec_zone_create_fe_lineseg() -> None:
    """Test PLT FE lineseg zone using classic API."""
    try:
        x, y, nodes = _create_FE_lineseg()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_create_fe_lineseg.plt",
            variables=["x", "y", "c"],
            file_format=FileFormat.PLT,
            file_type=FileType.FULL,
        )
        libtecio.teczne142(
            "test_fe_lineseg",
            ZoneType.FELINESEG,
            len(x),  # imx = num nodes
            len(nodes),  # jmx = num elements
            0,  # kmx = 0 for FE
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecnode142(nodes.ravel(order="F"))
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_fe_lineseg")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_create_fe_lineseg: {e}")


def test_plt_tec_zone_create_fe_tri() -> None:
    """Test PLT FE triangle zone."""
    try:
        x, y, nodes = _create_FE_tri()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_create_fe_tri.plt",
            variables=["x", "y", "c"],
        )
        libtecio.teczne142(
            "test_fe_tri",
            ZoneType.FETRIANGLE,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecnode142(nodes.ravel(order="F"))
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_fe_tri")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_create_fe_tri: {e}")


def test_plt_tec_zone_create_fe_quad() -> None:
    """Test PLT FE quadrilateral zone."""
    try:
        x, y, nodes = _create_FE_quad()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_create_fe_quad.plt",
            variables=["x", "y", "c"],
        )
        libtecio.teczne142(
            "test_fe_quad",
            ZoneType.FEQUADRILATERAL,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecnode142(nodes.ravel(order="F"))
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_fe_quad")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_create_fe_quad: {e}")


def test_plt_tec_zone_create_fe_tet() -> None:
    """Test PLT FE tetrahedron zone."""
    try:
        x, y, z, nodes = _create_FE_tet()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_create_fe_tet.plt",
            variables=["x", "y", "z", "c"],
        )
        libtecio.teczne142(
            "test_fe_tet",
            ZoneType.FETETRAHEDRON,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(z.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecnode142(nodes.ravel(order="F"))
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_fe_tet")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_create_fe_tet: {e}")


def test_plt_tec_zone_create_fe_pyramid() -> None:
    """Test PLT pyramid as degenerate FEBRICK."""
    try:
        x, y, z, nodes = _create_FE_pyramid()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_create_fe_pyramid.plt",
            variables=["x", "y", "z", "c"],
        )
        libtecio.teczne142(
            "test_fe_pyramid",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(z.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecnode142(nodes.ravel(order="F"))
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_fe_pyramid")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_create_fe_pyramid: {e}")


def test_plt_tec_zone_create_fe_prism() -> None:
    """Test PLT triangular prism as degenerate FEBRICK."""
    try:
        x, y, z, nodes = _create_FE_prism()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_create_fe_prism.plt",
            variables=["x", "y", "z", "c"],
        )
        libtecio.teczne142(
            "test_fe_prism",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(z.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecnode142(nodes.ravel(order="F"))
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_fe_prism")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_create_fe_prism: {e}")


def test_plt_tec_zone_create_fe_brick() -> None:
    """Test PLT FEBRICK zone."""
    try:
        x, y, z, faces, nodes = _create_FE_brick()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_create_fe_brick.plt",
            variables=["x", "y", "z", "c"],
        )
        libtecio.teczne142(
            "test_fe_brick",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(z.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecnode142(nodes.ravel(order="F"))
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_fe_brick")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_create_fe_brick: {e}")


def test_plt_tec_zone_face_nbr_write_connections() -> None:
    """Test PLT face neighbor connections."""
    try:
        x, y, z, nodes, face_neighbors = _create_FE_two_bricks()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        libtecio.tecini142(
            "test_plt_zone_face_nbr.plt",
            variables=["x", "y", "z", "c"],
        )
        libtecio.teczne142(
            "test_two_bricks",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
            num_face_connections=2,
            face_nbr_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(z.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecnode142(nodes.ravel(order="F"))
        libtecio.tecface142(face_neighbors.ravel(order="F"))
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_face_nbr_write_connections")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_face_nbr_write_connections: {e}")


def test_plt_tec_zone_aux_data_and_unsteady() -> None:
    """Test PLT aux data and unsteady options with multiple solution times."""
    try:
        i, j, k = (3, 4, 5)
        x, y, z = _create_ordered((i, j, k))
        solution_times = np.linspace(0.0, 2 * np.pi, 50)

        libtecio.tecini142(
            "test_plt_aux_data_unsteady.plt",
            variables=["x", "y", "z", "c"],
        )

        # Dataset-level aux data
        libtecio.tecauxstr142("Author", "test_tecio_lite")
        libtecio.tecauxstr142("CreatedBy", "libtecio")
        libtecio.tecauxstr142("Version", "1.0")

        # Variable-level aux data (must be before first teczne142)
        libtecio.tecvauxstr142(1, "Units", "meters")
        libtecio.tecvauxstr142(2, "Units", "meters")
        libtecio.tecvauxstr142(3, "Units", "meters")
        libtecio.tecvauxstr142(4, "Units", "dimensionless")
        libtecio.tecvauxstr142(4, "Description", "sin*cos field")

        for num, t in enumerate(solution_times):
            c = np.sin(2 * np.pi * x + t) * np.cos(2 * np.pi * y + t)

            libtecio.teczne142(
                f"test_aux_unsteady_{num + 1}",
                ZoneType.ORDERED,
                i,
                j,
                k,
                value_locations=[ValueLocation.NODAL] * 4,
                strand=1,
                solution_time=t,
            )
            # Zone-level aux data (must be immediately after teczne142)
            libtecio.teczauxstr142("MeshType", "structured")
            libtecio.teczauxstr142("IJK", f"{i}x{j}x{k}")

            libtecio.tecdat142(x.ravel(order="F"), is_double=False)
            libtecio.tecdat142(y.ravel(order="F"), is_double=False)
            libtecio.tecdat142(z.ravel(order="F"), is_double=False)
            libtecio.tecdat142(c.ravel(order="F"), is_double=False)

        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_aux_data_and_unsteady")
    except Exception as e:
        print(f"FAIL: test_plt_tec_zone_aux_data_and_unsteady: {e}")


#=======================================================================================
# Run all tests
#=======================================================================================
if __name__ == "__main__":
    # New API tests (SZL/.szplt)
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
    # Classic API tests (PLT/.plt)
    test_plt_tec_zone_create_ijk()
    test_plt_tec_zone_create_fe_lineseg()
    test_plt_tec_zone_create_fe_tri()
    test_plt_tec_zone_create_fe_quad()
    test_plt_tec_zone_create_fe_tet()
    test_plt_tec_zone_create_fe_pyramid()
    test_plt_tec_zone_create_fe_prism()
    test_plt_tec_zone_create_fe_brick()
    test_plt_tec_zone_face_nbr_write_connections()
    test_plt_tec_zone_aux_data_and_unsteady()
