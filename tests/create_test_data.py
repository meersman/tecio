"""Library of functions that create various data types for tecio tests."""

# ruff: noqa: F403, F405

import numpy as np
import numpy.typing as npt

# ======================================================================================
# Functions to create all supported Tecplot data formats
# ======================================================================================


def scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Return a simple sin-cos scalar field over the supplied coordinate arrays."""
    if z is not None:
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1.0 + 0.1 * z)
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def create_ordered(
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


def create_FE_lineseg() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FELINESEG zone type.

    Node Map:
    [0]--<C1>--[1]--<C2>--[2]
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
        [0, 1],  # line seg AB
        [1, 2],  # line seg BC
    ])
    return x, y, nodes


def create_FE_tri() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    r"""Create coordinates and nodemap for FELTRIANGLE zone type.

    Node Map: Two triangles sharing edge 1-2:
    2 --- 3  Cell 1: 0-1-2
    |\ C2 |  Cell 2: 1-3-2
    | \   |
    |C1 \ |
    0 --- 1
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
        [0, 1, 2],
        [1, 3, 2],
    ])
    return x, y, nodes


def create_FE_quad() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FEQUADRILATERAL zone type.

    Node Map: Two quads sharing edge 1-4:
    3 --- 4 --- 5  Cell 1: 0-1-4-3
    | C1  | C2  |  Cell 2: 1-2-5-4
    0 --- 1 --- 2
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
        [0, 1, 4, 3],
        [1, 2, 5, 4],
    ])
    return x, y, nodes


def create_FE_polygon() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    r"""Create coordinates and node map for FEPOLYGON zone type.

    Node map: single octogon cell
             F5
           6 -- 5
      F6 /        \ F4
       7            4
    F7 |     E1     | F3
       8            3
      F8 \        / F2
           1 -- 2
             F1
    """
    points = np.array([
        [0.25, 0],  # node 1
        [0.75, 0],  # node 2
        [1, 0.25],  # node 3
        [1, 0.75],  # node 4
        [0.75, 1],  # node 5
        [0.25, 1],  # node 6
        [0, 0.75],  # node 7
        [0, 0.25],  # node 8
    ])
    x = points[:, 0]
    y = points[:, 1]
    # Nodemap
    faces = np.array([
        [2, 1],  # Face 1
        [3, 2],  # Face 2
        [4, 3],  # Face 3
        [5, 4],  # Face 4
        [6, 5],  # Face 5
        [7, 6],  # Face 6
        [8, 7],  # Face 7
        [1, 8],  # Face 8
    ])
    return x, y, faces


def create_FE_tet() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    r"""Create coordinates and nodemap for FETETRAHEDRON zone type.

    Node Map: Two tetrahedra sharing base face 0-1-2, apices above (3)
    and below (4):

        3        <- apex tet 1 (z > 0)
       /|\
      / | \
     /  |  \
    0---+---1   <- shared base triangle
     \  |  /
      \ | /
       \|/
        4        <- apex tet 2 (z < 0)

    Cell 1: [0, 1, 2, 3]
    Cell 2: [0, 2, 1, 4]
    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.5, 1.0, 0.0],  # 2
        [0.5, 0.5, 1.0],  # 3 - apex of tet 1
        [0.5, 0.5, -1.0],  # 4 - apex of tet 2
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Each row: base triangle (CCW from outside) + apex
    nodes = np.array([
        [0, 1, 2, 3],  # tet 1 - apex above
        [0, 2, 1, 4],  # tet 2 - apex below
    ])
    return x, y, z, nodes


def create_FE_pyramid() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for a pyramid using FEBRICK zone type.

    Node: Tecplot represents pyramids as degenerate bricks where nodes 4,5,6,7 are all
    the apex node repeated.

    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 0 - base
        [1.0, 0.0, 0.0],  # 1 - base
        [1.0, 1.0, 0.0],  # 2 - base
        [0.0, 1.0, 0.0],  # 3 - base
        [0.5, 0.5, 1.0],  # 4 - apex
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Repeat apex node (4) for the top 4 nodes of the brick
    nodes = np.array([
        [0, 1, 2, 3, 4, 4, 4, 4],
    ])
    return x, y, z, nodes


def create_FE_prism() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for a triangular prism using FEBRICK.

    Note: Represented as degenerate brick: bottom tri nodes 0,1,2 paired with
    top tri nodes 3,4,5 — each tri edge node repeated to fill 8-node brick.

    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 0 - bottom tri
        [1.0, 0.0, 0.0],  # 1 - bottom tri
        [0.5, 1.0, 0.0],  # 2 - bottom tri
        [0.0, 0.0, 1.0],  # 3 - top tri
        [1.0, 0.0, 1.0],  # 4 - top tri
        [0.5, 1.0, 1.0],  # 5 - top tri
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Bottom tri: 0,1,2,2 / Top tri: 3,4,5,5
    nodes = np.array([
        [0, 1, 2, 2, 3, 4, 5, 5],
    ])
    return x, y, z, nodes


def create_FE_brick() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for FEBRICK zone type."""
    points = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Nodemap
    faces = np.array([
        [0, 1, 2, 3],  # Face 1
        [0, 3, 7, 4],  # Face 2
        [4, 7, 6, 5],  # Face 3
        [1, 5, 6, 2],  # Face 4
        [5, 1, 0, 4],  # Face 5
        [2, 6, 7, 3],  # Face 6
    ])
    nodes = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    return x, y, z, faces, nodes


def create_FE_two_bricks() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Two bricks sharing a face, with explicit face neighbor connectivity.

    Node Map:
      7 --- 6 --- 11
     /|    /|    /|
    4 --- 5 --- 10|
    | 3 --|-2 --|-8
    |/    |/    |/
    0 --- 1 --- 9

    Notes:
    - For the first FEBRICK cell shown above the faces are numbered
    f0: n0-n4-n7-n3 left face (Imin)
    f1: n1-n2-n6-n5 right face (Imax)
    f2: n0-n1-n5-n5 front face (Jmin)
    f3: n2-n3-n7-n6 back face (Jmax)
    f4: n0-n1-n2-n3 bottom face (Kmin)
    f5: n4-n5-n6-n7 top face (Kmax)

    """
    points = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
        [2, 0, 0],  # 8
        [2, 1, 0],  # 9
        [2, 0, 1],  # 10
        [2, 1, 1],  # 11
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    nodes = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],  # cell 0
        [1, 8, 9, 2, 5, 10, 11, 6],  # cell 1
    ])
    # 6 faces per cell: left, right, front, back, bottom, top. cz1/cz2 and fz
    # (the face ordinal) are all.
    face_neighbors = np.array([
        [0, 1, 1],  # cell 0: right face (f1) neighbors cell 1 (cz1, fz, cz2)
        [1, 0, 0],  # cell 1: left face (f0) neighbors cell 0
    ])
    return x, y, z, nodes, face_neighbors


def create_FE_mixed() -> tuple[
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
            [0.0, 0.0, 0.0],  #  0
            [1.0, 0.0, 0.0],  #  1
            [1.0, 1.0, 0.0],  #  2
            [0.0, 1.0, 0.0],  #  3
            [2.0, 0.0, 0.0],  #  4
            [2.0, 1.0, 0.0],  #  5
            [3.0, 0.0, 0.0],  #  6
            [3.0, 1.0, 0.0],  #  7
            [4.0, 0.0, 0.0],  #  8
            [4.0, 1.0, 0.0],  #  9
            # Mid layer (z=1)
            [2.0, 0.0, 1.0],  # 10
            [2.0, 1.0, 1.0],  # 11
            [3.0, 0.0, 1.0],  # 12
            [3.0, 1.0, 1.0],  # 13
            [4.0, 0.0, 1.0],  # 14
            [4.0, 1.0, 1.0],  # 15
            # Apex nodes
            [0.5, 0.5, 1.0],  # 16 - tet apex
            [1.5, 0.5, 1.0],  # 17 - pyramid apex
        ],
        dtype=np.float32,
    )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Cell 0: Tet - base triangle 0,1,2 + apex 16
    # Cell 1: Pyramid - base quad 1,4,5,2 + apex 17
    # Cell 2: Prism - bottom tri 4,6,5 + top tri 10,12,11
    # Cell 3: Hex - bottom quad 6,8,9,7 + top quad 12,14,15,13
    # fmt: off
    node_map = np.array(
        [
            0, 1, 2, 16,  # tet: 4 nodes
            1, 4, 5, 2, 17,  # pyramid: 5 nodes
            4, 6, 5, 10, 12, 11,  # prism: 6 nodes
            6, 8, 9, 7, 12, 14, 15, 13,  # hex: 8 nodes
        ],
    )
    # fmt: on

    # One entry per cell: number of nodes in that cell
    num_nodes_per_element = np.array([4, 5, 6, 8])

    return x, y, z, node_map, num_nodes_per_element


def create_FE_hanging_node_poly() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    r"""Create coordinates and face data for an FEPOLYGON zone.

    Geometry: two adjacent convex polygons sharing one edge.

    - Element 1: pentagon  — nodes 1, 2, 3, 4, 5
    - Element 2: triangle  — nodes 2, 6, 3

    Layout (approximate)::

        4 ---- 3 --- 6
        |  E1  | E  /
        5      | 2 /
         \     |  /
          1--- 3 /

    For FEPOLYGON, every face is an *edge* (2 nodes). Faces are defined
    once and shared between adjacent elements via left/right element indices.
    A value of 0 in left/right means the face is on the domain boundary.

    Face table (1-based node indices, 1-based element indices)::

        Face  Nodes   Left  Right
        ----  ------  ----  -----
         1    1–2       1     0    boundary of E1
         2    2–3       1     2    shared edge
         3    3–4       1     0    boundary of E1
         4    4–5       1     0    boundary of E1
         5    5–1       1     0    boundary of E1
         6    2–6       2     0    boundary of E2
         7    6–3       2     0    boundary of E2

    Returns:
        x, y:             Node coordinates (float32, shape ``(6,)``).
        face_node_counts: Nodes per face (int32, shape ``(7,)`` — all 2).
        face_nodes:       Concatenated node lists (int32, shape ``(14,)``).
        face_left_elems:  Left element per face (int32, shape ``(7,)``).
        face_right_elems: Right element per face (int32, shape ``(7,)``).

    """
    points = np.array(
        [
            [0.0, 0.0],  # 1
            [1.0, 0.0],  # 2
            [1.0, 1.0],  # 3
            [0.5, 1.5],  # 4
            [-0.5, 1.0],  # 5
            [2.0, 1.0],  # 6
        ],
        dtype=np.float32,
    )
    x = points[:, 0]
    y = points[:, 1]

    # Each row: [node_a, node_b, left_elem, right_elem]
    faces = np.array([
        [1, 2],  # face 1 — E1 boundary
        [2, 3],  # face 2 — shared
        [3, 4],  # face 3 — E1 boundary
        [4, 5],  # face 4 — E1 boundary
        [5, 1],  # face 5 — E1 boundary
        [2, 6],  # face 6 — E2 boundary
        [6, 3],  # face 7 — E2 boundary
    ])
    # All polygon faces are edges → 2 nodes each
    num_faces = len(faces)
    face_node_counts = np.full(num_faces, 2)
    face_nodes = faces.ravel()
    face_left_elems = faces[:, 0]
    face_right_elems = faces[:, 1]

    return x, y, face_nodes, face_node_counts, face_left_elems, face_right_elems
