#!/usr/bin/env python3
"""
Simple script to test the libtecio functions directly.
"""


import numpy as np
import numpy.typing as npt

from tecio.libtecio import *


def _create_ordered(ijk: tuple[int, int, int]) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
]:
    """Create ordered coordinates"""
    x_ = np.linspace(0., ijk[0], ijk[0])
    y_ = np.linspace(0., ijk[1], ijk[1])
    z_ = np.linspace(0., ijk[2], ijk[2])
    x, y = np.meshgrid(x_, y_, indexing='xy')
    x = np.array([x]*ijk[2])
    y = np.array([y]*ijk[2])
    z = np.repeat(z_, ijk[0]*ijk[1])
    return x, y, z

def _create_FE_lineseg() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FELINESEG zone type"""
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
    """Create coordinates and nodemap for FELTRIANGLE zone type"""
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
    """Create coordinates and nodemap for FEQUADRILATERAL zone type"""
    # Can create polygon from the same two FE tri cells above
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


def _create_FE_polygon() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FEPOLYGON zone type"""
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


def _create_FE_brick() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for FEPOLYGON zone type"""
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
    return x, y, z, faces


def test_tec_zone_create_ijk() -> None:
    """Setup and test the functionality of tec_zone_create_ijk using minimal input"""
    # Create samople data
    i, j, k = (3, 4, 5)
    x, y, z = _create_ordered((i, j, k))
    c = np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

    # Create szl file handle
    handle = tec_file_writer_open(
        "test_tec_zone_create_ijk.szplt",
        "test_tecio",
        "x, y, z, c",
        FileType.FULL,
    )

    # Create ijk ordered zone
    zone_idx = tec_zone_create_ijk(
        handle,
        "test_ordered_3d",
         i,
         j,
         k,
         value_locations = [ValueLocation.NODAL, ValueLocation.NODAL, ValueLocation.NODAL, ValueLocation.NODAL],
    )

    print(zone_idx)

    # Write variable data
    tec_zone_var_write_float_values(handle, zone_idx, 1, x.flatten())
    tec_zone_var_write_float_values(handle, zone_idx, 2, y.flatten())
    tec_zone_var_write_float_values(handle, zone_idx, 3, z.flatten())
    tec_zone_var_write_float_values(handle, zone_idx, 4, c.flatten())
    # tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
    # tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
    # tec_zone_var_write_float_values(handle, zone_idx, 3, z.ravel())
    # tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel())

    # Close file
    tec_file_writer_close(handle)


def test_tec_zone_create_fe() -> None:
    """Setup and test the functionality of tec_zone_create_ijk using minimal input"""
    # Open szl file
    handle = tec_file_writer_open(
        "test_tec_zone_create_fe.szplt",
        "test_tecio",
        "x, y, z, c",
        FileType.FULL,
    )
    x, y, nodes = _create_FE_lineseg()
    c = np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    # zone_idx = tec_zone_create_fe(
    #     handle,
    #     "test_fe_lineseg",
    #      len(x), # number of nodes
    #      ,
    #      k,
    #      value_locations = [ValueLocation.NODAL, ValueLocation.NODAL, ValueLocation.NODAL, ValueLocation.NODAL],
    # )
    # tec_zone_var_write_float_values(handle, zone_idx, 1, x.ravel())
    # tec_zone_var_write_float_values(handle, zone_idx, 2, y.ravel())
    # tec_zone_var_write_float_values(handle, zone_idx, 4, c.ravel())
    # tec_file_writer_close(handle)


if __name__ == "__main__":
    test_tec_zone_create_ijk()
