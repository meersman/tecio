#!/usr/bin/env python3
"""Lite test for libtecio functions directly."""

# ruff: noqa: F403, F405

import numpy as np
from create_test_data import *

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
        x, y, z = create_ordered((i, j, k))
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_ijk.szplt",
            variables=["x", "y", "z", "c"],
        )
        izone = libtecio.tec_zone_create_ijk(
            handle,
            "test_ordered_ijk",
            i,
            j,
            k,
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 4, c.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_ijk")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_ijk: {e}")


def test_tec_zone_create_fe_lineseg() -> None:
    """Test FELINESEG zone type."""
    try:
        # Create sample FELINESEG data
        x, y, nodes = create_FE_lineseg()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_lineseg.szplt",
            variables=["x", "y","c"],
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_lineseg",
            ZoneType.FELINESEG,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, izone, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_lineseg")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_lineseg: {e}")


def test_tec_zone_create_fe_tri() -> None:
    """Test FETRIANGLE zone type."""
    try:
        # Create sample FETRIANGLE data
        x, y, nodes = create_FE_tri()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_tri.szplt",
            variables=["x", "y", "c"],
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_triangle",
            ZoneType.FETRIANGLE,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, izone, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_tri")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_tri: {e}")


def test_tec_zone_create_fe_quad() -> None:
    """Test FEQUADRILATERAL zone type."""
    try:
        # Create sample FEQUADRILATERAL data
        x, y, nodes = create_FE_quad()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_quad.szplt",
            variables=["x", "y", "c"],
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_quad",
            ZoneType.FEQUADRILATERAL,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 3,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, izone, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_quad")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_quad: {e}")


def test_tec_zone_create_fe_tet() -> None:
    """Test FETETRAHEDRON zone type."""
    try:
        x, y, z, nodes = create_FE_tet()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_tet.szplt",
            variables=["x", "y", "z", "c"],
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_tet",
            ZoneType.FETETRAHEDRON,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, izone, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_tet")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_tet: {e}")


def test_tec_zone_create_fe_pyramid() -> None:
    """Test pyramid as degenerate FEBRICK zone type."""
    try:
        x, y, z, nodes = create_FE_pyramid()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_pyramid.szplt",
            variables=["x", "y", "z", "c"],
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_pyramid",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, izone, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_pyramid")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_pyramid: {e}")


def test_tec_zone_create_fe_prism() -> None:
    """Test triangular prism as degenerate FEBRICK zone type."""
    try:
        x, y, z, nodes = create_FE_prism()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_prism.szplt",
            variables=["x", "y", "z", "c"],
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_prism",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, izone, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_prism")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_prism: {e}")


def test_tec_zone_create_fe_brick() -> None:
    """Test FEBRICK zone type."""
    try:
        # Create sample FEBRICK data
        x, y, z, faces, nodes = create_FE_brick()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_create_fe_brick.szplt",
            variables=["x", "y", "z", "c"],
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_brick",
            ZoneType.FEBRICK,
            len(x),  # number of nodes
            len(nodes),
            var_types=[DataType.FLOAT] * 4,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, izone, nodes.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_create_fe_brick")
    except Exception as e:
        print(f"FAIL: test_tec_zone_create_fe_brick: {e}")


def test_tec_zone_face_nbr_write_connections() -> None:
    """Test face neighbor connections with two adjacent bricks."""
    try:
        x, y, z, nodes, face_neighbors = create_FE_two_bricks()
        c = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        handle = libtecio.tec_file_writer_open(
            "test_tec_zone_face_nbr.szplt",
            variables=["x", "y", "z", "c"],
        )
        izone = libtecio.tec_zone_create_fe(
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
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 4, c.ravel(order="F"))
        libtecio.tec_zone_node_map_write32(handle, izone, nodes.ravel(order="F"))
        libtecio.tec_zone_face_nbr_write_connections32(
            handle, izone, face_neighbors.ravel(order="F")
        )
        libtecio.tec_file_writer_close(handle)
        print("PASS: test_tec_zone_face_nbr_write_connections")
    except Exception as e:
        print(f"FAIL: test_tec_zone_face_nbr_write_connections: {e}")


def test_tec_zone_aux_data_and_unsteady() -> None:
    """Test dataset, variable, and zone auxiliary data plus unsteady options."""
    try:
        i, j, k = (3, 4, 5)
        x, y, z = create_ordered((i, j, k))
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

            izone = libtecio.tec_zone_create_ijk(
                handle,
                f"test_aux_unsteady_{num + 1}",
                i,
                j,
                k,
                var_types=[DataType.FLOAT] * 4,
                value_locations=[ValueLocation.NODAL] * 4,
            )

            # Zone-level aux data
            libtecio.tec_zone_add_aux_data(handle, izone, "MeshType", "structured")
            libtecio.tec_zone_add_aux_data(handle, izone, "IJK", f"{i}x{j}x{k}")

            # Unsteady options - strand 1 links all zones into a time series
            libtecio.tec_zone_set_unsteady_options(
                handle,
                izone,
                solution_time=t,
                strand=1,
            )

            libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
            libtecio.tec_zone_var_write_float_values(handle, izone, 2, y.ravel(order="F"))
            libtecio.tec_zone_var_write_float_values(handle, izone, 3, z.ravel(order="F"))
            libtecio.tec_zone_var_write_float_values(handle, izone, 4, c.ravel(order="F"))

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
        x, y, z = create_ordered((i, j, k))
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
        x, y, nodes = create_FE_lineseg()
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
        x, y, nodes = create_FE_tri()
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
        x, y, nodes = create_FE_quad()
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
        x, y, z, nodes = create_FE_tet()
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
        x, y, z, nodes = create_FE_pyramid()
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
        x, y, z, nodes = create_FE_prism()
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
        x, y, z, faces, nodes = create_FE_brick()
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
        x, y, z, nodes, face_neighbors = create_FE_two_bricks()
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
        x, y, z = create_ordered((i, j, k))
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


def test_plt_tec_zone_create_fe_polygon() -> None:
    """Test PLT FEPOLYGON zone using classic API."""
    try:
        x, y, faces = create_FE_polygon()  # creates one polygon
        c = np.array([8])

        num_nodes = len(x)
        num_elements = 1  # single polygon element
        num_faces = len(faces)
        total_face_nodes = len(faces.ravel())
        face_node_counts = np.full(num_faces, 2)
        # Left/right elements: since single cell, face nodes are ordered such that follow
        # right hand rule (current element is on right, no element on left)
        left_elems = np.full(num_faces, 0)
        right_elems = np.full(num_faces, 1)

        libtecio.tecini142(
            "test_plt_fe_polygon.plt",
            variables=["x", "y", "c"],
        )
        libtecio.tecpolyzne142(
            zone_title="test_fe_polygon",
            zone_type=ZoneType.FEPOLYGON,
            num_nodes=num_nodes,
            num_faces=num_faces,
            num_elements=num_elements,
            total_num_face_nodes=total_face_nodes,
            value_locations=[
                ValueLocation.NODAL,
                ValueLocation.NODAL,
                ValueLocation.CELL_CENTERED,
            ],
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=False)
        libtecio.tecpolyface142(
            face_node_counts,
            faces.ravel(order="C"),
            left_elems,
            right_elems,
        )
        libtecio.tecend142()
        print("PASS: test_plt_tec_zone_create_fe_polygon")
    except Exception as exc:
        print(f"FAIL: test_plt_tec_zone_create_fe_polygon: {exc}")


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
    test_plt_tec_zone_create_fe_polygon()
