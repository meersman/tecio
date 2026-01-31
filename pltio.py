from __future__ import annotations

import ctypes
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

import tecutils

# Load tecio library
TECIO_LIB_PATH = tecutils.get_tecio_lib()
tecio = ctypes.cdll.LoadLibrary(TECIO_LIB_PATH)


# --------------------------------------------------------------------
# ---- Tecio exception classes ---------------------------------------
# --------------------------------------------------------------------


class TecioError(RuntimeError):
    """Base exception for all szlio C/C++ library errors."""


# --------------------------------------------------------------------
# ---- Definition of Enums -------------------------------------------
# --------------------------------------------------------------------


class FileType(Enum):
    FULL = 0
    GRID = 1
    SOLUTION = 2


class ZoneType(Enum):
    ORDERED = 0
    FELINESEG = 1
    FETRIANGLE = 2
    FEQUADRILATERAL = 3
    FETETRAHEDRON = 4
    FEBRICK = 5
    FEPOLYGON = 6
    FEPOLYHEDRON = 7
    FEMIXED = 8


class DataType(Enum):
    FLOAT = 1
    DOUBLE = 2
    INT32 = 3
    INT16 = 4
    BYTE = 5


class ValueLocation(Enum):
    CELL_CENTERED = 0
    NODE_CENTERED = 1


class VarStatus(Enum):
    ACTIVE = 0
    PASSIVE = 1


class FeCellShape(Enum):
    BAR = 0
    TRIANGLE = 1
    QUADRILATERAL = 2
    TETRAHEDRON = 3
    HEXAHEDRON = 4
    PYRAMID = 5
    PRISM = 6


class FileFormat(Enum):
    PLT = 0
    SZPLT = 1


class Debug(Enum):
    FALSE = 0
    TRUE = 1


# --------------------------------------------------------------------
# ---- C library bindings: PLT Write ---------------------------------
# --------------------------------------------------------------------

# ---- Create/destroy SZL objects ------------------------------------
tecio.tecini142.restype = ctypes.c_int32
tecio.tecini142.argtypes = (
    ctypes.c_char_p,  # Title
    ctypes.c_char_p,  # Variables
    ctypes.c_char_p,  # FName
    ctypes.c_char_p,  # ScratchDir
    ctypes.POINTER(ctypes.c_int32),  # FileFormat
    ctypes.POINTER(ctypes.c_int32),  # FileType
    ctypes.POINTER(ctypes.c_int32),  # Debug
    ctypes.POINTER(ctypes.c_int32),  # IsDouble
)
tecio.tecend142.restype = ctypes.c_int32
# tecio.tecend142.argtypes=(,)
tecio.teczne142.restype = ctypes.c_int32
tecio.teczne142.argtypes = (
    ctypes.c_char_p,  # ZoneTitle
    ctypes.POINTER(ctypes.c_int32),  # ZoneType
    ctypes.POINTER(ctypes.c_int32),  # IMax
    ctypes.POINTER(ctypes.c_int32),  # JMax
    ctypes.POINTER(ctypes.c_int32),  # KMax
    ctypes.POINTER(ctypes.c_int32),  # ICellMax
    ctypes.POINTER(ctypes.c_int32),  # JCellMax
    ctypes.POINTER(ctypes.c_int32),  # KCellMax
    ctypes.POINTER(ctypes.c_double),  # SolutionTime
    ctypes.POINTER(ctypes.c_int32),  # StrandID
    ctypes.POINTER(ctypes.c_int32),  # ParentZone
    ctypes.POINTER(ctypes.c_int32),  # IsBlock
    ctypes.POINTER(ctypes.c_int32),  # NumFaceConnections
    ctypes.POINTER(ctypes.c_int32),  # FaceNeighborMode
    ctypes.POINTER(ctypes.c_int32),  # TotalNumFaceNodes
    ctypes.POINTER(ctypes.c_int32),  # NumConnectedBoundaryFaces
    ctypes.POINTER(ctypes.c_int32),  # TotalNumBoundaryConnections
    ctypes.POINTER(ctypes.c_int32),  # PassiveVarList
    ctypes.POINTER(ctypes.c_int32),  # ValueLocation
    ctypes.POINTER(ctypes.c_int32),  # ShareVarFromZone
    ctypes.POINTER(ctypes.c_int32),  # ShareConnectivityFromZone
)
tecio.TECZNEFEMIXED142.restype = ctypes.c_int32
tecio.TECZNEFEMIXED142.argtypes = (
    ctypes.c_char_p,  # ZoneTitle
    ctypes.POINTER(ctypes.c_int64),  # NumNodes
    ctypes.POINTER(ctypes.c_int32),  # NumSections
    ctypes.POINTER(ctypes.c_int32),  # CellShapePerSection
    ctypes.POINTER(ctypes.c_int32),  # GridOrderPerSection
    ctypes.POINTER(ctypes.c_int32),  # BasisFnPerSection
    ctypes.POINTER(ctypes.c_int64),  # NumElementsPerSection
    ctypes.POINTER(ctypes.c_double),  # SolutionTime
    ctypes.POINTER(ctypes.c_int32),  # StrandID
    ctypes.POINTER(ctypes.c_int32),  # NumFaceConnections
    ctypes.POINTER(ctypes.c_int32),  # FaceNeighborMode
    ctypes.POINTER(ctypes.c_int32),  # PassiveVarList
    ctypes.POINTER(ctypes.c_int32),  # ValueLocation
    ctypes.POINTER(ctypes.c_int32),  # ShareVarFromZone
    ctypes.POINTER(ctypes.c_int32),  # ShareConnectivityFromZone
)
tecio.tecflush142.restype = ctypes.c_int32
tecio.tecflush142.argtypes = (
    ctypes.POINTER(ctypes.c_int32),  # Num Zones
    ctypes.POINTER(ctypes.c_int32),  # Zones array
)

# ---- Write variable value functions --------------------------------
tecio.tecnode142.restype = ctypes.c_int32
tecio.tecnode142.argtypes = (
    ctypes.POINTER(ctypes.c_int32),  # Num Nodes
    ctypes.POINTER(ctypes.c_int32),  # Node array
)
tecio.tecpolyface142.restype = ctypes.c_int32
tecio.tecpolyface142.argtypes = (
    ctypes.POINTER(ctypes.c_int32),  # NumFaces
    ctypes.POINTER(ctypes.c_int32),  # FaceNodeCounts array
    ctypes.POINTER(ctypes.c_int32),  # FaceNodes array
    ctypes.POINTER(ctypes.c_int32),  # Face Left Elems array
    ctypes.POINTER(ctypes.c_int32),  # Face Right Elems array
)
tecio.tecdat142.restype = ctypes.c_int32
tecio.tecdat142.argtypes = (
    ctypes.POINTER(ctypes.c_int32),  # NumPts
    ctypes.c_void_p,  # values (float or double)
    ctypes.POINTER(ctypes.c_int32),  # isdouble
)

# ---- Write aux item functions --------------------------------------
tecio.tecauxstr142.restype = ctypes.c_int32
tecio.tecauxstr142.argtypes = (
    ctypes.c_char_p, # Key string
    ctypes.c_char_p, # Value string
)
tecio.tecvauxstr142.restype = ctypes.c_int32
tecio.tecvauxstr142.argtypes = (
    ctypes.POINTER(ctypes.c_int32), # VarNum
    ctypes.c_char_p, # Key string
    ctypes.c_char_p, # Value string
)
tecio.teczauxstr142.restype = ctypes.c_int32
tecio.teczauxstr142.argtypes = (
    ctypes.c_char_p,
    ctypes.c_char_p,
)


# --------------------------------------------------------------------
# ---- Wrappers for C functions: TECIO Write -------------------------
# --------------------------------------------------------------------


# ---- helper to prepare numpy arrays for ctypes -----------------------
def _prepare_array_for_ctypes(
    values: npt.ArrayLike, np_dtype, ctype
) -> tuple[ctypes.POINTER, int, npt.NDArray]:
    """
    Convert an input array-like to a contiguous numpy array and return a ctypes pointer.

    Inputs:
    - values: array-like (list, tuple, numpy array)
    - np_dtype: numpy dtype object or type (e.g. np.float32)
    - ctype: corresponding ctypes scalar type (e.g. ctypes.c_float)

    Returns:
    - (ptr, count, backing_array)
      * ptr: ctypes pointer suitable for passing to the C API
      * count: int number of elements
      * backing_array: the numpy array object (returned to keep it alive)

    Notes:
    - Caller should keep the returned backing_array alive until the native call completes.
    - This function enforces dtype and C-contiguity.
    """
    arr = np.ascontiguousarray(values, dtype=np_dtype)
    count = int(arr.size)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctype))
    return ptr, count, arr


# ---- Create TECIO objects ------------------------------------------
def tecini142(
    title: str,
    variables: str,
    file_name: str,
    scratch_dir: str,
    file_format: FileFormat,
    file_type: FileType,
    debug: Debug,
    is_double: bool,
) -> ctypes.c_void_p:
    """
    Wrapper for TECINI142. Opens output file object Can write plt and
    szplt file formats. Must be used in conjunction with TECEND142.
    """

    filetype = ctypes.c_int32(file_type)  # 0=Grid&Solution, 1=Grid, 2=Solution
    debug = ctypes.c_int32(0)  # 0=No, 1=Yes
    is_double = ctypes.c_int32(1 if is_double else 0)  # 0=float, 1=double

    ret = tecio.tecini142(
        ctypes.c_char_p(bytes(dataset_title, encoding="UTF-8")),
        ctypes.c_char_p(bytes(varnamelist, encoding="UTF-8")),
        ctypes.c_char_p(bytes(file_name, encoding="UTF-8")),
        ctypes.c_char_p(bytes(scratch_dir, encoding="UTF-8")),
        ctypes.byref(fileformat),
        ctypes.byref(filetype),
        ctypes.byref(debug),
        ctypes.byref(isdouble),
    )
    if ret != 0:
        raise Exception("open_file Error")
    return ret


def tecend142():
    """
    Wrapper for TECEND142. Closes open tecplot file.
    """
    ret = tecio.tecend142()
    if ret != 0:
        raise Exception("close_file Error")
    return ret


# ---- Write variable value functions --------------------------------
def teczne142(
    zone_name,
    zone_type,
    imax,
    jmax,
    kmax,
    solution_time=0,
    strand=0,
    parent_zone=0,
    num_face_connections=0,
    face_neighbor_mode=0,
    total_num_face_nodes=0,
    num_connected_boundary_faces=int(0),
    total_num_boundary_connections=int(0),
    var_sharing=None,
    passive_vars=None,
    value_locations=None,
) -> int:
    """
    Wrapper for TECZNE142. Main tecio function for creating zone
    records.
    """
    zone_type = ctypes.c_int32(zone_type)
    imax = ctypes.c_int32(imax)
    jmax = ctypes.c_int32(jmax)
    kmax = ctypes.c_int32(kmax)
    parent_zone = ctypes.c_int32(parent_zone)
    ignored = ctypes.c_int32(0)
    block_format = ctypes.c_int32(1)
    num_face_connections = ctypes.c_int32(num_face_connections)
    face_neighbor_mode = ctypes.c_int32(face_neighbor_mode)
    total_num_face_nodes = ctypes.c_int32(total_num_face_nodes)
    num_connected_boundary_faces = ctypes.c_int32(num_connected_boundary_faces)
    total_num_boundary_connections = ctypes.c_int32(total_num_boundary_connections)

    passive_var_list = None
    if passive_vars:
        passive_var_list = (ctypes.c_int32 * len(passive_vars))(*passive_vars)
    var_share_list = None
    if var_sharing:
        var_share_list = (ctypes.c_int32 * len(var_sharing))(*var_sharing)
    value_location_list = None
    if value_locations:
        value_location_list = (ctypes.c_int32 * len(value_locations))(*value_locations)

    ret = tecio.teczne142(
        ctypes.c_char_p(bytes(zone_name, encoding="UTF-8")),
        ctypes.byref(zone_type),
        ctypes.byref(imax),
        ctypes.byref(jmax),
        ctypes.byref(kmax),
        ctypes.byref(ignored),
        ctypes.byref(ignored),
        ctypes.byref(ignored),
        ctypes.byref(ctypes.c_double(solution_time)),
        ctypes.byref(ctypes.c_int32(strand)),
        ctypes.byref(parent_zone),
        ctypes.byref(block_format),
        ctypes.byref(num_face_connections),
        ctypes.byref(face_neighbor_mode),
        ctypes.byref(total_num_face_nodes),  # only applies to poly data
        ctypes.byref(num_connected_boundary_faces),  # only applies to poly data
        ctypes.byref(total_num_boundary_connections),  # only applies to poly data
        passive_var_list,
        value_location_list,
        var_share_list,
        ctypes.byref(ctypes.c_int32(0)),
    )  # ShareConnectivityFromZone
    return ret


def tecznefemixed(
    zone_name,
    num_nodes,
    num_sections,
    num_elements_per_section,
    cell_shape_per_section,
    grid_order_per_section=None,
    basis_function_per_section=None,
    solution_time=0,
    strand=0,
    num_face_connections=0,
    face_neighbor_mode=0,
    var_sharing=None,
    passive_vars=None,
    value_locations=None,
) -> int:
    """
    Wrapper for TECZNEFEMIXED142. Creates mixed type unstructured zone
    record.
    """
    num_nodes = ctypes.c_int64(num_nodes)
    assert num_sections <= 16
    if grid_order_per_section == None:
        grid_order_per_section = [1] * num_sections
    if basis_function_per_section == None:
        basis_function_per_section = [0] * num_sections
    num_sections = ctypes.c_int32(num_sections)

    cell_shape_per_section = (ctypes.c_int32 * len(cell_shape_per_section))(
        *cell_shape_per_section
    )
    num_elements_per_section = (ctypes.c_int64 * len(num_elements_per_section))(
        *num_elements_per_section
    )

    grid_order_per_section = (ctypes.c_int32 * len(grid_order_per_section))(
        *grid_order_per_section
    )

    basis_function_per_section = (ctypes.c_int32 * len(basis_function_per_section))(
        *basis_function_per_section
    )

    ignored = ctypes.c_int32(0)
    num_face_connections = ctypes.c_int32(num_face_connections)
    face_neighbor_mode = ctypes.c_int32(face_neighbor_mode)

    passive_var_list = None
    if passive_vars:
        passive_var_list = (ctypes.c_int32 * len(passive_vars))(*passive_vars)
    var_share_list = None
    if var_sharing:
        var_share_list = (ctypes.c_int32 * len(var_sharing))(*var_sharing)
    value_location_list = None
    if value_locations:
        value_location_list = (ctypes.c_int32 * len(value_locations))(*value_locations)

    ret = tecio.TECZNEFEMIXED142(
        ctypes.c_char_p(bytes(zone_name, encoding="UTF-8")),
        ctypes.byref(num_nodes),
        ctypes.byref(num_sections),
        cell_shape_per_section,
        grid_order_per_section,
        basis_function_per_section,
        num_elements_per_section,
        ctypes.byref(ctypes.c_double(solution_time)),
        ctypes.byref(ctypes.c_int32(strand)),
        ctypes.byref(num_face_connections),
        ctypes.byref(face_neighbor_mode),
        passive_var_list,
        value_location_list,
        var_share_list,
        ctypes.byref(ctypes.c_int32(0)),
    )  # ShareConnectivityFromZone

    if ret != 0:
        raise Exception("create_fe_mixed_zone Error")
    return ret


def tecnode142(nodes) -> None:

    nodes = np.asarray(nodes, dtype=np.int32).flatten()
    num_nodes = len(nodes)
    ret = tecio.tecnode142(
        ctypes.byref(ctypes.c_int32(num_nodes)),
        ctypes.cast(nodes.ctypes.data, ctypes.POINTER(ctypes.c_int32)),
    )
    if ret != 0:
        raise Exception("tecnode Error")


def tecpolyface142(
    num_faces, face_node_counts, face_nodes, face_left_elems, face_right_elems
) -> None:

    face_node_count_array = None
    if face_node_counts:
        face_node_counts = np.asarray(face_node_counts, dtype=np.int32)
        face_node_count_array = ctypes.cast(
            face_node_counts.ctypes.data, ctypes.POINTER(ctypes.c_int32)
        )

    face_nodes = np.asarray(face_nodes, dtype=np.int32)
    face_left_elems = np.asarray(face_left_elems, dtype=np.int32)
    face_right_elems = np.asarray(face_right_elems, dtype=np.int32)
    ret = tecio.tecpolyface142(
        ctypes.byref(ctypes.c_int32(num_faces)),
        face_node_count_array,  # ctypes.cast(face_node_counts.ctypes.data, ctypes.POINTER(ctypes.c_int32)),
        ctypes.cast(face_nodes.ctypes.data, ctypes.POINTER(ctypes.c_int32)),
        ctypes.cast(face_left_elems.ctypes.data, ctypes.POINTER(ctypes.c_int32)),
        ctypes.cast(face_right_elems.ctypes.data, ctypes.POINTER(ctypes.c_int32)),
    )
    if ret != 0:
        raise Exception("tecpolyface Error")


def tecdat142(values: ntp.NDArray[np.floating]) -> None:
    """
    Wrapper for TECDAT142. Accepts either single or double precision
    floats formatted as an NDArray.
    """
    if not values.flags["C_CONTIGUOUS"]:
        values = np.ascontiguousarray(values)

    if values.dtype == np.float64:
        ret = tecio.tecdat142(
            ctypes.byref(ctypes.c_int32(values.size)),
            ctypes.cast(
                values.ctypes.data,
                ctypes.POINTER(ctypes.c_double),
            ),
            ctypes.byref(ctypes.c_int32(1)),
        )
    elif values.dtype == np.float32:
        ret = tecio.tecdat142(
            ctypes.byref(ctypes.c_int32(values.size)),
            ctypes.cast(
                values.ctypes.data,
                ctypes.POINTER(ctypes.c_float),
            ),
            ctypes.byref(ctypes.c_int32(0)),
        )
    else:
        raise TecioError(
            f"Unsupported dtype {values.dtype}; expected float32 or float64"
        )

    if ret != 0:
        raise Exception("tecdat142 Error")


# ---- Write aux data items functions --------------------------------
def tecauxstr142(key: str, value: str) -> None:
    """
    Adds Dataset Auxiliary Data
    """
    ret = tecio.tecauxstr142(
        ctypes.c_char_p(bytes(key, encoding="UTF-8")),
        ctypes.c_char_p(bytes(value, encoding="UTF-8")),
    )
    if ret != 0:
        raise Exception("tecauxstr142 Error")


def tecvarauxstr142(varnum: int, key: str, value: str) -> None:
    """
    Adds Variable Auxiliary Data
    """
    varnum = ctypes.c_int32(varnum)
    ret = tecio.tecvauxstr142(
        ctypes.byref(varnum),
        ctypes.c_char_p(bytes(key, encoding="UTF-8")),
        ctypes.c_char_p(bytes(value, encoding="UTF-8")),
    )
    if ret != 0:
        raise Exception("tecvarauxstr142 Error")


def teczauxstr142(key: str, value: str) -> None:
    """
    Adds Zone Auxiliary Data to the zone that is currently being written to.
    Must be called immediately after adding a new zone via: teczne or other add_xxx_zone functions.
    """
    ret = tecio.teczauxstr142(
        ctypes.c_char_p(bytes(key, encoding="UTF-8")),
        ctypes.c_char_p(bytes(value, encoding="UTF-8")),
    )
    if ret != 0:
        raise Exception("teczauxstr142 Error")
