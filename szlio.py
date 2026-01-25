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
    NODAL = 1


# --------------------------------------------------------------------
# ---- C library bindings: SZL Read ----------------------------------
# --------------------------------------------------------------------

# ---- Reading SZL files ---------------------------------------------
tecio.tecFileReaderOpen.restype = ctypes.c_int32
tecio.tecFileReaderOpen.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
]
tecio.tecFileGetType.restype = ctypes.c_int32
tecio.tecFileGetType.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecDataSetGetTitle.restype = ctypes.c_int32
tecio.tecDataSetGetTitle.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
]
tecio.tecDataSetGetNumVars.restype = ctypes.c_int32
tecio.tecDataSetGetNumVars.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecDataSetGetNumZones.restype = ctypes.c_int32
tecio.tecDataSetGetNumZones.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecDataSetAuxDataGetNumItems.restype = ctypes.c_int32
tecio.tecDataSetAuxDataGetNumItems.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]

# ---- Reading SZL zones ---------------------------------------------
tecio.tecZoneGetIJK.restype = ctypes.c_int32
tecio.tecZoneGetIJK.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
]
tecio.tecZoneGetTitle.restype = ctypes.c_int32
tecio.tecZoneGetTitle.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
]
tecio.tecZoneGetType.restype = ctypes.c_int32
tecio.tecZoneGetType.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneIsEnabled.restype = ctypes.c_int32
tecio.tecZoneIsEnabled.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneGetSolutionTime.restype = ctypes.c_int32
tecio.tecZoneGetSolutionTime.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_double),
]
tecio.tecZoneGetStrandID.restype = ctypes.c_int32
tecio.tecZoneGetStrandID.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneNodeMapIs64Bit.restype = ctypes.c_int32
tecio.tecZoneNodeMapIs64Bit.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneNodeMapGet64.restype = ctypes.c_int32
tecio.tecZoneNodeMapGet64.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
]
tecio.tecZoneNodeMapGet.restype = ctypes.c_int32
tecio.tecZoneNodeMapGet.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int32),
]

# ---- Reading SZL variable data -------------------------------------
tecio.tecVarGetName.restype = ctypes.c_int32
tecio.tecVarGetName.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
]
tecio.tecVarIsEnabled.restype = ctypes.c_int32
tecio.tecVarIsEnabled.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneVarGetType.restype = ctypes.c_int32
tecio.tecZoneVarGetType.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneVarGetValueLocation.restype = ctypes.c_int32
tecio.tecZoneVarGetValueLocation.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneVarIsPassive.restype = ctypes.c_int32
tecio.tecZoneVarIsPassive.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneVarGetSharedZone.restype = ctypes.c_int32
tecio.tecZoneVarGetSharedZone.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneVarGetNumValues.restype = ctypes.c_int32
tecio.tecZoneVarGetNumValues.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneVarGetFloatValues.restype = ctypes.c_int32
tecio.tecZoneVarGetFloatValues.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_float),  # Values
]
tecio.tecZoneVarGetDoubleValues.restype = ctypes.c_int32
tecio.tecZoneVarGetDoubleValues.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_double),  # Values
]
tecio.tecZoneVarGetInt32Values.restype = ctypes.c_int32
tecio.tecZoneVarGetInt32Values.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_int32),  # Values
]
tecio.tecZoneVarGetInt16Values.restype = ctypes.c_int32
tecio.tecZoneVarGetInt16Values.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_int16),  # Values
]
tecio.tecZoneVarGetUInt8Values.restype = ctypes.c_int32
tecio.tecZoneVarGetUInt8Values.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_uint8),  # Values
]

# ---- Reading SZL aux data ------------------------------------------
tecio.tecDataSetAuxDataGetNumItems.restype = ctypes.c_int32
tecio.tecDataSetAuxDataGetNumItems.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecDataSetAuxDataGetItem.restype = ctypes.c_int32
tecio.tecDataSetAuxDataGetItem.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]
tecio.tecVarAuxDataGetNumItems.restype = ctypes.c_int32
tecio.tecVarAuxDataGetNumItems.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecVarAuxDataGetItem.restype = ctypes.c_int32
tecio.tecVarAuxDataGetItem.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]
tecio.tecZoneAuxDataGetNumItems.restype = ctypes.c_int32
tecio.tecZoneAuxDataGetNumItems.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneAuxDataGetItem.restype = ctypes.c_int32
tecio.tecZoneAuxDataGetItem.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]

# --------------------------------------------------------------------
# ---- C library bindings: SZL Write ---------------------------------
# --------------------------------------------------------------------

# ---- Create SZL objects --------------------------------------------
tecio.tecFileWriterOpen.restype = ctypes.c_int32
tecio.tecFileWriterOpen.argtypes = [
    ctypes.c_char_p,  # fileName
    ctypes.c_char_p,  # dataSetTitle
    ctypes.c_char_p,  # varNames (comma separated)
    ctypes.c_int32,  # useSZL (1)
    ctypes.c_int32,  # fileType
    ctypes.c_int32,  # reserved / options
    ctypes.c_void_p,  # gridFileHandle (optional)
    ctypes.POINTER(ctypes.c_void_p),  # out fileHandle
]
tecio.tecFileWriterClose.restype = ctypes.c_int32
tecio.tecFileWriterClose.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
]
tecio.tecZoneCreateIJK.restype = ctypes.c_int32
tecio.tecZoneCreateIJK.argtypes = [
    ctypes.c_void_p,  # file_handle
    ctypes.c_char_p,  # zoneTitle
    ctypes.c_int64,  # I
    ctypes.c_int64,  # J
    ctypes.c_int64,  # K
    ctypes.POINTER(ctypes.c_int32),  # varTypes
    ctypes.POINTER(ctypes.c_int32),  # shareVarFromZone
    ctypes.POINTER(ctypes.c_int32),  # valueLocations
    ctypes.POINTER(ctypes.c_int32),  # passiveVarList
    ctypes.c_int32,  # shareFaceNeighborsFromZone
    ctypes.c_int64,  # numFaceConnections
    ctypes.c_int32,  # faceNeighborMode
    ctypes.POINTER(ctypes.c_int32),  # out zone
]
tecio.tecZoneSetUnsteadyOptions.restype = ctypes.c_int32
tecio.tecZoneSetUnsteadyOptions.argtypes = [
    ctypes.c_void_p,  # file_handle
    ctypes.c_int32,  # zone
    ctypes.c_double,  # solutionTime
    ctypes.c_int32,  # strand
]

# ---- Write variable value functions --------------------------------
tecio.tecZoneVarWriteDoubleValues.restype = ctypes.c_int32
tecio.tecZoneVarWriteDoubleValues.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,  # partition
    ctypes.c_int64,  # count
    ctypes.POINTER(ctypes.c_double),
]
tecio.tecZoneVarWriteFloatValues.restype = ctypes.c_int32
tecio.tecZoneVarWriteFloatValues.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_float),
]
tecio.tecZoneVarWriteInt32Values.restype = ctypes.c_int32
tecio.tecZoneVarWriteInt32Values.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int32),
]
tecio.tecZoneVarWriteInt16Values.restype = ctypes.c_int32
tecio.tecZoneVarWriteInt16Values.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int16),
]
tecio.tecZoneVarWriteUInt8Values.restype = ctypes.c_int32
tecio.tecZoneVarWriteUInt8Values.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint8),
]


# --------------------------------------------------------------------
# ---- Wrappers for C functions: SZL Read ----------------------------
# --------------------------------------------------------------------


# ---- Reading SZL files ---------------------------------------------
def tec_file_reader_open(file_name: str) -> ctypes.c_void_p:
    """
    Open an SZL reader file.

    Inputs:
    - file_name: path to the .szplt file (string).

    Returns:
    - ctypes.c_void_p: a handle used by other tecio functions.

    Raises:
    - TecioError if the underlying tecio.tecFileReaderOpen returns non-zero.
    """
    handle = ctypes.c_void_p(0)

    ret = tecio.tecFileReaderOpen(
        ctypes.c_char_p(bytes(file_name, encoding="UTF-8")),
        ctypes.byref(handle),
    )
    if ret != 0:
        raise TecioError(
            f"SzlFile Initialization Error: file_name={file_name}, return_code={ret}"
        )

    return handle


def tec_file_get_type(handle: ctypes.c_void_p) -> FileType:
    """
    Get the FileType for an opened SZL file.

    Inputs:
    - handle: ctypes.c_void_p returned by tec_file_reader_open.

    Returns:
    - FileType enum indicating FULL, GRID or SOLUTION.

    Raises:
    - TecioError on non-zero return code from tecio.
    """
    file_type = ctypes.c_int32(0)

    ret = tecio.tecFileGetType(handle, ctypes.byref(file_type))
    if ret != 0:
        raise TecioError(f"Error getting file type: handle:{handle}, return_code={ret}")

    return FileType(file_type.value)


def tec_data_set_get_title(handle: ctypes.c_void_p) -> str:
    """
    Read the dataset title string.

    Inputs:
    - handle: ctypes.c_void_p file handle.

    Returns:
    - str: UTF-8 decoded dataset title.

    Raises:
    - TecioError on failure.
    """
    title = ctypes.c_char_p(0)

    ret = tecio.tecDataSetGetTitle(handle, ctypes.byref(title))
    if ret != 0:
        raise TecioError(
            f"Error getting data set title: handle={handle}, return_code={ret}"
        )

    return title.value.decode("utf-8")


def tec_data_set_get_num_vars(handle: ctypes.c_void_p) -> int:
    """
    Query the number of variables in the dataset.

    Inputs:
    - handle: ctypes.c_void_p file handle.

    Returns:
    - int: number of variables.

    Raises:
    - TecioError on failure.
    """
    num_vars = ctypes.c_int32(0)

    ret = tecio.tecDataSetGetNumVars(handle, ctypes.byref(num_vars))
    if ret != 0:
        raise TecioError(
            f"Error getting number of variables: handle={handle}, return_code={ret}"
        )

    return num_vars.value


def tec_data_set_get_num_zones(handle: ctypes.c_void_p) -> int:
    """
    Query the number of zones in the dataset.

    Inputs:
    - handle: ctypes.c_void_p file handle.

    Returns:
    - int: number of zones.

    Raises:
    - TecioError on failure.
    """
    num_zones = ctypes.c_int32(0)

    ret = tecio.tecDataSetGetNumZones(handle, ctypes.byref(num_zones))
    if ret != 0:
        raise TecioError(
            f"Error getting number of zones: handle={handle}, return_code={ret}"
        )

    return num_zones.value


# ---- Reading SZL zones ---------------------------------------------
def tec_zone_get_ijk(handle: ctypes.c_void_p, zone_index: int) -> Tuple[int, int, int]:
    """
    Get zone dimensions or FE counts.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index (int)

    Returns:
    - tuple (I, J, K):
      * For ORDERED zones: I,J,K are the zone dimensions.
      * For FE zones: I = number of nodes, J = number of elements, K unused.

    Raises:
    - TecioError on failure.
    """
    I = ctypes.c_int64(0)
    J = ctypes.c_int64(0)
    K = ctypes.c_int64(0)

    ret = tecio.tecZoneGetIJK(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(I),
        ctypes.byref(J),
        ctypes.byref(K),
    )
    if ret != 0:
        raise TecioError(
            f"Error getting zone data indices: : handle={handle}, zone_index={zone_index}, return_code={ret}"
        )

    return I.value, J.value, K.value


def tec_zone_get_title(handle: ctypes.c_void_p, zone_index: int) -> str:
    """
    Read the title for a given zone.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index

    Returns:
    - str: zone title (UTF-8 decoded)

    Raises:
    - TecioError on failure.
    """
    zone_title = ctypes.c_char_p(0)

    ret = tecio.tecZoneGetTitle(
        handle, ctypes.c_int32(zone_index), ctypes.byref(zone_title)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetTitle Error: handle={handle}, zone_index={zone_index}, return_code={ret}"
        )

    return zone_title.value.decode("utf-8")


def tec_zone_get_type(handle: ctypes.c_void_p, zone_index: int) -> ZoneType:
    """
    Query the ZoneType for the specified zone.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based index

    Returns:
    - ZoneType enum

    Raises:
    - TecioError on failure.
    """
    zone_type = ctypes.c_int32(0)

    ret = tecio.tecZoneGetType(
        handle, ctypes.c_int32(zone_index), ctypes.byref(zone_type)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetType Error: handle={handle}, zone_index={zone_index}, return_code={ret}"
        )

    return ZoneType(zone_type.value)


def tec_zone_is_enabled(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """
    Check whether a zone is enabled (not suppressed).

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based index

    Returns:
    - bool: True if enabled, False if suppressed.

    Raises:
    - TecioError on failure.
    """
    is_enabled = ctypes.c_int32(0)

    ret = tecio.tecZoneIsEnabled(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is_enabled)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneIsEnabled Error: handle={handle}, zone_index={zone_index}, return_code={ret}"
        )

    return bool(is_enabled.value)


def tec_zone_get_solution_time(handle: ctypes.c_void_p, zone_index: int) -> float:
    """
    Read the solution time for an unsteady zone.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based index

    Returns:
    - float: solution time (double precision)

    Raises:
    - TecioError on failure.
    """
    solution_time = ctypes.c_double(0)

    ret = tecio.tecZoneGetSolutionTime(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(solution_time),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetSolutionTime Error: handle={handle}, zone_index={zone_index}, return_code={ret}"
        )

    return solution_time.value


def tec_zone_get_strand_id(handle: ctypes.c_void_p, zone_index: int) -> int:
    """
    Get the strand ID for an unsteady zone.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based index

    Returns:
    - int: strand id

    Raises:
    - TecioError on failure.
    """
    strand_id = ctypes.c_int32(0)

    ret = tecio.tecZoneGetStrandID(
        handle, ctypes.c_int32(zone_index), ctypes.byref(strand_id)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetStrandID Error: handle={handle}, zone_index={zone_index}, return_code={ret}"
        )

    return strand_id.value


def is_64bit(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """
    Determine whether the zone's node-map indices are 64-bit.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based index

    Returns:
    - bool: True if node-map uses 64-bit indices, False if 32-bit.

    Raises:
    - TecioError on failure.
    """
    is64bit = ctypes.c_int32(0)
    ret = tecio.tecZoneNodeMapIs64Bit(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is64bit)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapIs64Bit Error: handle={handle}, zone_index={zone_index}, return_code={ret}"
        )

    return bool(is64bit.value)


def tec_zone_node_map_get_64(
    handle: ctypes.c_void_p,
    zone_index: int,
    num_elements: int,
    nodes_per_cell: int,
) -> npt.NDArray[np.int64]:
    """
    Read a 64-bit node-map for an FE zone.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based index
    - num_elements: number of elements/rows to read
    - nodes_per_cell: number of node indices per element (columns)

    Returns:
    - numpy.ndarray of shape (num_elements, nodes_per_cell) with dtype int64

    Raises:
    - TecioError on failure.
    """
    size_of_array = num_elements * nodes_per_cell
    nodemap = (ctypes.c_int64 * size_of_array)()

    ret = tecio.tecZoneNodeMapGet64(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int64(1),
        ctypes.c_int64(num_elements),
        ctypes.cast(
            nodemap,
            ctypes.POINTER(ctypes.c_int64),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapGet64 Error: handle={handle}, zone_index={zone_index}, "
            f"num_elements={num_elements}, nodes_per_cell={nodes_per_cell}, return_code={ret}"
        )

    return np.ctypeslib.as_array(nodemap).reshape(num_elements, nodes_per_cell)


def tec_zone_node_map_get(
    handle: ctypes.c_void_p,
    zone_index: int,
    num_elements: int,
    nodes_per_cell: int,
) -> npt.NDArray[np.int32]:
    """
    Read a 32-bit node-map for an FE zone.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based index
    - num_elements: number of elements/rows to read
    - nodes_per_cell: number of node indices per element (columns)

    Returns:
    - numpy.ndarray of shape (num_elements, nodes_per_cell) with dtype int32

    Raises:
    - TecioError on failure.
    """
    size_of_array = num_elements * nodes_per_cell
    nodemap = (ctypes.c_int32 * size_of_array)()

    ret = tecio.tecZoneNodeMapGet(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int64(1),
        ctypes.c_int64(num_elements),
        ctypes.cast(
            nodemap,
            ctypes.POINTER(ctypes.c_int32),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapGet Error: handle={handle}, zone_index={zone_index}, "
            f"num_elements={num_elements}, nodes_per_cell={nodes_per_cell}, return_code={ret}"
        )

    return np.ctypeslib.as_array(nodemap).reshape(num_elements, nodes_per_cell)


# ---- Reading SZL variable data -------------------------------------
def tec_var_get_name(handle: ctypes.c_void_p, var_index: int) -> str:
    """
    Get the name of a variable by index.

    Inputs:
    - handle: ctypes.c_void_p
    - var_index: 1-based variable index

    Returns:
    - str: variable name

    Raises:
    - TecioError on failure.
    """
    var_name = ctypes.c_char_p(0)

    ret = tecio.tecVarGetName(handle, ctypes.c_int32(var_index), ctypes.byref(var_name))
    if ret != 0:
        raise TecioError(
            f"tecVarGetName Error: handle={handle}, var_index={var_index}, return_code={ret}"
        )

    return var_name.value.decode("utf-8")


def tec_var_is_enabled(handle: ctypes.c_void_p, var_index: int) -> bool:
    """
    Check whether a variable is enabled.

    Inputs:
    - handle: ctypes.c_void_p
    - var_index: 1-based index

    Returns:
    - bool: True if enabled

    Raises:
    - TecioError on failure.
    """
    is_enabled = ctypes.c_int32(0)

    ret = tecio.tecVarIsEnabled(
        handle, ctypes.c_int32(var_index), ctypes.byref(is_enabled)
    )
    if ret != 0:
        raise TecioError(
            f"tecVarIsEnabled Error: handle={handle}, var_index={var_index}, return_code={ret}"
        )

    return bool(is_enabled.value)


def tec_zone_var_get_type(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> DataType:
    """
    Get the DataType for a variable in a zone.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index

    Returns:
    - DataType enum

    Raises:
    - TecioError on failure.
    """
    var_type = ctypes.c_int32(0)

    ret = tecio.tecZoneVarGetType(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(var_type),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetType Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return DataType(var_type.value)


def tec_zone_var_get_value_location(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> ValueLocation:
    """
    Get the value location (cell-centered or nodal) for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index

    Returns:
    - ValueLocation enum

    Raises:
    - TecioError on failure.
    """
    value_location = ctypes.c_int32(0)

    ret = tecio.tecZoneVarGetValueLocation(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(value_location),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetValueLocation Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return ValueLocation(value_location.value)


def tec_zone_var_is_passive(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> bool:
    """
    Check whether a zone variable is passive.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index

    Returns:
    - bool: True if variable is passive

    Raises:
    - TecioError on failure.
    """
    is_passive = ctypes.c_int32(0)

    ret = tecio.tecZoneVarIsPassive(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(is_passive),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarIsPassive Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return bool(is_passive.value)


def tec_zone_var_get_shared_zone(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> Optional[int]:
    """Wrapper for tecZoneVarGetSharedZone. Outputs shared zone index (0 if none)

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index

    Returns:
    - int | None: shared zone index (None if no shared zone)

    Raises:
    - TecioError on failure.
    """
    shared_zone = ctypes.c_int32(0)

    ret = tecio.tecZoneVarGetSharedZone(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(shared_zone),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetSharedZone Error: handle={handle}, zone_index={zone_index}, var_index={var_index}, return_code={ret}"
        )

    return shared_zone.value if shared_zone.value != 0 else None


def tec_zone_var_get_num_values(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> int:
    """
    Query how many values are available for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index

    Returns:
    - int: number of values for that variable in the zone

    Raises:
    - TecioError on failure.
    """
    num_values = ctypes.c_int32(0)

    ret = tecio.tecZoneVarGetNumValues(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(num_values),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetNumValues Error: handle={handle}, zone_index={zone_index}, var_index={var_index}, return_code={ret}"
        )

    return num_values.value


def tec_zone_var_get_float_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.float32]:
    """
    Read float32 values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index
    - start_index: 1-based start index to read from
    - num_values: number of values to read

    Returns:
    - numpy.ndarray (float32) of length num_values

    Raises:
    - TecioError on failure.
    """
    values = (ctypes.c_float * num_values)()

    ret = tecio.tecZoneVarGetFloatValues(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_float),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetFloatValues Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_double_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.float64]:
    """
    Read float64 (double) values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index
    - start_index: 1-based start index
    - num_values: number of values to read

    Returns:
    - numpy.ndarray (float64) of length num_values

    Raises:
    - TecioError on failure.
    """
    values = (ctypes.c_double * num_values)()

    ret = tecio.tecZoneVarGetDoubleValues(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_double),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetDoubleValues Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_int32_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.int32]:
    """
    Read int32 values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index
    - start_index: 1-based start index
    - num_values: number of values to read

    Returns:
    - numpy.ndarray (int32) of length num_values

    Raises:
    - TecioError on failure.
    """
    values = (ctypes.c_int32 * num_values)()

    ret = tecio.tecZoneVarGetInt32Values(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_int32),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetInt32Values Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_int16_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.int16]:
    """
    Read int16 values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index
    - start_index: 1-based start index
    - num_values: number of values to read

    Returns:
    - numpy.ndarray (int16) of length num_values

    Raises:
    - TecioError on failure.
    """
    values = (ctypes.c_int16 * num_values)()

    ret = tecio.tecZoneVarGetInt16Values(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_int16),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetInt16Values Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_uint8_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.uint8]:
    """
    Read uint8 values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - var_index: 1-based variable index
    - start_index: 1-based start index
    - num_values: number of values to read

    Returns:
    - numpy.ndarray (uint8) of length num_values

    Raises:
    - TecioError on failure.
    """
    values = (ctypes.c_uint8 * num_values)()

    ret = tecio.tecZoneVarGetUInt8Values(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_uint8),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetUInt8Values Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


# ---- Reading SZL aux data ------------------------------------------
def tec_data_set_aux_data_get_num_items(handle: ctypes.c_void_p) -> int:
    """
    Get the number of dataset-level auxiliary data items.

    Inputs:
    - handle: ctypes.c_void_p

    Returns:
    - int: number of aux-data items

    Raises:
    - TecioError on failure.
    """
    num_auxdata_items = ctypes.c_int32(0)

    ret = tecio.tecDataSetAuxDataGetNumItems(handle, ctypes.byref(num_auxdata_items))
    if ret != 0:
        raise TecioError(
            f"tecDataSetAuxDataGetNumItems Error: handle={handle}, return_code={ret}"
        )

    return num_auxdata_items.value


def tec_data_set_aux_data_get_item(
    handle: ctypes.c_void_p, item_index: int
) -> Tuple[str, str]:
    """
    Read a dataset-level auxiliary data item.

    Inputs:
    - handle: ctypes.c_void_p
    - item_index: 1-based item index

    Returns:
    - (name, value): tuple of strings

    Raises:
    - TecioError on failure.
    """
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = tecio.tecDataSetAuxDataGetItem(
        handle,
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecDataSetAuxDataGetItem Error: handle={handle}, item_index={item_index}, return_code={ret}"
        )

    return name.value.decode("utf-8"), value.value.decode("utf-8")


def tec_var_aux_data_get_num_items(handle: ctypes.c_void_p, var_index: int) -> int:
    """
    Get number of auxiliary data items attached to a variable.

    Inputs:
    - handle: ctypes.c_void_p
    - var_index: 1-based variable index

    Returns:
    - int: number of aux-data items

    Raises:
    - TecioError on failure.
    """
    num_items = ctypes.c_int32(0)

    ret = tecio.tecVarAuxDataGetNumItems(
        handle, ctypes.c_int32(var_index), ctypes.byref(num_items)
    )
    if ret != 0:
        raise TecioError(
            f"tecVarAuxDataGetNumItems Error: handle={handle}, var_index={var_index}, return_code={ret}"
        )

    return num_items.value


def tec_var_aux_data_get_item(
    handle: ctypes.c_void_p, var_index: int, item_index: int
) -> Tuple[str, str]:
    """
    Read a variable-level auxiliary data item.

    Inputs:
    - handle: ctypes.c_void_p
    - var_index: 1-based variable index
    - item_index: 1-based item index

    Returns:
    - (name, value): tuple of strings

    Raises:
    - TecioError on failure.
    """
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = tecio.tecVarAuxDataGetItem(
        handle,
        ctypes.c_int32(var_index),
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecVarAuxDataGetItem Error: handle={handle}, var_index={var_index}, item_index={item_index}, return_code={ret}"
        )

    return name.value.decode("utf-8"), value.value.decode("utf-8")


def tec_zone_aux_data_get_num_items(handle: ctypes.c_void_p, zone_index: int) -> int:
    """
    Get number of auxiliary data items attached to a zone.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index

    Returns:
    - int: number of aux-data items

    Raises:
    - TecioError on failure.
    """
    num_items = ctypes.c_int32(0)

    ret = tecio.tecZoneAuxDataGetNumItems(
        handle, ctypes.c_int32(zone_index), ctypes.byref(num_items)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneAuxDataGetNumItems Error: handle={handle}, zone_index={zone_index}, return_code={ret}"
        )

    return num_items.value


def tec_zone_aux_data_get_item(
    handle: ctypes.c_void_p, zone_index: int, item_index: int
) -> Tuple[str, str]:
    """
    Read a zone-level auxiliary data item.

    Inputs:
    - handle: ctypes.c_void_p
    - zone_index: 1-based zone index
    - item_index: 1-based item index

    Returns:
    - (name, value): tuple of strings

    Raises:
    - TecioError on failure.
    """
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = tecio.tecZoneAuxDataGetItem(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneAuxDataGetItem Error: handle={handle}, zone_index={zone_index}, item_index={item_index}, return_code={ret}"
        )

    return name.value.decode("utf-8"), value.value.decode("utf-8")


# --------------------------------------------------------------------
# ---- Wrappers for C functions: SZL Write ---------------------------
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


# ---- Create SZL objects --------------------------------------------
def tec_file_writer_open(
    file_name: str,
    dataset_title: str,
    var_names_csv: str,
    file_type: FileType,
    use_szl: int = 1,
    grid_file_handle: Optional[ctypes.c_void_p] = None,
) -> ctypes.c_void_p:
    """
    Open a writer handle for creating SZL (.szplt) files.

    Inputs:
    - file_name: output file path
    - dataset_title: dataset title string
    - var_names_csv: comma-separated variable names
    - file_type: FileType enum (FULL/GRID/SOLUTION)
    - use_szl: integer flag (1 to use SZL)
    - grid_file_handle: optional ctypes.c_void_p handle for a grid-only file when writing
      a solution file that references an existing grid.

    Returns:
    - ctypes.c_void_p: writer handle (to pass to other writer functions)

    Raises:
    - TypeError if file_type is not FileType
    - TecioError on non-zero tecio return code
    """
    if not isinstance(file_type, FileType):
        raise TypeError("file_type must be a szlio.FileType enum")

    handle = ctypes.c_void_p()
    ft_int = int(file_type.value)

    ret = tecio.tecFileWriterOpen(
        ctypes.c_char_p(file_name.encode("utf-8")),
        ctypes.c_char_p(dataset_title.encode("utf-8")),
        ctypes.c_char_p(var_names_csv.encode("utf-8")),
        ctypes.c_int32(use_szl),
        ctypes.c_int32(ft_int),
        ctypes.c_int32(0),
        grid_file_handle if grid_file_handle is not None else None,
        ctypes.byref(handle),
    )
    if ret != 0:
        raise TecioError(
            f"tecFileWriterOpen Error: file_name={file_name!r}, dataset_title={dataset_title!r}, "
            f"var_names={var_names_csv!r}, file_type={file_type!r}, return_code={ret}"
        )
    return handle


def tec_file_writer_close(handle: ctypes.c_void_p) -> None:
    """
    Close a writer handle and finalize the output file.

    Inputs:
    - handle: ctypes.c_void_p returned from tec_file_writer_open

    Returns:
    - None

    Raises:
    - TecioError on non-zero return code.
    """
    ret = tecio.tecFileWriterClose(ctypes.byref(handle))
    if ret != 0:
        raise TecioError(
            f"tecFileWriterClose Error: handle={handle}, return_code={ret}"
        )


def tec_zone_create_ijk(
    handle: ctypes.c_void_p,
    zone_title: str,
    I: int,
    J: int,
    K: int,
    var_types: Optional[Sequence[DataType]] = None,
    var_sharing: Optional[Sequence[int]] = None,
    value_locations: Optional[Sequence[ValueLocation]] = None,
) -> int:
    """
    Create an ordered I x J x K zone for writing.

    Inputs:
    - handle: ctypes.c_void_p writer handle
    - zone_title: zone title string
    - I, J, K: zone dimensions (integers)
    - var_types: optional sequence of DataType enums specifying storage type
      per variable (length should match dataset variables if provided)
    - var_sharing: optional sequence indicating variable sharing (per-var)
    - value_locations: optional sequence of ValueLocation enums per variable

    Returns:
    - int: created zone index (1-based) as returned by TecIO

    Raises:
    - TypeError if enum sequences are of incorrect type
    - TecioError on non-zero tecio return code
    """
    zone_out = ctypes.c_int32()

    var_types_ptr = None
    if var_types is not None:
        vt_list = []
        for v in var_types:
            if not isinstance(v, DataType):
                raise TypeError("All var_types entries must be szlio.DataType enums")
            vt_list.append(int(v.value))
        arr = (ctypes.c_int32 * len(vt_list))(*vt_list)
        var_types_ptr = arr

    var_sharing_ptr = None
    if var_sharing is not None:
        arr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))
        var_sharing_ptr = arr

    value_locations_ptr = None
    if value_locations is not None:
        vl_list = []
        for v in value_locations:
            if not isinstance(v, ValueLocation):
                raise TypeError(
                    "All value_locations entries must be szlio.ValueLocation enums"
                )
            vl_list.append(int(v.value))
        arr = (ctypes.c_int32 * len(vl_list))(*vl_list)
        value_locations_ptr = arr

    ret = tecio.tecZoneCreateIJK(
        handle,
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int64(I),
        ctypes.c_int64(J),
        ctypes.c_int64(K),
        var_types_ptr,
        var_sharing_ptr,
        value_locations_ptr,
        None,  # passiveVarList
        ctypes.c_int32(0),
        ctypes.c_int64(0),
        ctypes.c_int32(0),
        ctypes.byref(zone_out),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneCreateIJK Error: zone_title={zone_title!r}, I={I}, J={J}, K={K}, "
            f"var_types_len={len(var_types) if var_types is not None else 0}, return_code={ret}"
        )
    return zone_out.value


def tec_zone_set_unsteady_options(
    handle: ctypes.c_void_p, zone: int, strand: int = 0, solution_time: float = 0.0
) -> None:
    """
    Set unsteady (time/strand) metadata for a zone.

    Inputs:
    - handle: ctypes.c_void_p writer handle
    - zone: zone index (1-based)
    - strand: integer strand id
    - solution_time: double precision solution time

    Returns:
    - None

    Raises:
    - TecioError on non-zero return code.
    """
    ret = tecio.tecZoneSetUnsteadyOptions(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_double(solution_time),
        ctypes.c_int32(strand),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneSetUnsteadyOptions Error: zone={zone}, strand={strand}, "
            f"solution_time={solution_time}, return_code={ret}"
        )


# ---- Write variable value functions --------------------------------
def tec_zone_var_write_double_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """
    Write double-precision (float64) values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p writer handle
    - zone: 1-based zone index
    - var: 1-based variable index
    - values: array-like of float64 values (will be converted to contiguous np.float64)

    Returns:
    - None

    Raises:
    - TecioError on non-zero return code.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(
        values, np.float64, ctypes.c_double
    )

    ret = tecio.tecZoneVarWriteDoubleValues(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteDoubleValues Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


def tec_zone_var_write_float_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """
    Write single-precision (float32) values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p writer handle
    - zone: 1-based zone index
    - var: 1-based variable index
    - values: array-like of float32 values (converted to contiguous np.float32)

    Returns:
    - None

    Raises:
    - TecioError on non-zero return code.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.float32, ctypes.c_float)

    ret = tecio.tecZoneVarWriteFloatValues(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteFloatValues Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


def tec_zone_var_write_int32_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """
    Write int32 values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p writer handle
    - zone: 1-based zone index
    - var: 1-based variable index
    - values: array-like of int32 values (converted to contiguous np.int32)

    Returns:
    - None

    Raises:
    - TecioError on non-zero return code.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.int32, ctypes.c_int32)

    ret = tecio.tecZoneVarWriteInt32Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteInt32Values Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


def tec_zone_var_write_int16_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """
    Write int16 values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p writer handle
    - zone: 1-based zone index
    - var: 1-based variable index
    - values: array-like of int16 values (converted to contiguous np.int16)

    Returns:
    - None

    Raises:
    - TecioError on non-zero return code.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.int16, ctypes.c_int16)

    ret = tecio.tecZoneVarWriteInt16Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteInt16Values Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


def tec_zone_var_write_uint8_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """
    Write unsigned 8-bit values for a zone variable.

    Inputs:
    - handle: ctypes.c_void_p writer handle
    - zone: 1-based zone index
    - var: 1-based variable index
    - values: array-like of uint8 values (converted to contiguous np.uint8)

    Returns:
    - None

    Raises:
    - TecioError on non-zero return code.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.uint8, ctypes.c_uint8)

    ret = tecio.tecZoneVarWriteUInt8Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteUInt8Values Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )
