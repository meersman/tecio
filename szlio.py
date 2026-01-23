from __future__ import annotations

import ctypes
from enum import Enum
from typing import Tuple

import numpy as np
import numpy.typing as npt

import tecutils

TECIO_LIB_PATH = tecutils.get_tecio_lib()
tecio = ctypes.cdll.LoadLibrary(TECIO_LIB_PATH)


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

class SzlioError(RuntimeError):
    """Runtime error for szlio functions"""

# --------------------------------------------------------------------
# ---- Wrappers for C functions: SZL Read ----------------------------
# --------------------------------------------------------------------


# ---- Reading SZL files ---------------------------------------------
def tec_file_reader_open(file_name: str) -> ctypes.c_void_p:
    """
    Wrapper for tecFileReaderOpen. Modifies generated file handle C pointer

    Input: file name path string
    Output: C pointer to file handle (used for other tecio functions)
    """
    handle = ctypes.c_void_p(0)

    ret = tecio.tecFileReaderOpen(
        ctypes.c_char_p(bytes(file_name, encoding="UTF-8")),
        ctypes.byref(handle),
    )
    if ret != 0:
        raise Exception("SZLFile Initialization Error")

    return handle


def tec_file_get_type(handle: ctypes.c_void_p) -> FileType:
    """
    Wrapper for tecFileGetType. Outputs file type. Either full, grid, or solution.

    Input: file handle C pointer
    Output: FileType (class defined above)
    """
    file_type = ctypes.c_int32(0)

    ret = tecio.tecFileGetType(handle, ctypes.byref(file_type))
    if ret != 0:
        raise Exception("tecFileGetType Error")

    return FileType(file_type.value)


def tec_data_set_get_title(handle: ctypes.c_void_p) -> str:
    """
    Wrapper for tecDataSetGetTitle. Outputs dataset title string.

    Input: file handle C pointer
    Output: title string
    """
    title = ctypes.c_char_p(0)

    ret = tecio.tecDataSetGetTitle(handle, ctypes.byref(title))
    if ret != 0:
        raise Exception("tecDataSetGetTitle Error")

    return title.value.decode("utf-8")


def tec_data_set_get_num_vars(handle: ctypes.c_void_p) -> int:
    """
    Wrapper for tecDataSetGetNumVars. Outputs integer number of variables.

    Input: file handle C pointer
    Output: integer number of variables
    """
    num_vars = ctypes.c_int32(0)

    ret = tecio.tecDataSetGetNumVars(handle, ctypes.byref(num_vars))
    if ret != 0:
        raise Exception("tecDataGetNumVars Error")

    return num_vars.value


def tec_data_set_get_num_zones(handle: ctypes.c_void_p) -> int:
    """
    Wrapper for tecDataSetGetNumZones. Outputs integer number of zones.

    Input: file handle C pointer
    Output: integer number of zones
    """
    num_zones = ctypes.c_int32(0)

    ret = tecio.tecDataSetGetNumZones(handle, ctypes.byref(num_zones))
    if ret != 0:
        raise Exception("tecDataGetNumZones Error")

    return num_zones.value


def tec_data_set_aux_data_get_num_items(handle: ctypes.c_void_p) -> int:
    """
    Wrapper for tecDataSetAuxDataGetNumItems. Outputs integer number of
    total auxiliary data tokens.

    Input: file handle C pointer
    Output: integer number of auxiliary data tokens
    """
    num_auxdata_items = ctypes.c_int32(0)

    ret = tecio.tecDataSetAuxDataGetNumItems(handle, ctypes.byref(num_auxdata_items))
    if ret != 0:
        raise Exception("tecDataSetAuxDataGetNumItems Error")

    return num_auxdata_items.value


# ---- Reading SZL zones ---------------------------------------------
def tec_zone_get_ijk(handle: ctypes.c_void_p, zone_index: int) -> Tuple[int, int, int]:
    """
    Wrapper for tecZoneGetIJK. Gets zone indices. Different behavior
    for FE vs Ordered, see below.

    Input: file handle C pointer, zone index integer
    Output: integer number of auxiliary data tokens
    For Ordered zones, I, J, K are the dimensions of the zone
    For FE Zones:
    - I = number of points
    - J = number of elements
    - K = 0 / not used
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
        raise Exception("tecZoneGetIJK Error")

    return I.value, J.value, K.value


def tec_zone_get_title(handle: ctypes.c_void_p, zone_index: int) -> str:
    """
    Wrapper for tecZoneGetTitle. Returns zone tile string for input
    index.

    Input: file handle C pointer, zone index integer
    Output: zone title string
    """
    zone_title = ctypes.c_char_p(0)

    ret = tecio.tecZoneGetTitle(
        handle, ctypes.c_int32(zone_index), ctypes.byref(zone_title)
    )
    if ret != 0:
        raise Exception("tecZoneGetTitle Error")

    return zone_title.value.decode("utf-8")


def tec_zone_get_type(handle: ctypes.c_void_p, zone_index: int) -> ZoneType:
    """
    Wrapper for tecZoneGetType. Returns zone type (see enum def).

    Input: file handle C pointer, zone index integer
    Output: type of zone
    """
    zone_type = ctypes.c_int32(0)

    ret = tecio.tecZoneGetType(
        handle, ctypes.c_int32(zone_index), ctypes.byref(zone_type)
    )
    if ret != 0:
        raise Exception("tecZoneGetType Error")

    return ZoneType(zone_type.value)


def tec_zone_is_enabled(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """
    Wrapper for tecZoneIsEnabled. Returns True/False if zone is
    suppressed or not.

    Input: file handle C pointer, zone index integer
    Output: True/False
    """
    is_enabled = ctypes.c_int32(0)

    ret = tecio.tecZoneIsEnabled(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is_enabled)
    )
    if ret != 0:
        raise Exception("tecZoneIsEnabled Error")

    return bool(is_enabled.value)


def tec_zone_get_solution_time(handle: ctypes.c_void_p, zone_index: int) -> int:
    solution_time = ctypes.c_double(0)

    ret = tecio.tecZoneGetSolutionTime(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(solution_time),
    )
    if ret != 0:
        raise Exception("tecZoneGetSolutionTime Error")

    return solution_time.value


def tec_zone_get_strand_id(handle: ctypes.c_void_p, zone_index: int) -> int:
    strand_id = ctypes.c_int32(0)

    ret = tecio.tecZoneGetStrandID(
        handle, ctypes.c_int32(zone_index), ctypes.byref(strand_id)
    )
    if ret != 0:
        raise Exception("tecZoneGetStrandID Error")

    return strand_id.value


def is_64bit(handle: ctypes.c_void_p, zone_index: int) -> bool:

    is64bit = ctypes.c_int32(0)
    ret = tecio.tecZoneNodeMapIs64Bit(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is64bit)
    )
    if ret != 0:
        raise Exception("tecZoneNodeMapIs64Bit Error")

    return bool(is64bit.value)


def tec_zone_node_map_get_64(
    handle: ctypes.c_void_p,
    zone_index: int,
    num_elements: int,
    nodes_per_cell: int,
) -> npt.NDArray[np.int64]:
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
        raise Exception("tecZoneNodeMapGet64 Error")

    return np.ctypeslib.as_array(nodemap).reshape(num_elements, nodes_per_cell)


def tec_zone_node_map_get(
    handle: ctypes.c_void_p,
    zone_index: int,
    num_elements: int,
    nodes_per_cell: int,
) -> npt.NDArray[np.int32]:
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
        raise Exception("tecZoneNodeMapGet64 Error")

    return np.ctypeslib.as_array(nodemap).reshape(num_elements, nodes_per_cell)


# ---- Reading SZL variable data -------------------------------------
def tec_var_get_name(handle: ctypes.c_void_p, var_index: int) -> str:
    var_name = ctypes.c_char_p(0)

    ret = tecio.tecVarGetName(handle, ctypes.c_int32(var_index), ctypes.byref(var_name))
    if ret != 0:
        raise Exception("tecVarGetName Error")

    return var_name.value.decode("utf-8")


def tec_var_is_enabled(handle: ctypes.c_void_p, var_index: int) -> bool:
    is_enabled = ctypes.c_int32(0)

    ret = tecio.tecVarIsEnabled(
        handle, ctypes.c_int32(var_index), ctypes.byref(is_enabled)
    )
    if ret != 0:
        raise Exception("tecZoneIsEnabled Error")

    return bool(is_enabled.value)


def tec_zone_var_get_type(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> DataType:
    var_type = ctypes.c_int32(0)

    ret = tecio.tecZoneVarGetType(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(var_type),
    )
    if ret != 0:
        raise Exception("tecZoneVarGetType Error")

    return DataType(var_type.value)


def tec_zone_var_get_value_location(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> ValueLocation:
    value_location = ctypes.c_int32(0)

    ret = tecio.tecZoneVarGetValueLocation(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(value_location),
    )
    if ret != 0:
        raise Exception("tecZoneVarGetValueLocation Error")

    return ValueLocation(value_location.value)


def tec_zone_var_is_passive(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> bool:
    is_passive = ctypes.c_int32(0)

    ret = tecio.tecZoneVarIsPassive(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(is_passive),
    )
    if ret != 0:
        raise Exception("tecZoneVarIsPassive Error")

    return bool(is_passive.value)


def tec_zone_var_get_shared_zone(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> int | None:
    """Wrapper for tecZoneVarGetSharedZone. Outputs shared zone index (0 if none)"""
    shared_zone = ctypes.c_int32(0)

    ret = tecio.tecZoneVarGetSharedZone(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(shared_zone),
    )
    if ret != 0:
        raise Exception("tecZoneVarGetSharedZone Error")

    return shared_zone.value if shared_zone.value != 0 else None


def tec_zone_var_get_num_values(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> int:
    num_values = ctypes.c_int32(0)

    ret = tecio.tecZoneVarGetNumValues(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(num_values),
    )
    if ret != 0:
        raise Exception("tecZoneVarGetNumValues Error")

    return num_values.value


def tec_zone_var_get_float_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.float32]:
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
        raise Exception("tecZoneVarGetFloatValues Error")

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_double_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.float64]:
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
        raise Exception("tecZoneVarGetDoubleValues Error")

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_int32_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.int32]:
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
        raise Exception("tecZoneVarGetInt32Values Error")

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_int16_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.int16]:
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
        raise Exception("tecZoneVarGetInt16Values Error")

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_uint8_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.uint8]:
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
        raise Exception("tecZoneVarGetUInt8Values Error")

    return np.ctypeslib.as_array(values)
