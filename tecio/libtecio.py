"""Python bindings for the TecIO C library.

Provides enum types for Tecplot constants, ctypes bindings for both the
new SZL API (``tec_*``) and the classic PLT API (``tec*142``), and
Python wrapper functions for each C entry point.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _utils

# Load tecio library
TECIO_LIB_PATH = _utils.get_tecio_lib()
lib = ctypes.cdll.LoadLibrary(TECIO_LIB_PATH)


class TecioError(RuntimeError):
    """Exception for TecIO C library errors."""


# ======================================================================================
# Meaningful integers
# - The TecIO library often uses integers with special meanings (zone types, data types,
#   data locations)
# - The same values are used both for writing (tec*142 functions) and for SZL reading
#   and writing (tec_* functions)
# - The classes below provide a more readable format of these values
# - Where available, the equivalent keywords used in Tecplot ASCII files are set as the
#   class property, returning the corresponding int value.
# ======================================================================================


class FileFormat(Enum):
    """Binary data format selector.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``PLT``
         - ``0``
         - Classic PLT binary format.
       * - ``SZPLT``
         - ``1``
         - SZL subzone-loadable format.
    """

    PLT = 0
    SZPLT = 1


class FileType(Enum):
    """Tecplot file type.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``FULL``
         - ``0``
         - Contains both grid and solution data.
       * - ``GRID``
         - ``1``
         - Grid coordinates only.
       * - ``SOLUTION``
         - ``2``
         - Solution variables only.
    """

    FULL = 0
    GRID = 1
    SOLUTION = 2


class ZoneType(Enum):
    """Tecplot zone type.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``ORDERED``
         - ``0``
         - Structured IJK grid.
       * - ``FELINESEG``
         - ``1``
         - Finite-element line segments.
       * - ``FETRIANGLE``
         - ``2``
         - Finite-element triangles.
       * - ``FEQUADRILATERAL``
         - ``3``
         - Finite-element quadrilaterals.
       * - ``FETETRAHEDRON``
         - ``4``
         - Finite-element tetrahedra.
       * - ``FEBRICK``
         - ``5``
         - Finite-element hexahedra.
       * - ``FEPOLYGON``
         - ``6``
         - Finite-element polygons (face-based).
       * - ``FEPOLYHEDRON``
         - ``7``
         - Finite-element polyhedra (face-based).
       * - ``FEMIXED``
         - ``8``
         - Mixed finite-element types.
    """

    ORDERED = 0
    FELINESEG = 1
    FETRIANGLE = 2
    FEQUADRILATERAL = 3
    FETETRAHEDRON = 4
    FEBRICK = 5
    FEPOLYGON = 6
    FEPOLYHEDRON = 7
    FEMIXED = 8


class FeCellShape(Enum):
    """Unstructured cell shape category.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``BAR``
         - ``0``
         - 2D two-node element.
       * - ``TRIANGLE``
         - ``1``
         - 2D three-node element.
       * - ``QUADRILATERAL``
         - ``2``
         - 2D four-node element.
       * - ``TETRAHEDRON``
         - ``3``
         - 3D four-node element.
       * - ``HEXAHEDRON``
         - ``4``
         - 3D six-node element.
       * - ``PYRAMID``
         - ``5``
         - 3D five-node element.
       * - ``PRISM``
         - ``6``
         - 3D eight-node element.
    """

    BAR = 0
    TRIANGLE = 1
    QUADRILATERAL = 2
    TETRAHEDRON = 3
    HEXAHEDRON = 4
    PYRAMID = 5
    PRISM = 6


class FaceNeighborMode(Enum):
    """Boundary face-sharing mode between zones.

    .. list-table::
       :header-rows: 1
       :widths: 35 10 55

       * - Attribute
         - Value
         - Description
       * - ``LOCAL_ONE_TO_ONE``
         - ``0``
         - Each face has at most one local neighbor.
       * - ``LOCAL_ONE_TO_MANY``
         - ``1``
         - Each face may have multiple local neighbors (hanging nodes).
       * - ``GLOBAL_ONE_TO_ONE``
         - ``2``
         - Each face has at most one neighbor in any zone.
       * - ``GLOBAL_ONE_TO_MANY``
         - ``3``
         - Each face may have multiple neighbors in any zone (hanging nodes).
    """

    LOCAL_ONE_TO_ONE = 0
    LOCAL_ONE_TO_MANY = 1
    GLOBAL_ONE_TO_ONE = 2
    GLOBAL_ONE_TO_MANY = 3


class ValueLocation(Enum):
    """Data value location within a cell.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``CELL_CENTERED``
         - ``0``
         - Values stored at cell centres.
       * - ``NODAL``
         - ``1``
         - Values stored at grid nodes.
    """

    CELL_CENTERED = 0
    NODAL = 1


class DataPacking(Enum):
    """Zone data packing order for ASCII (``.dat``) files.

    Controls whether data is laid out variable-by-variable or point-by-point
    in the ASCII file.  The ``DATAPACKING`` keyword in a zone header takes one
    of these two values.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``POINT``
         - ``0``
         - One row per node/cell containing all variable values.
       * - ``BLOCK``
         - ``1``
         - One contiguous block per variable containing all node/cell values.
           Tecplot default; faster for variable-at-a-time access patterns.
    """

    POINT = 0
    BLOCK = 1


class DataType(Enum):
    """On-disk storage type for variable data.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``FLOAT``
         - ``1``
         - 32-bit IEEE floating point.
       * - ``DOUBLE``
         - ``2``
         - 64-bit IEEE floating point.
       * - ``INT32``
         - ``3``
         - 32-bit signed integer.
       * - ``INT16``
         - ``4``
         - 16-bit signed integer.
       * - ``BYTE``
         - ``5``
         - 8-bit unsigned integer.
    """

    FLOAT = 1
    DOUBLE = 2
    INT32 = 3
    INT16 = 4
    BYTE = 5


class VarStatus(Enum):
    """Variable active/passive flag.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``ACTIVE``
         - ``0``
         - Variable has data in this zone.
       * - ``PASSIVE``
         - ``1``
         - Variable has no data in this zone.
    """

    ACTIVE = 0
    PASSIVE = 1


class Boolean(Enum):
    """Boolean flag for C function arguments.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``FALSE``
         - ``0``
         - Logical false.
       * - ``TRUE``
         - ``1``
         - Logical true.
    """

    FALSE = 0
    TRUE = 1


class Debug(Enum):
    """Debug flag for C function arguments.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``FALSE``
         - ``0``
         - Debug output disabled.
       * - ``TRUE``
         - ``1``
         - Debug output enabled.
    """

    FALSE = 0
    TRUE = 1


# ======================================================================================
# Helper functions
# - Used to convert Numpy array-like input to C-compatible pointers for passing to the C
#   API
# - Used to convert special integer values (enums, ints, or sequences) to int values for
#   passing to the C API, with optional validation against an Enum class
# ======================================================================================


def _prepare_array_for_ctypes(
    values: npt.ArrayLike, np_dtype, ctype
) -> tuple[Any, int, npt.NDArray]:
    """Convert an input array to contiguous numpy array and return a ctypes pointer.

    Args:
        values: array-like (list, tuple, numpy array)
        np_dtype: numpy dtype object or type (e.g. np.float32)
        ctype: corresponding ctypes scalar type (e.g. ctypes.c_float)

    Returns:
        ptr: ctypes pointer suitable for passing to the C API
        count: int number of elements
        backing_array: the numpy array object (returned to keep it alive)

    Note:
        Caller should keep the returned backing_array alive until the native call
        completes.

    Note:
        This function enforces dtype and C-contiguity.
    """
    arr = np.ascontiguousarray(values, dtype=np_dtype)
    count = int(arr.size)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctype))
    return ptr, count, arr


def _to_int_value(value: int | Enum, enum_class: type[Enum] | None = None) -> int:
    """Convert Enum or int to int value.

    (Optional check against enum_class if provided.)
    """
    if isinstance(value, Enum):
        return value.value
    v = int(value)
    if enum_class is not None:
        try:
            enum_class(v)
        except ValueError as e:
            raise ValueError(
                f"Invalid value {v} for enum {enum_class.__name__}. "
                f"Valid values: {[e.value for e in enum_class]}"
            ) from e
    return v


def _process_sequence(seq: Sequence[int | Enum] | None) -> ctypes.Array | None:
    """Convert sequence of int/Enum to ctypes array, handling None."""
    if seq is None:
        return None
    values = [_to_int_value(v) for v in seq]
    return (ctypes.c_int32 * len(values))(*values)


def _decode(value: bytes | None) -> str:
    """Decode a ctypes c_char_p value to str, raising if unexpectedly null."""
    if value is None:
        raise ValueError("Unexpected null pointer returned from TecIO C library.")
    return value.decode("utf-8")


# ======================================================================================
# C library bindings
# - Set the input and output C types for each C function used in this library
# ======================================================================================

# --------------------------------------------------------------------------------------
# New SZL API bindings
# --------------------------------------------------------------------------------------

# Reading SZL files
lib.tecFileReaderOpen.restype = ctypes.c_int32
lib.tecFileReaderOpen.argtypes = [
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
]
lib.tecFileReaderClose.restype = ctypes.c_int32
lib.tecFileReaderClose.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
]
lib.tecFileGetType.restype = ctypes.c_int32
lib.tecFileGetType.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecDataSetGetTitle.restype = ctypes.c_int32
lib.tecDataSetGetTitle.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
]
lib.tecDataSetGetNumVars.restype = ctypes.c_int32
lib.tecDataSetGetNumVars.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecDataSetGetNumZones.restype = ctypes.c_int32
lib.tecDataSetGetNumZones.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecDataSetAuxDataGetNumItems.restype = ctypes.c_int32
lib.tecDataSetAuxDataGetNumItems.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]

# Reading SZL zones
lib.tecZoneGetIJK.restype = ctypes.c_int32
lib.tecZoneGetIJK.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
]
lib.tecZoneGetTitle.restype = ctypes.c_int32
lib.tecZoneGetTitle.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
]
lib.tecZoneGetType.restype = ctypes.c_int32
lib.tecZoneGetType.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneIsEnabled.restype = ctypes.c_int32
lib.tecZoneIsEnabled.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneGetSolutionTime.restype = ctypes.c_int32
lib.tecZoneGetSolutionTime.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_double),
]
lib.tecZoneGetStrandID.restype = ctypes.c_int32
lib.tecZoneGetStrandID.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneNodeMapIs64Bit.restype = ctypes.c_int32
lib.tecZoneNodeMapIs64Bit.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneNodeMapGet64.restype = ctypes.c_int32
lib.tecZoneNodeMapGet64.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64),
]
lib.tecZoneNodeMapGet.restype = ctypes.c_int32
lib.tecZoneNodeMapGet.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int32),
]

# Reading SZL variable data
lib.tecVarGetName.restype = ctypes.c_int32
lib.tecVarGetName.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
]
lib.tecVarIsEnabled.restype = ctypes.c_int32
lib.tecVarIsEnabled.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneVarGetType.restype = ctypes.c_int32
lib.tecZoneVarGetType.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneVarGetValueLocation.restype = ctypes.c_int32
lib.tecZoneVarGetValueLocation.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneVarIsPassive.restype = ctypes.c_int32
lib.tecZoneVarIsPassive.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneVarGetSharedZone.restype = ctypes.c_int32
lib.tecZoneVarGetSharedZone.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneConnectivityGetSharedZone.restype = ctypes.c_int32
lib.tecZoneConnectivityGetSharedZone.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneVarGetNumValues.restype = ctypes.c_int32
lib.tecZoneVarGetNumValues.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneVarGetFloatValues.restype = ctypes.c_int32
lib.tecZoneVarGetFloatValues.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_float),  # Values
]
lib.tecZoneVarGetDoubleValues.restype = ctypes.c_int32
lib.tecZoneVarGetDoubleValues.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_double),  # Values
]
lib.tecZoneVarGetInt32Values.restype = ctypes.c_int32
lib.tecZoneVarGetInt32Values.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_int32),  # Values
]
lib.tecZoneVarGetInt16Values.restype = ctypes.c_int32
lib.tecZoneVarGetInt16Values.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_int16),  # Values
]
lib.tecZoneVarGetUInt8Values.restype = ctypes.c_int32
lib.tecZoneVarGetUInt8Values.argtypes = [
    ctypes.c_void_p,  # File Handle
    ctypes.c_int32,  # Zone index
    ctypes.c_int32,  # Variable index
    ctypes.c_int64,  # Start index
    ctypes.c_int64,  # Number of values
    ctypes.POINTER(ctypes.c_uint8),  # Values
]

# Reading SZL aux data
lib.tecDataSetAuxDataGetNumItems.restype = ctypes.c_int32
lib.tecDataSetAuxDataGetNumItems.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecDataSetAuxDataGetItem.restype = ctypes.c_int32
lib.tecDataSetAuxDataGetItem.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]
lib.tecVarAuxDataGetNumItems.restype = ctypes.c_int32
lib.tecVarAuxDataGetNumItems.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecVarAuxDataGetItem.restype = ctypes.c_int32
lib.tecVarAuxDataGetItem.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]
lib.tecZoneAuxDataGetNumItems.restype = ctypes.c_int32
lib.tecZoneAuxDataGetNumItems.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneAuxDataGetItem.restype = ctypes.c_int32
lib.tecZoneAuxDataGetItem.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
]

# Output file initialization and file handling
lib.tecFileWriterOpen.restype = ctypes.c_int32
lib.tecFileWriterOpen.argtypes = [
    ctypes.c_char_p,  # fileName
    ctypes.c_char_p,  # dataSetTitle
    ctypes.c_char_p,  # varNames (comma separated)
    ctypes.c_int32,  # useSZL (1)
    ctypes.c_int32,  # fileType
    ctypes.c_int32,  # reserved / options
    ctypes.c_void_p,  # gridFileHandle (optional)
    ctypes.POINTER(ctypes.c_void_p),  # out fileHandle
]
lib.tecFileWriterFlush.restype = ctypes.c_int32
lib.tecFileWriterFlush.argtypes = [
    ctypes.c_void_p,  # fileHandle
    ctypes.c_int32,  # numZonesToRetain
    ctypes.POINTER(ctypes.c_int32),  # zonesToRetain
]
lib.tecFileWriterClose.restype = ctypes.c_int32
lib.tecFileWriterClose.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
]

# Write Zone Headers
lib.tecZoneCreateIJK.restype = ctypes.c_int32
lib.tecZoneCreateIJK.argtypes = [
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
lib.tecZoneCreateFE.restype = ctypes.c_int32
lib.tecZoneCreateFE.argtypes = [
    ctypes.c_void_p,  # file_handle
    ctypes.c_char_p,  # zoneTitle
    ctypes.c_int32,  # ZoneType
    ctypes.c_int64,  # numNodes
    ctypes.c_int64,  # numCells
    ctypes.POINTER(ctypes.c_int32),  # varTypes
    ctypes.POINTER(ctypes.c_int32),  # shareVarFromZone
    ctypes.POINTER(ctypes.c_int32),  # valueLocations
    ctypes.POINTER(ctypes.c_int32),  # passiveVarList
    ctypes.c_int32,  # shareFaceNeighborsFromZone
    ctypes.c_int64,  # numFaceConnections
    ctypes.c_int32,  # faceNeighborMode
    ctypes.POINTER(ctypes.c_int32),  # out zone
]

# Optional fields
lib.tecZoneSetUnsteadyOptions.restype = ctypes.c_int32
lib.tecZoneSetUnsteadyOptions.argtypes = [
    ctypes.c_void_p,  # file_handle
    ctypes.c_int32,  # zone
    ctypes.c_double,  # solutionTime
    ctypes.c_int32,  # strand
]
lib.tecDataSetAddAuxData.restype = ctypes.c_int32
lib.tecDataSetAddAuxData.argtypes = [
    ctypes.c_void_p,  # fileHandle
    ctypes.c_char_p,  # name
    ctypes.c_char_p,  # value
]
lib.tecVarAddAuxData.restype = ctypes.c_int32
lib.tecVarAddAuxData.argtypes = [
    ctypes.c_void_p,  # fileHandle
    ctypes.c_int32,  # varIndex (1-based)
    ctypes.c_char_p,  # name
    ctypes.c_char_p,  # value
]
lib.tecZoneAddAuxData.restype = ctypes.c_int32
lib.tecZoneAddAuxData.argtypes = [
    ctypes.c_void_p,  # fileHandle
    ctypes.c_int32,  # zoneIndex (1-based)
    ctypes.c_char_p,  # name
    ctypes.c_char_p,  # value
]

# Write variable value functions
lib.tecZoneVarWriteDoubleValues.restype = ctypes.c_int32
lib.tecZoneVarWriteDoubleValues.argtypes = [
    ctypes.c_void_p,  # file handle
    ctypes.c_int32,  # zone index
    ctypes.c_int32,  # variable index
    ctypes.c_int32,  # partition index (0 for non-partitioned zones)
    ctypes.c_int64,  # number of values to write
    ctypes.POINTER(ctypes.c_double),  # pointer to values array
]
lib.tecZoneVarWriteFloatValues.restype = ctypes.c_int32
lib.tecZoneVarWriteFloatValues.argtypes = [
    ctypes.c_void_p,  # file handle
    ctypes.c_int32,  # zone index
    ctypes.c_int32,  # variable index
    ctypes.c_int32,  # partition index (0 for non-partitioned zones)
    ctypes.c_int64,  # number of values to write
    ctypes.POINTER(ctypes.c_float),  # pointer to values array
]
lib.tecZoneVarWriteInt32Values.restype = ctypes.c_int32
lib.tecZoneVarWriteInt32Values.argtypes = [
    ctypes.c_void_p,  # file handle
    ctypes.c_int32,  # zone index
    ctypes.c_int32,  # variable index
    ctypes.c_int32,  # partition index (0 for non-partitioned zones)
    ctypes.c_int64,  # number of values to write
    ctypes.POINTER(ctypes.c_int32),  # pointer to values array
]
lib.tecZoneVarWriteInt16Values.restype = ctypes.c_int32
lib.tecZoneVarWriteInt16Values.argtypes = [
    ctypes.c_void_p,  # file handle
    ctypes.c_int32,  # zone index
    ctypes.c_int32,  # variable index
    ctypes.c_int32,  # partition index (0 for non-partitioned zones)
    ctypes.c_int64,  # number of values to write
    ctypes.POINTER(ctypes.c_int16),  # pointer to values array
]
lib.tecZoneVarWriteUInt8Values.restype = ctypes.c_int32
lib.tecZoneVarWriteUInt8Values.argtypes = [
    ctypes.c_void_p,  # file handle
    ctypes.c_int32,  # zone index
    ctypes.c_int32,  # variable index
    ctypes.c_int32,  # partition index (0 for non-partitioned zones)
    ctypes.c_int64,  # number of values to write
    ctypes.POINTER(ctypes.c_uint8),  # pointer to values array
]
lib.tecZoneVarWriteUInt8Values.argtypes = [
    ctypes.c_void_p,  # file handle
    ctypes.c_int32,  # zone index
    ctypes.c_int32,  # variable index
    ctypes.c_int32,  # partition index (0 for non-partitioned zones)
    ctypes.c_int64,  # number of values to write
    ctypes.POINTER(ctypes.c_uint8),  # pointer to values array
]

# Write Zone Connectivity (FE zones only)
lib.tecZoneNodeMapWrite32.restype = ctypes.c_int32
lib.tecZoneNodeMapWrite32.argtypes = [
    ctypes.c_void_p,  # fileHandle
    ctypes.c_int32,  # zone index(1-based)
    ctypes.c_int32,  # partition index (MPI)
    ctypes.c_int32,  # isOneBased (Boolean)
    ctypes.c_int64,  # nodeCount
    ctypes.POINTER(ctypes.c_int32),  # array of nodes
]
lib.tecZoneNodeMapWrite64.restype = ctypes.c_int32
lib.tecZoneNodeMapWrite64.argtypes = [
    ctypes.c_void_p,  # fileHandle
    ctypes.c_int32,  # zone index(1-based)
    ctypes.c_int32,  # partition index (MPI)
    ctypes.c_int32,  # isOneBased (Boolean)
    ctypes.c_int64,  # nodeCount
    ctypes.POINTER(ctypes.c_int64),  # array of nodes
]
lib.tecZoneFaceNbrWriteConnections32.restype = ctypes.c_int32
lib.tecZoneFaceNbrWriteConnections32.argtypes = [
    ctypes.c_void_p,  # fileHandle
    ctypes.c_int32,  # zone (1-based)
    ctypes.POINTER(ctypes.c_int32),  # faceNeighbors
]
lib.tecZoneFaceNbrWriteConnections64.restype = ctypes.c_int32
lib.tecZoneFaceNbrWriteConnections64.argtypes = [
    ctypes.c_void_p,  # fileHandle
    ctypes.c_int32,  # zone (1-based)
    ctypes.POINTER(ctypes.c_int64),  # faceNeighbors
]

# --------------------------------------------------------------------------------------
# Classic API bindings
# --------------------------------------------------------------------------------------

# File initialization and finalization
lib.tecini142.restype = ctypes.c_int32
lib.tecini142.argtypes = [
    ctypes.c_char_p,  # Title
    ctypes.c_char_p,  # Variables
    ctypes.c_char_p,  # FName
    ctypes.c_char_p,  # ScratchDir
    ctypes.POINTER(ctypes.c_int32),  # FileFormat (0=PLT, 1=SZPLT)
    ctypes.POINTER(ctypes.c_int32),  # FileType (0=FULL, 1=GRID, 2=SOLUTION)
    ctypes.POINTER(ctypes.c_int32),  # Debug
    ctypes.POINTER(ctypes.c_int32),  # VIsDouble
]
lib.tecend142.restype = ctypes.c_int32
lib.tecend142.argtypes = []
lib.tecflush142.restype = ctypes.c_int32
lib.tecflush142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # NumZonesToRetain
    ctypes.POINTER(ctypes.c_int32),  # ZonesToRetain
]
lib.tecfil142.restype = ctypes.c_int32
lib.tecfil142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # OutputFileHandle
]
lib.tecforeign142.restype = ctypes.c_int32
lib.tecforeign142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # OutputForeignByteOrder
]

# Zone creation
lib.teczne142.restype = ctypes.c_int32
lib.teczne142.argtypes = [
    ctypes.c_char_p,  # ZoneTitle
    ctypes.POINTER(ctypes.c_int32),  # ZoneType
    ctypes.POINTER(ctypes.c_int32),  # IMx (or NumNodes for FE)
    ctypes.POINTER(ctypes.c_int32),  # JMx (or NumElements for FE)
    ctypes.POINTER(ctypes.c_int32),  # KMx
    ctypes.POINTER(ctypes.c_int32),  # ICellMax
    ctypes.POINTER(ctypes.c_int32),  # JCellMax
    ctypes.POINTER(ctypes.c_int32),  # KCellMax
    ctypes.POINTER(ctypes.c_double),  # SolutionTime
    ctypes.POINTER(ctypes.c_int32),  # StrandID
    ctypes.POINTER(ctypes.c_int32),  # ParentZone
    ctypes.POINTER(ctypes.c_int32),  # IsBlock (1=Block, 0=Point)
    ctypes.POINTER(ctypes.c_int32),  # NumFaceConnections
    ctypes.POINTER(ctypes.c_int32),  # FaceNeighborMode
    ctypes.POINTER(ctypes.c_int32),  # TotalNumFaceNodes (for poly zones)
    ctypes.POINTER(ctypes.c_int32),  # NumConnectedBoundaryFaces (for poly)
    ctypes.POINTER(ctypes.c_int32),  # TotalNumBoundaryConnections (for poly)
    ctypes.POINTER(ctypes.c_int32),  # PassiveVarList
    ctypes.POINTER(ctypes.c_int32),  # ValueLocation
    ctypes.POINTER(ctypes.c_int32),  # ShareVarFromZone
    ctypes.POINTER(ctypes.c_int32),  # ShareConnectivityFromZone
]
lib.tecpolyzne142.restype = ctypes.c_int32
lib.tecpolyzne142.argtypes = [
    ctypes.c_char_p,  # ZoneTitle
    ctypes.POINTER(ctypes.c_int32),  # ZoneType (FEPOLYGON or FEPOLYHEDRON)
    ctypes.POINTER(ctypes.c_int32),  # NumNodes
    ctypes.POINTER(ctypes.c_int32),  # NumElements
    ctypes.POINTER(ctypes.c_int32),  # NumFaces
    ctypes.POINTER(ctypes.c_int32),  # TotalNumFaceNodes
    ctypes.POINTER(ctypes.c_double),  # SolutionTime
    ctypes.POINTER(ctypes.c_int32),  # StrandID
    ctypes.POINTER(ctypes.c_int32),  # ParentZone
    ctypes.POINTER(ctypes.c_int32),  # NumConnectedBoundaryFaces
    ctypes.POINTER(ctypes.c_int32),  # TotalNumBoundaryConnections
    ctypes.POINTER(ctypes.c_int32),  # PassiveVarList
    ctypes.POINTER(ctypes.c_int32),  # ValueLocation
    ctypes.POINTER(ctypes.c_int32),  # ShareVarFromZone
    ctypes.POINTER(ctypes.c_int32),  # ShareConnectivityFromZone
]
lib.tecznefemixed142.restype = ctypes.c_int32
lib.tecznefemixed142.argtypes = [
    ctypes.c_char_p,  # ZoneTitle
    ctypes.POINTER(ctypes.c_int32),  # NumNodes
    ctypes.POINTER(ctypes.c_int32),  # NumElements
    ctypes.POINTER(ctypes.c_int32),  # NumNodesPerElement
    ctypes.POINTER(ctypes.c_double),  # SolutionTime
    ctypes.POINTER(ctypes.c_int32),  # StrandID
    ctypes.POINTER(ctypes.c_int32),  # ParentZone
    ctypes.POINTER(ctypes.c_int32),  # IsBlock
    ctypes.POINTER(ctypes.c_int32),  # NumFaceConnections
    ctypes.POINTER(ctypes.c_int32),  # FaceNeighborMode
    ctypes.POINTER(ctypes.c_int32),  # PassiveVarList
    ctypes.POINTER(ctypes.c_int32),  # ValueLocation
    ctypes.POINTER(ctypes.c_int32),  # ShareVarFromZone
    ctypes.POINTER(ctypes.c_int32),  # ShareConnectivityFromZone
]

# Partitioned zone creation
lib.tecijkptn142.restype = ctypes.c_int32
lib.tecijkptn142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # PartitionOwnerZone
    ctypes.POINTER(ctypes.c_int32),  # IMin
    ctypes.POINTER(ctypes.c_int32),  # JMin
    ctypes.POINTER(ctypes.c_int32),  # KMin
    ctypes.POINTER(ctypes.c_int32),  # IMax
    ctypes.POINTER(ctypes.c_int32),  # JMax
    ctypes.POINTER(ctypes.c_int32),  # KMax
]
lib.tecfeptn142.restype = ctypes.c_int32
lib.tecfeptn142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # PartitionOwnerZone
    ctypes.POINTER(ctypes.c_int32),  # NumNodes
    ctypes.POINTER(ctypes.c_int32),  # NumElements
]
lib.tecfemixedptn142.restype = ctypes.c_int32
lib.tecfemixedptn142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # PartitionOwnerZone
    ctypes.POINTER(ctypes.c_int32),  # NumNodes
    ctypes.POINTER(ctypes.c_int32),  # NumElements
    ctypes.POINTER(ctypes.c_int32),  # NumNodesPerElement
]

# Data writing
lib.tecdat142.restype = ctypes.c_int32
lib.tecdat142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # N (number of values)
    ctypes.c_void_p,  # FieldData (void pointer for flexibility)
    ctypes.POINTER(ctypes.c_int32),  # IsDouble (1=double, 0=float)
]

# Connectivity writing
lib.tecnod142.restype = ctypes.c_int32
lib.tecnod142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # NData (connectivity array)
]
lib.tecnode142.restype = ctypes.c_int32
lib.tecnode142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # N (number of values)
    ctypes.POINTER(ctypes.c_int32),  # NData (connectivity array)
]
lib.tecznemap142.restype = ctypes.c_int32
lib.tecznemap142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # N (number of values)
    ctypes.POINTER(ctypes.c_int32),  # NodeMap
]
lib.tecface142.restype = ctypes.c_int32
lib.tecface142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # FaceConnections
]
lib.tecpolyface142.restype = ctypes.c_int32
lib.tecpolyface142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # NumFaces
    ctypes.POINTER(ctypes.c_int32),  # FaceNodeCounts
    ctypes.POINTER(ctypes.c_int32),  # FaceNodes
    ctypes.POINTER(ctypes.c_int32),  # FaceLeftElems
    ctypes.POINTER(ctypes.c_int32),  # FaceRightElems
]
lib.tecpolybconn142.restype = ctypes.c_int32
lib.tecpolybconn142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # NumBoundaryFaces
    ctypes.POINTER(ctypes.c_int32),  # BoundaryConnectionCounts
    ctypes.POINTER(ctypes.c_int32),  # BoundaryConnectionElems
    ctypes.POINTER(ctypes.c_int16),  # BoundaryConnectionZones
]

# Auxiliary data
lib.tecauxstr142.restype = ctypes.c_int32
lib.tecauxstr142.argtypes = [
    ctypes.c_char_p,  # Name
    ctypes.c_char_p,  # Value
]

lib.tecvauxstr142.restype = ctypes.c_int32
lib.tecvauxstr142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # Var (1-based variable index)
    ctypes.c_char_p,  # Name
    ctypes.c_char_p,  # Value
]

lib.teczauxstr142.restype = ctypes.c_int32
lib.teczauxstr142.argtypes = [
    ctypes.c_char_p,  # Name
    ctypes.c_char_p,  # Value
]

# MPI initialization (for parallel I/O)
lib.tecmpiinit142.restype = ctypes.c_int32
lib.tecmpiinit142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # Communicator
    ctypes.POINTER(ctypes.c_int32),  # MainRank
]

# User-defined data (custom records)
lib.tecusr142.restype = ctypes.c_int32
lib.tecusr142.argtypes = [
    ctypes.c_char_p,  # UserRec
]


# ======================================================================================
# Wrappers for C functions
# - 1-to-1 python functions to format python inputs/ouputs to C TecIO functions
# - "New" API functions (tecXxxXxx) are wrapped by equivaltly named tec_xxx_xxx
#   python funcitons
# - "Classic" API functions (TECXXXX142) wrapped by equivaltly named tecxxx142 python
#   funcitons
# - Scope for this library is limited to IO for data related file records (geometry and
#   text records are ignored)
# - Some MPI funcion python wrappers are included, but not yet fully implemented
# - Wherever data arrays are output, conversion to numpy arrays is handled in the
#   wrapper function
# ======================================================================================

# --------------------------------------------------------------------------------------
# New read API (SZL/.szplt):
# - tec_file_reader_open returns an explicit file handle that is passed to every
#   subsequent call, making the target file unambiguous
# - Multiple files can be read simultaneously by holding multiple handles at once
# - Fucntions are available to query dataset/zone/variable metadata before reading any
#   data, and data can be read in any order by referencing the file handle +
#   zone/variable index
# - General organization of functions:
#   - tec_file_* functions operate on the file
#   - tec_data_set_* functions query whole dataset metadata
#   - tec_zone_* functions query zone level metadata (type, dimensions, title, solution
#     time, strand ID, node map, aux data)
#   - tec_var_* functions query variable-level metadata (name, aux data)
#   - tec_zone_var_* functions read variable values for a given zone and variable index
# --------------------------------------------------------------------------------------


# Reading SZL files
def tec_file_reader_open(file_name: str) -> ctypes.c_void_p:
    """Open an SZL file for reading.

    Args:
        file_name (str): Path to the ``.szplt`` file.

    Returns:
        Opaque file handle for subsequent TecIO calls.

    Raises:
        TecioError: If the file cannot be opened.
    """
    handle = ctypes.c_void_p(0)

    ret = lib.tecFileReaderOpen(
        ctypes.c_char_p(bytes(file_name, encoding="UTF-8")),
        ctypes.byref(handle),
    )
    if ret != 0:
        raise TecioError(
            f"SzlFile Initialization Error: file_name={file_name}, return_code={ret}"
        )

    return handle


def tec_file_reader_close(handle: ctypes.c_void_p) -> None:
    """Close an SZL file reader handle and release its resources.

    Args:
        handle: File handle from :func:`tec_file_reader_open`.

    Raises:
        TecioError: On C library error.
    """
    ret = lib.tecFileReaderClose(ctypes.byref(handle))
    if ret != 0:
        raise TecioError(
            f"tecFileReaderClose Error: handle={handle}, return_code={ret}"
        )


def tec_file_get_type(handle: ctypes.c_void_p) -> FileType:
    """Get the file type for an opened SZL file.

    Args:
        handle (ctypes.c_void_p): File handle from :func:`tec_file_reader_open`.

    Returns:
        :class:`FileType` enum (FULL, GRID, or SOLUTION).

    Raises:
        TecioError: On C library error.
    """
    file_type = ctypes.c_int32(0)

    ret = lib.tecFileGetType(handle, ctypes.byref(file_type))
    if ret != 0:
        raise TecioError(f"Error getting file type: handle:{handle}, return_code={ret}")

    return FileType(file_type.value)


def tec_data_set_get_title(handle: ctypes.c_void_p) -> str:
    """Read the dataset title string.

    Args:
        handle (ctypes.c_void_p): File handle.

    Returns:
        UTF-8 decoded dataset title.

    Raises:
        TecioError: On C library error.
    """
    title = ctypes.c_char_p(0)

    ret = lib.tecDataSetGetTitle(handle, ctypes.byref(title))
    if ret != 0:
        raise TecioError(
            f"Error getting data set title: handle={handle}, return_code={ret}"
        )

    return _decode(title.value)


def tec_data_set_get_num_vars(handle: ctypes.c_void_p) -> int:
    """Query the number of variables in the dataset.

    Args:
        handle (ctypes.c_void_p): File handle.

    Returns:
        Number of variables.

    Raises:
        TecioError: On C library error.
    """
    num_vars = ctypes.c_int32(0)

    ret = lib.tecDataSetGetNumVars(handle, ctypes.byref(num_vars))
    if ret != 0:
        raise TecioError(
            f"Error getting number of variables: handle={handle}, return_code={ret}"
        )

    return num_vars.value


def tec_data_set_get_num_zones(handle: ctypes.c_void_p) -> int:
    """Query the number of zones in the dataset.

    Args:
        handle (ctypes.c_void_p): File handle.

    Returns:
        Number of zones.

    Raises:
        TecioError: On C library error.
    """
    num_zones = ctypes.c_int32(0)

    ret = lib.tecDataSetGetNumZones(handle, ctypes.byref(num_zones))
    if ret != 0:
        raise TecioError(
            f"Error getting number of zones: handle={handle}, return_code={ret}"
        )

    return num_zones.value


# Reading SZL zones
def tec_zone_get_ijk(handle: ctypes.c_void_p, zone_index: int) -> tuple[int, int, int]:
    """Get zone dimensions (ORDERED) or node/element counts (FE).

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        ``(I, J, K)`` for ORDERED zones, or ``(num_nodes, num_elements, 0)``
        for FE zones.

    Raises:
        TecioError: On C library error.
    """
    imax = ctypes.c_int64(0)
    jmax = ctypes.c_int64(0)
    kmax = ctypes.c_int64(0)

    ret = lib.tecZoneGetIJK(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(imax),
        ctypes.byref(jmax),
        ctypes.byref(kmax),
    )
    if ret != 0:
        raise TecioError(
            f"Error getting zone data indices: : handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return imax.value, jmax.value, kmax.value


def tec_zone_get_title(handle: ctypes.c_void_p, zone_index: int) -> str:
    """Read the title for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Zone title string.

    Raises:
        TecioError: On C library error.
    """
    zone_title = ctypes.c_char_p(0)

    ret = lib.tecZoneGetTitle(
        handle, ctypes.c_int32(zone_index), ctypes.byref(zone_title)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetTitle Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return _decode(zone_title.value)


def tec_zone_get_type(handle: ctypes.c_void_p, zone_index: int) -> ZoneType:
    """Query the zone type.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        :class:`ZoneType` enum value.

    Raises:
        TecioError: On C library error.
    """
    zone_type = ctypes.c_int32(0)

    ret = lib.tecZoneGetType(
        handle, ctypes.c_int32(zone_index), ctypes.byref(zone_type)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetType Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return ZoneType(zone_type.value)


def tec_zone_is_enabled(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """Check whether a zone is enabled.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        True if the zone is enabled.

    Raises:
        TecioError: On C library error.
    """
    is_enabled = ctypes.c_int32(0)

    ret = lib.tecZoneIsEnabled(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is_enabled)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneIsEnabled Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return bool(is_enabled.value)


def tec_zone_get_solution_time(handle: ctypes.c_void_p, zone_index: int) -> float:
    """Read the solution time for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Solution time (double precision).

    Raises:
        TecioError: On C library error.
    """
    solution_time = ctypes.c_double(0)

    ret = lib.tecZoneGetSolutionTime(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(solution_time),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetSolutionTime Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return solution_time.value


def tec_zone_get_strand_id(handle: ctypes.c_void_p, zone_index: int) -> int:
    """Get the strand ID for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Strand ID integer.

    Raises:
        TecioError: On C library error.
    """
    strand_id = ctypes.c_int32(0)

    ret = lib.tecZoneGetStrandID(
        handle, ctypes.c_int32(zone_index), ctypes.byref(strand_id)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetStrandID Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return strand_id.value


def is_64bit(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """Check whether a zone's node-map uses 64-bit indices.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        True if 64-bit, False if 32-bit.

    Raises:
        TecioError: On C library error.
    """
    is64bit = ctypes.c_int32(0)
    ret = lib.tecZoneNodeMapIs64Bit(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is64bit)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapIs64Bit Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return bool(is64bit.value)


def tec_zone_node_map_get_64(
    handle: ctypes.c_void_p,
    zone_index: int,
    num_elements: int,
    nodes_per_cell: int,
) -> npt.NDArray[np.int64]:
    """Read a 64-bit node map for an FE zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        num_elements (int): Number of elements to read.
        nodes_per_cell (int): Nodes per element.

    Returns:
        Array of shape ``(num_elements, nodes_per_cell)`` with dtype int64.

    Raises:
        TecioError: On C library error.
    """
    size_of_array = num_elements * nodes_per_cell
    nodemap = (ctypes.c_int64 * size_of_array)()

    ret = lib.tecZoneNodeMapGet64(
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
            f"tecZoneNodeMapGet64 Error: handle={handle}, "
            f"zone_index={zone_index}, num_elements={num_elements}, "
            f"nodes_per_cell={nodes_per_cell}, return_code={ret}"
        )

    return np.ctypeslib.as_array(nodemap).reshape(num_elements, nodes_per_cell)


def tec_zone_node_map_get(
    handle: ctypes.c_void_p,
    zone_index: int,
    num_elements: int,
    nodes_per_cell: int,
) -> npt.NDArray[np.int32]:
    """Read a 32-bit node map for an FE zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        num_elements (int): Number of elements to read.
        nodes_per_cell (int): Nodes per element.

    Returns:
        Array of shape ``(num_elements, nodes_per_cell)`` with dtype int32.

    Raises:
        TecioError: On C library error.
    """
    size_of_array = num_elements * nodes_per_cell
    nodemap = (ctypes.c_int32 * size_of_array)()

    ret = lib.tecZoneNodeMapGet(
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
            f"tecZoneNodeMapGet Error: handle={handle}, "
            f"zone_index={zone_index}, num_elements={num_elements}, "
            f"nodes_per_cell={nodes_per_cell}, return_code={ret}"
        )

    return np.ctypeslib.as_array(nodemap).reshape(num_elements, nodes_per_cell)


# Reading SZL variable data
def tec_var_get_name(handle: ctypes.c_void_p, var_index: int) -> str:
    """Get a variable name by index.

    Args:
        handle (ctypes.c_void_p): File handle.
        var_index (int): 1-based variable index.

    Returns:
        Variable name string.

    Raises:
        TecioError: On C library error.
    """
    var_name = ctypes.c_char_p(0)

    ret = lib.tecVarGetName(handle, ctypes.c_int32(var_index), ctypes.byref(var_name))
    if ret != 0:
        raise TecioError(
            f"tecVarGetName Error: handle={handle}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return _decode(var_name.value)


def tec_var_is_enabled(handle: ctypes.c_void_p, var_index: int) -> bool:
    """Check whether a variable is enabled.

    Args:
        handle (ctypes.c_void_p): File handle.
        var_index (int): 1-based variable index.

    Returns:
        True if the variable is enabled.

    Raises:
        TecioError: On C library error.
    """
    is_enabled = ctypes.c_int32(0)

    ret = lib.tecVarIsEnabled(
        handle, ctypes.c_int32(var_index), ctypes.byref(is_enabled)
    )
    if ret != 0:
        raise TecioError(
            f"tecVarIsEnabled Error: handle={handle}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return bool(is_enabled.value)


def tec_zone_var_get_type(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> DataType:
    """Get the data type for a variable in a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        :class:`DataType` enum value.

    Raises:
        TecioError: On C library error.
    """
    var_type = ctypes.c_int32(0)

    ret = lib.tecZoneVarGetType(
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
    """Get the value location for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        :class:`ValueLocation` enum (NODAL or CELL_CENTERED).

    Raises:
        TecioError: On C library error.
    """
    value_location = ctypes.c_int32(0)

    ret = lib.tecZoneVarGetValueLocation(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(value_location),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetValueLocation Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, return_code={ret}"
        )

    return ValueLocation(value_location.value)


def tec_zone_var_is_passive(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> bool:
    """Check whether a zone variable is passive.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        True if the variable is passive in this zone.

    Raises:
        TecioError: On C library error.
    """
    is_passive = ctypes.c_int32(0)

    ret = lib.tecZoneVarIsPassive(
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
) -> int | None:
    """Get the shared zone index for a variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        Shared zone index, or None if not shared.

    Raises:
        TecioError: On C library error.
    """
    shared_zone = ctypes.c_int32(0)

    ret = lib.tecZoneVarGetSharedZone(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(shared_zone),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetSharedZone Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, return_code={ret}"
        )

    return shared_zone.value if shared_zone.value != 0 else None


def tec_zone_connectivity_get_shared_zone(
    handle: ctypes.c_void_p, zone_index: int
) -> int | None:
    """Get the shared zone index for unstructured mesh connectivity data.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Shared zone index, or None if not shared.

    Raises:
        TecioError: On C library error.
    """
    shared_zone = ctypes.c_int32(0)

    ret = lib.tecZoneConnectivityGetSharedZone(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(shared_zone),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneConnectivityGetSharedZone Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return shared_zone.value if shared_zone.value != 0 else None


def tec_zone_var_get_num_values(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> int:
    """Query the number of values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        Number of values.

    Raises:
        TecioError: On C library error.
    """
    num_values = ctypes.c_int32(0)

    ret = lib.tecZoneVarGetNumValues(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(num_values),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetNumValues Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, return_code={ret}"
        )

    return num_values.value


def tec_zone_var_get_float_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.float32]:
    """Read float32 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy float32 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    values = (ctypes.c_float * num_values)()

    ret = lib.tecZoneVarGetFloatValues(
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
            f"tecZoneVarGetFloatValues Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_double_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.float64]:
    """Read float64 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy float64 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    values = (ctypes.c_double * num_values)()

    ret = lib.tecZoneVarGetDoubleValues(
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
            f"tecZoneVarGetDoubleValues Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_int32_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.int32]:
    """Read int32 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy int32 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    values = (ctypes.c_int32 * num_values)()

    ret = lib.tecZoneVarGetInt32Values(
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
            f"tecZoneVarGetInt32Values Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_int16_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.int16]:
    """Read int16 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy int16 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    values = (ctypes.c_int16 * num_values)()

    ret = lib.tecZoneVarGetInt16Values(
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
            f"tecZoneVarGetInt16Values Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


def tec_zone_var_get_uint8_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.uint8]:
    """Read uint8 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy uint8 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    values = (ctypes.c_uint8 * num_values)()

    ret = lib.tecZoneVarGetUInt8Values(
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
            f"tecZoneVarGetUInt8Values Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


# Reading SZL aux data
def tec_data_set_aux_data_get_num_items(handle: ctypes.c_void_p) -> int:
    """Get the number of dataset-level auxiliary data items.

    Args:
        handle (ctypes.c_void_p): File handle.

    Returns:
        Number of auxiliary data items.

    Raises:
        TecioError: On C library error.
    """
    num_auxdata_items = ctypes.c_int32(0)

    ret = lib.tecDataSetAuxDataGetNumItems(handle, ctypes.byref(num_auxdata_items))
    if ret != 0:
        raise TecioError(
            f"tecDataSetAuxDataGetNumItems Error: handle={handle}, return_code={ret}"
        )

    return num_auxdata_items.value


def tec_data_set_aux_data_get_item(
    handle: ctypes.c_void_p, item_index: int
) -> tuple[str, str]:
    """Read a dataset-level auxiliary data item.

    Args:
        handle (ctypes.c_void_p): File handle.
        item_index (int): 1-based item index.

    Returns:
        ``(name, value)`` string tuple.

    Raises:
        TecioError: On C library error.
    """
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = lib.tecDataSetAuxDataGetItem(
        handle,
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecDataSetAuxDataGetItem Error: handle={handle}, "
            f"item_index={item_index}, return_code={ret}"
        )

    return _decode(name.value), _decode(value.value)


def tec_var_aux_data_get_num_items(handle: ctypes.c_void_p, var_index: int) -> int:
    """Get the number of auxiliary data items for a variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        var_index (int): 1-based variable index.

    Returns:
        Number of auxiliary data items.

    Raises:
        TecioError: On C library error.
    """
    num_items = ctypes.c_int32(0)

    ret = lib.tecVarAuxDataGetNumItems(
        handle, ctypes.c_int32(var_index), ctypes.byref(num_items)
    )
    if ret != 0:
        raise TecioError(
            f"tecVarAuxDataGetNumItems Error: handle={handle}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return num_items.value


def tec_var_aux_data_get_item(
    handle: ctypes.c_void_p, var_index: int, item_index: int
) -> tuple[str, str]:
    """Read a variable-level auxiliary data item.

    Args:
        handle (ctypes.c_void_p): File handle.
        var_index (int): 1-based variable index.
        item_index (int): 1-based item index.

    Returns:
        ``(name, value)`` string tuple.

    Raises:
        TecioError: On C library error.
    """
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = lib.tecVarAuxDataGetItem(
        handle,
        ctypes.c_int32(var_index),
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecVarAuxDataGetItem Error: handle={handle}, var_index={var_index}, "
            f"item_index={item_index}, return_code={ret}"
        )

    return _decode(name.value), _decode(value.value)


def tec_zone_aux_data_get_num_items(handle: ctypes.c_void_p, zone_index: int) -> int:
    """Get the number of auxiliary data items for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Number of auxiliary data items.

    Raises:
        TecioError: On C library error.
    """
    num_items = ctypes.c_int32(0)

    ret = lib.tecZoneAuxDataGetNumItems(
        handle, ctypes.c_int32(zone_index), ctypes.byref(num_items)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneAuxDataGetNumItems Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return num_items.value


def tec_zone_aux_data_get_item(
    handle: ctypes.c_void_p, zone_index: int, item_index: int
) -> tuple[str, str]:
    """Read a zone-level auxiliary data item.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        item_index (int): 1-based item index.

    Returns:
        ``(name, value)`` string tuple.

    Raises:
        TecioError: On C library error.
    """
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = lib.tecZoneAuxDataGetItem(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneAuxDataGetItem Error: handle={handle}, zone_index={zone_index}, "
            f"item_index={item_index}, return_code={ret}"
        )

    return _decode(name.value), _decode(value.value)


# --------------------------------------------------------------------------------------
# New write API (SZL/.szplt):
# - tec_file_writer_open returns an explicit file handle that is passed to every
#   subsequent call, making the target file unambiguous
# - Multiple files can be written simultaneously by holding multiple handles at once
#   tec_zone_create_ijk / tec_zone_create_fe append a new zone record and return a
#   1-based zone index
# - Variable data, node maps, aux data, and unsteady options are written by referencing
#   the file handle + zone/variable index, so they can be written in any order after
#   zone creation
# - tec_file_writer_close finalizes and flushes the file
# --------------------------------------------------------------------------------------


# Initialization and File Handling
def tec_file_writer_open(
    filename: str,
    variables: Sequence[str],
    title: str = "Untitled",
    file_type: FileType = FileType.FULL,
    use_szl: int = 1,
    grid_file_handle: ctypes.c_void_p | None = None,
) -> ctypes.c_void_p:
    """Open a writer handle for creating SZL files.

    Args:
        filename (str): Output file path.
        variables (Sequence[str]): Variable name list.
        title (str): Dataset title.
        file_type (FileType): File type enum.
        use_szl (int): SZL flag (1 to use SZL format).
        grid_file_handle (ctypes.c_void_p | None): Optional handle to an
            existing grid file for solution-only output.

    Returns:
        Opaque writer handle for subsequent write calls.

    Raises:
        TecioError: On C library error.
    """
    if not isinstance(file_type, FileType):
        raise TypeError("file_type must be a libtecio.FileType enum")

    assert all(len(v) <= 128 for v in variables), "Variables limited to 128 characters"

    varstring = ", ".join(variables)
    handle = ctypes.c_void_p()

    ret = lib.tecFileWriterOpen(
        ctypes.c_char_p(filename.encode("utf-8")),
        ctypes.c_char_p(title.encode("utf-8")),
        ctypes.c_char_p(varstring.encode("utf-8")),
        ctypes.c_int32(use_szl),
        ctypes.c_int32(file_type.value),
        ctypes.c_int32(0),
        grid_file_handle if grid_file_handle is not None else None,
        ctypes.byref(handle),
    )
    if ret != 0:
        raise TecioError(
            f"tecFileWriterOpen Error: file_name={filename!r}, title={title!r}, "
            f"variables={variables!r}, file_type={file_type!r}, return_code={ret}"
        )
    return handle


def tec_file_writer_flush(
    handle: ctypes.c_void_p,
    num_zones_to_retain: int = 0,
    zones_to_retain: Sequence[int] | None = None,
) -> None:
    """Flush written zone data to a temporary intermediate file.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        num_zones_to_retain (int): Number of zones to keep in memory.
        zones_to_retain (Sequence[int] | None): 1-based zone indices to retain.

    Raises:
        TecioError: On C library error.

    Important:
        SZL Only!

    Note:
        Used to reduce memory usage for large files. All zone data written so far, other
        than any zones listed in ``zones_to_retain``, is written out to a temporary file
        on disk and released from memory.

    Note:
        Retained zones can still be modified.

    Note:
        Temporary files created by flushing are merged into the final output file when
        :func:`tec_file_writer_close` is called.
    """
    zones_ptr = None
    if zones_to_retain is not None and len(zones_to_retain) > 0:
        zones_array = (ctypes.c_int32 * len(zones_to_retain))(*zones_to_retain)
        zones_ptr = ctypes.cast(zones_array, ctypes.POINTER(ctypes.c_int32))
    else:
        zones_ptr = ctypes.POINTER(ctypes.c_int32)()

    ret = lib.tecFileWriterFlush(
        handle,
        ctypes.c_int32(num_zones_to_retain),
        zones_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecFileWriterFlush Error: handle={handle}, "
            f"num_zones_to_retain={num_zones_to_retain}, "
            f"zones_to_retain={zones_to_retain}, "
            f"return_code={ret}"
        )


def tec_file_writer_close(handle: ctypes.c_void_p) -> None:
    """Close a writer handle and finalise the output file.

    Args:
        handle (ctypes.c_void_p): Writer handle.

    Raises:
        TecioError: On C library error.
    """
    ret = lib.tecFileWriterClose(ctypes.byref(handle))
    if ret != 0:
        raise TecioError(
            f"tecFileWriterClose Error: handle={handle}, return_code={ret}"
        )


# Write Zone Headers
def tec_zone_create_ijk(
    handle: ctypes.c_void_p,
    zone_title: str,
    imax: int,
    jmax: int,
    kmax: int,
    var_types: Sequence[DataType | int],
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[VarStatus | bool | int] | None = None,
    face_nbr_sharing: int = 0,
    num_face_cons: int = 0,
    face_nbr_mode: FaceNeighborMode | int = FaceNeighborMode.LOCAL_ONE_TO_ONE,
) -> int:
    """Create an ordered IJK zone for writing.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone_title (str): Zone title.
        imax (int): I dimension.
        jmax (int): J dimension.
        kmax (int): K dimension.
        var_types (Sequence[DataType | int]): Per-variable data types.
        var_sharing (Sequence[int] | None): Optional per-variable sharing source
            zones. Must be same length as var_types if provided. None/null means all no
            variable sharing.
        value_locations (Sequence[int | ValueLocation] | None): Optional per-variable
            data locations.
        pas_vars (Sequence[bool | int] | None): Optional per-variable passive flags or
            1/True if passive else 0/False. Must be same length as var_types if
            provided. None/null means all variables are active.
        face_nbr_sharing (int): Optional face-neighbor sharing source zone.
        num_face_cons (int): Optional number of face connections.
        face_nbr_mode (FaceNeighborMode | int): Optional face-neighbor mode.

    Returns:
        1-based zone index of the created zone.

    Raises:
        TecioError: On C library error.
    """
    zone_out = ctypes.c_int32()

    # Create C array for varable types
    var_types_ptr = (ctypes.c_int32 * len(var_types))(*[
        _to_int_value(v, DataType) for v in var_types
    ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    ret = lib.tecZoneCreateIJK(
        handle,
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int64(imax),
        ctypes.c_int64(jmax),
        ctypes.c_int64(kmax),
        var_types_ptr,
        var_sharing_ptr,
        value_locations_ptr,
        pas_vars_ptr,
        ctypes.c_int32(face_nbr_sharing),  # face neighbor sharing source zone
        ctypes.c_int64(num_face_cons),  # number of face connections
        ctypes.c_int32(
            _to_int_value(face_nbr_mode, FaceNeighborMode)
        ),  # face neighbor mode
        ctypes.byref(zone_out),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneCreateIJK Error: zone_title={zone_title!r}, "
            f"imax={imax}, jmax={jmax}, kmax={kmax}, "
            f"var_types_len={len(var_types) if var_types is not None else 0}, "
            f"return_code={ret}"
        )
    return zone_out.value


def tec_zone_create_fe(
    handle: ctypes.c_void_p,
    zone_title: str,
    zone_type: int | ZoneType,
    num_nodes: int,
    num_cells: int,
    var_types: Sequence[DataType | int],
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[bool | int] | None = None,
    con_sharing: int = 0,
    num_face_cons: int = 0,
    face_nbr_mode: FaceNeighborMode | int = FaceNeighborMode.LOCAL_ONE_TO_ONE,
) -> int:
    """Create a finite-element zone for writing.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone_title (str): Zone title.
        zone_type (int | ZoneType): FE zone type.
        num_nodes (int): Number of nodes.
        num_cells (int): Number of cells/elements.
        var_types (Sequence[DataType | int]): Optional per-variable data types.
        var_sharing (Sequence[int] | None): Optional per-variable sharing source
            zones. Must be same length as var_types if provided. None/null means all no
            variable sharing.
        value_locations (Sequence[int | ValueLocation] | None): Optional per-variable
            data locations.
        pas_vars (Sequence[bool | int] | None): Optional per-variable passive flags or
            1/True if passive else 0/False. Must be same length as var_types if
            provided. None/null means all variables are active.
        con_sharing (int): Optional connectivity sharing source zone (0 =
            none). Connectivity and/or face neighbors cannot be shared when the face
            neighbor mode is set to Global. Connectivity cannot be shared between
            cell-based and face-based finite element zones.
        num_face_cons (int): Optional number of face connections.
        face_nbr_mode (FaceNeighborMode | int): Optional face-neighbor mode.

    Returns:
        1-based zone index of the created zone.

    Raises:
        TecioError: On C library error.
    """
    zone_out = ctypes.c_int32()

    # Create C array for varable types
    var_types_ptr = (ctypes.c_int32 * len(var_types))(*[
        _to_int_value(v, DataType) for v in var_types
    ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    ret = lib.tecZoneCreateFE(
        handle,
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int32(_to_int_value(zone_type)),
        ctypes.c_int64(num_nodes),
        ctypes.c_int64(num_cells),
        var_types_ptr,
        var_sharing_ptr,
        value_locations_ptr,
        pas_vars_ptr,
        ctypes.c_int32(con_sharing),  # connectivity sharing
        ctypes.c_int64(num_face_cons),  # num face connections
        ctypes.c_int32(
            _to_int_value(face_nbr_mode, FaceNeighborMode)
        ),  # face neighbor mode
        ctypes.byref(zone_out),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneCreateIJK Error: zone_title={zone_title!r}, "
            f"ZoneType={zone_type!r}, NODES={num_nodes}, CELLS={num_cells}, "
            f"var_types_len={len(var_types) if var_types is not None else 0}, "
            f"return_code={ret}"
        )
    return zone_out.value


# Optional fields
def tec_zone_set_unsteady_options(
    handle: ctypes.c_void_p, zone: int, strand: int = 0, solution_time: float = 0.0
) -> None:
    """Set time/strand metadata for a zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        strand (int): Strand ID.
        solution_time (float): Solution time.

    Raises:
        TecioError: On C library error.
    """
    ret = lib.tecZoneSetUnsteadyOptions(
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


def tec_data_set_add_aux_data(
    handle: ctypes.c_void_p,
    name: str,
    value: str,
) -> None:
    """Add a dataset-level auxiliary data record.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.
    """
    ret = lib.tecDataSetAddAuxData(
        handle,
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecDataSetAddAuxData Error: name={name!r}, value={value!r}, "
            f"return_code={ret}"
        )


def tec_var_add_aux_data(
    handle: ctypes.c_void_p,
    var_index: int,
    name: str,
    value: str,
) -> None:
    """Add a variable-level auxiliary data record.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        var_index (int): 1-based variable index.
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.
    """
    ret = lib.tecVarAddAuxData(
        handle,
        ctypes.c_int32(var_index),
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecVarAddAuxData Error: var_index={var_index}, name={name!r}, "
            f"value={value!r}, return_code={ret}"
        )


def tec_zone_add_aux_data(
    handle: ctypes.c_void_p,
    zone_index: int,
    name: str,
    value: str,
) -> None:
    """Add a zone-level auxiliary data record.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone_index (int): 1-based zone index.
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.
    """
    ret = lib.tecZoneAddAuxData(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneAddAuxData Error: zone_index={zone_index}, name={name!r}, "
            f"value={value!r}, return_code={ret}"
        )


# ---- Write variable value functions --------------------------------
def tec_zone_var_write_double_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """Write float64 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of float64 values.

    Raises:
        TecioError: On C library error.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(
        values, np.float64, ctypes.c_double
    )

    ret = lib.tecZoneVarWriteDoubleValues(
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
    """Write float32 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of float32 values.

    Raises:
        TecioError: On C library error.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.float32, ctypes.c_float)

    ret = lib.tecZoneVarWriteFloatValues(
        handle,
        ctypes.c_int32(zone),  # zone index (1-based)
        ctypes.c_int32(var),  # variable index (1-based)
        ctypes.c_int32(0),  # partition index (MPI)
        ctypes.c_int64(count),  # number of values
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
    """Write int32 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of int32 values.

    Raises:
        TecioError: On C library error.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.int32, ctypes.c_int32)

    ret = lib.tecZoneVarWriteInt32Values(
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
    """Write int16 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of int16 values.

    Raises:
        TecioError: On C library error.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.int16, ctypes.c_int16)

    ret = lib.tecZoneVarWriteInt16Values(
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
    """Write uint8 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of uint8 values.

    Raises:
        TecioError: On C library error.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.uint8, ctypes.c_uint8)

    ret = lib.tecZoneVarWriteUInt8Values(
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


# Write Zone Connectivity (FE zones only)
def tec_zone_node_map_write32(
    handle: ctypes.c_void_p,
    zone: int,
    nodes: npt.ArrayLike,
    partition: int = 0,
    is_one_based: bool | int = True,
) -> None:
    """Write 32-bit node map entries for an FE zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        nodes (npt.ArrayLike): Array of int32 node indices.
        partition (int): Partition index (0 for non-partitioned).
        is_one_based (bool | int): Whether indices are 1-based.

    Raises:
        TecioError: On C library error.

    Note:
        Use for zones with fewer than ~2 billion node map entries.

    Note:
        Can be called multiple times per zone; total entries must match zone
        definition.

    Note:
        Node indices are 1-based by convention in Tecplot.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(nodes, np.int32, ctypes.c_int32)

    ret = lib.tecZoneNodeMapWrite32(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(partition),
        ctypes.c_int32(_to_int_value(is_one_based, Boolean)),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapWrite32 Error: zone={zone}, partition={partition}, "
            f"count={count}, return_code={ret}"
        )


def tec_zone_node_map_write64(
    handle: ctypes.c_void_p,
    zone: int,
    nodes: npt.ArrayLike,
    partition: int = 0,
    is_one_based: bool | int = True,
) -> None:
    """Write 64-bit node map entries for an FE zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        nodes (npt.ArrayLike): Array of int64 node indices.
        partition (int): Partition index (0 for non-partitioned).
        is_one_based (bool | int): Whether indices are 1-based.

    Raises:
        TecioError: On C library error.

    Note:
        Use for zones with fewer than ~2 billion node map entries.

    Note:
        Can be called multiple times per zone; total entries must match zone
        definition.

    Note:
        Node indices are 1-based by convention in Tecplot.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(nodes, np.int64, ctypes.c_int64)

    ret = lib.tecZoneNodeMapWrite64(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(partition),
        ctypes.c_int32(_to_int_value(is_one_based, Boolean)),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapWrite64 Error: zone={zone}, partition={partition}, "
            f"count={count}, return_code={ret}"
        )


def tec_zone_face_nbr_write_connections32(
    handle: ctypes.c_void_p,
    zone: int,
    face_neighbors: npt.ArrayLike,
) -> None:
    """Write 32-bit face-neighbor connections for an FE zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        face_neighbors (npt.ArrayLike): Array of int32 face-neighbor indices.

    Raises:
        TecioError: On C library error.

    Note:
        ``num_face_cons`` and ``face_nbr_mode`` must be set when creating the zone via
        :func:`tec_zone_create_ijk` or :func:`tec_zone_create_fe`.

    Note:
        Use for zones with fewer than ~2 billion face neighbor entries.

    Note:
        Can be called multiple times. Total entries must match num_face_cons
        declared at zone creation.

    Note:
        Face neighbors have expensive performance implications. Use face neighbors
        only to manually specify connections that are not defined via the connectivity
        list.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(
        face_neighbors, np.int32, ctypes.c_int32
    )
    ret = lib.tecZoneFaceNbrWriteConnections32(
        handle,
        ctypes.c_int32(zone),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrWriteConnections32 Error: zone={zone}, "
            f"count={count}, return_code={ret}"
        )


def tec_zone_face_nbr_write_connections64(
    handle: ctypes.c_void_p,
    zone: int,
    face_neighbors: npt.ArrayLike,
) -> None:
    """Write 64-bit face-neighbor connections for an FE zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        face_neighbors (npt.ArrayLike): Array of int64 face-neighbor indices.

    Raises:
        TecioError: On C library error.

    Note:
        ``num_face_cons`` and ``face_nbr_mode`` must be set when creating the zone via
        :func:`tec_zone_create_ijk` or :func:`tec_zone_create_fe`.

    Note:
        Use for zones with more than ~2 billion face neighbor entries.

    Note:
        Can be called multiple times. Total entries must match num_face_cons
        declared at zone creation.

    Note:
        Face neighbors have expensive performance implications. Use face neighbors
        only to manually specify connections that are not defined via the connectivity
        list.
    """
    ptr, count, _backing = _prepare_array_for_ctypes(
        face_neighbors, np.int64, ctypes.c_int64
    )
    ret = lib.tecZoneFaceNbrWriteConnections64(
        handle,
        ctypes.c_int32(zone),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrWriteConnections64 Error: zone={zone}, "
            f"count={count}, return_code={ret}"
        )


# --------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------


# File initialization and finalization
def tecini142(
    filename: str,
    variables: Sequence[str],
    title: str = "Untitled",
    scratch_dir: str = ".",
    file_format: int | FileFormat = FileFormat.PLT,
    file_type: int | FileType = FileType.FULL,
    debug: int | Debug = Debug.FALSE,
    vis_double: int | DataType = DataType.DOUBLE,
) -> None:
    """Initialise a Tecplot data file (classic API).

    Args:
        filename (str): Output file path. Supports PLT and SZPLT extensions.
        variables (Sequence[str]): Variable name list.
        title (str): Dataset title.
        scratch_dir (str): Scratch directory for temporary files.
        file_format (int | FileFormat): FileFormat.PLT (0) or FileFormat.SZPLT (1)
        file_type (int | FileType): FileType.FULL (0), GRID (1), or SOLUTION (2)
        debug (int | Debug): Debug.FALSE (0) or Debug.TRUE (1)
        vis_double (int | DataType): DataType.DOUBLE (1) or DataType.FLOAT (0). PLT
            files do not suport integer zone variables. Use the new API functions to
            create integer zone variables for SZPLT files.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called before any zone or data operations

    Note:
        Call :func:`tecend142()` to finalize the file
    """
    if not isinstance(file_type, FileType):
        raise TypeError("file_type must be a libtecio.FileType enum")

    assert all(len(v) <= 128 for v in variables), "Variables limited to 128 characters"

    varstring = ", ".join(variables)
    vis_double_c = ctypes.c_int32(
        1 if _to_int_value(vis_double) == DataType.DOUBLE.value else 0
    )
    ret = lib.tecini142(
        ctypes.c_char_p(title.encode("utf-8")),
        ctypes.c_char_p(varstring.encode("utf-8")),
        ctypes.c_char_p(filename.encode("utf-8")),
        ctypes.c_char_p(scratch_dir.encode("utf-8")),
        ctypes.c_int32(_to_int_value(file_format)),
        ctypes.c_int32(_to_int_value(file_type)),
        ctypes.c_int32(_to_int_value(debug)),
        vis_double_c,
    )
    if ret != 0:
        raise TecioError(f"tecini142 Error: filename={filename!r}, return_code={ret}")


def tecend142() -> None:
    """Finalise and close the active Tecplot data file.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called after all data has been written

    Note:
        Flushes all pending data and closes the file
    """
    ret = lib.tecend142()
    if ret != 0:
        raise TecioError(f"tecend142 Error: return_code={ret}")


def tecflush142(
    num_zones_to_retain: int = 0,
    zones_to_retain: Sequence[int] | None = None,
) -> None:
    """Flush data to disk, optionally retaining zones in memory (SZL only).

    Args:
        num_zones_to_retain (int): Number of zones to keep in memory.
        zones_to_retain (Sequence[int] | None): 1-based zone indices to retain.

    Raises:
        TecioError: On C library error.

    Important:
        SZL Only!

    Note:
        Used to reduce memory usage for large files.

    Note:
        Retained zones can still be modified.
    """
    num_zones_c = ctypes.c_int32(num_zones_to_retain)

    zones_ptr = None
    if zones_to_retain is not None and len(zones_to_retain) > 0:
        zones_array = (ctypes.c_int32 * len(zones_to_retain))(*zones_to_retain)
        zones_ptr = ctypes.cast(zones_array, ctypes.POINTER(ctypes.c_int32))
    else:
        zones_ptr = ctypes.POINTER(ctypes.c_int32)()

    ret = lib.tecflush142(
        ctypes.byref(num_zones_c),
        zones_ptr,
    )
    if ret != 0:
        raise TecioError(f"tecflush142 Error: return_code={ret}")


def tecfil142() -> int:
    """Get the file handle for the current output file.

    Returns:
        File handle integer.

    Raises:
        TecioError: On C library error.

    Note:
        Used for advanced file operations.
    """
    output_file_handle = ctypes.c_int32(0)

    ret = lib.tecfil142(ctypes.byref(output_file_handle))
    if ret != 0:
        raise TecioError(f"tecfil142 Error: return_code={ret}")

    return output_file_handle.value


def tecforeign142(output_foreign_byte_order: int) -> None:
    """Set foreign byte order for output.

    Args:
        output_foreign_byte_order (int): 0 = native, 1 = foreign.

    Raises:
        TecioError: On C library error.
    """
    foreign_c = ctypes.c_int32(output_foreign_byte_order)

    ret = lib.tecforeign142(ctypes.byref(foreign_c))
    if ret != 0:
        raise TecioError(f"tecforeign142 Error: return_code={ret}")


# Zone creation
def teczne142(
    zone_title: str,
    zone_type: int | ZoneType,
    imax: int,
    jmax: int,
    kmax: int,
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[int | VarStatus] | None = None,
    con_sharing: int = 0,
    strand: int = 0,
    solution_time: float = 0.0,
    num_face_connections: int = 0,
    face_nbr_mode: int | FaceNeighborMode = FaceNeighborMode.LOCAL_ONE_TO_ONE,
    total_num_face_nodes: int = 0,
    num_connected_boundary_faces: int = 0,
    total_num_boundary_connections: int = 0,
) -> None:
    """Create a new zone in the active file (classic API).

    Args:
        zone_title (str): Zone title.
        zone_type (int | ZoneType): Zone type enum or integer (0=ORDERED, 1=FELINESEG,
            etc.).
        imax (int): I dimension (or num nodes for FE).
        jmax (int): J dimension (or num elements for FE).
        kmax (int): K dimension or NumFaces for FEPOLYGON and FEPOLYHEDRON. Not used for
            all other finite element zone types.
        var_sharing (Sequence[int] | None): List of zone indices to share variables from
            (0: No sharing, Null: no variables are shared from other zones)
        value_locations (Sequence[int | ValueLocation] | None): Per-variable
            data locations.
        pas_vars (Sequence[int | VarStatus] | None): List of VarStatus enums or 0/1 for
            passive variables. Must be same length as var_sharing if provided. None/null
            means all variables are active.
        con_sharing (int): Connectivity sharing source zone number.
        strand (int): Strand ID for transient data (0 for static data, positive integer
            for transient data).
        solution_time (float): Solution time.
        num_face_connections (int): Number of face connections (for cell-based FE zones
            only. The number of face connections that will be passed to ``tecface142``).
        face_nbr_mode (int | FaceNeighborMode): FaceNeighborMode enum. Used for
            cell-based FE and ordered zones only. Type of face connection that will be
            passed in routine ``tecface142``.
        total_num_face_nodes (int): Total face nodes (poly zones).
        num_connected_boundary_faces (int): Boundary faces (poly zones).
        total_num_boundary_connections (int): Boundary connections (poly zones).

    Raises:
        TecioError: On C library error.

    Hint:
        For ORDERED zones: imx, jmx, kmx are dimensions.

    Note:
        For FE zones: imax=NumNodes, jmax=NumElements, kmax=0 (unless higher order
        element)
    """
    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    ret = lib.teczne142(
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int32(_to_int_value(zone_type)),
        ctypes.c_int32(imax),
        ctypes.c_int32(jmax),
        ctypes.c_int32(kmax),
        ctypes.c_int32(0),  # I-cell dimension (Reserved for future use. Set to zero)
        ctypes.c_int32(0),  # J-cell dimension (Reserved for future use. Set to zero)
        ctypes.c_int32(0),  # K-cell dimension (Reserved for future use. Set to zero)
        ctypes.c_double(solution_time),
        ctypes.c_int32(strand),
        ctypes.c_int32(0),  # Deprecated. Enter 0 for this value
        ctypes.c_int32(_to_int_value(DataPacking.BLOCK)),  # Deprecated. Always set to 1
        ctypes.c_int32(num_face_connections),
        ctypes.c_int32(_to_int_value(face_nbr_mode)),
        ctypes.c_int32(total_num_face_nodes),
        ctypes.c_int32(num_connected_boundary_faces),
        ctypes.c_int32(total_num_boundary_connections),
        pas_vars_ptr,
        value_locations_ptr,
        var_sharing_ptr,
        ctypes.c_int32(con_sharing),
    )
    if ret != 0:
        raise TecioError(
            f"teczne142 Error: zone_title={zone_title!r}, return_code={ret}"
        )


def tecpolyzne142(
    zone_title: str,
    zone_type: int | ZoneType,
    num_nodes: int,
    num_faces: int,
    num_elements: int,
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[int | VarStatus] | None = None,
    con_sharing: int = 0,
    strand: int = 0,
    solution_time: float = 0.0,
    total_num_face_nodes: int = 0,
    num_connected_boundary_faces: int = 0,
    total_num_boundary_connections: int = 0,
) -> None:
    """Create a polygonal or polyhedral zone (classic API).

    Args:
        zone_title (str): Zone title.
        zone_type (int | ZoneType): ``ZoneType.FEPOLYGON`` or ``ZoneType.FEPOLYHEDRON``.
        num_nodes (int): Number of nodes.
        num_faces (int): Number of faces.
        num_elements (int): Number of elements.
        var_sharing (Sequence[int] | None): List of zone indices to share variables from
            (0: No sharing, Null: no variables are shared from other zones). Must be
            same length as value_locations and pas_vars if provided.
        value_locations (Sequence[int | ValueLocation] | None): Per-variable
            data locations.
        pas_vars (Sequence[int | VarStatus] | None): ist of VarStatus enums or 0/1 for
            passive variables. Must be same length as value_locations and var_sharing if
            provided. None/null means all variables are active.
        con_sharing (int): Connectivity sharing source zone.
        strand (int): Strand ID for transient data (0 for static data, positive integer
            for transient data).
        solution_time (float): Solution time for transient data.
        total_num_face_nodes (int): Total face nodes.
        num_connected_boundary_faces (int): Boundary faces.
        total_num_boundary_connections (int): Boundary connections.

    Raises:
        TecioError: On C library error.
    """
    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    ret = lib.tecpolyzne142(
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int32(_to_int_value(zone_type)),
        ctypes.c_int32(num_nodes),
        ctypes.c_int32(num_elements),
        ctypes.c_int32(num_faces),
        ctypes.c_int32(total_num_face_nodes),
        ctypes.c_double(solution_time),
        ctypes.c_int32(strand),
        ctypes.c_int32(0),  # Parent zone deprecated.
        ctypes.c_int32(num_connected_boundary_faces),
        ctypes.c_int32(total_num_boundary_connections),
        pas_vars_ptr,
        value_locations_ptr,
        var_sharing_ptr,
        ctypes.c_int32(con_sharing),
    )
    if ret != 0:
        raise TecioError(
            f"tecpolyzne142 Error: zone_title={zone_title!r}, return_code={ret}"
        )


def tecznefemixed142(
    zone_title: str,
    num_nodes: int,
    num_elements: int,
    num_nodes_per_element: int,
    solution_time: float = 0.0,
    strand_id: int = 0,
    parent_zone: int = 0,
    datapacking: int | DataPacking = DataPacking.BLOCK,
    num_face_connections: int = 0,
    face_neighbor_mode: int | FaceNeighborMode = FaceNeighborMode.LOCAL_ONE_TO_ONE,
    passive_var_list: Sequence[int | VarStatus] | None = None,
    value_location: Sequence[int | ValueLocation] | None = None,
    share_var_from_zone: Sequence[int] | None = None,
    share_connectivity_from_zone: int = 0,
) -> None:
    """Create a mixed finite-element zone (classic API).

    Args:
        zone_title (str): Zone title.
        num_nodes (int): Number of nodes.
        num_elements (int): Number of elements.
        num_nodes_per_element (int): Nodes per element.
        solution_time (float): Solution time.
        strand_id (int): Strand ID for transient data (0 for static data, positive
            integer for transient data).
        parent_zone (int): Parent zone index (0 = none).
        datapacking (int | DataPacking): DataPacking.BLOCK (1) or DataPacking.POINT (0)
        num_face_connections (int): Number of face connections.
        face_neighbor_mode (int | FaceNeighborMode): Face-neighbor mode.
        passive_var_list (Sequence[int | VarStatus] | None): PList of VarStatus enums or
            0/1 for passive variables. None/null means all variables are active. Must be
            same length as value_location if provided.
        value_location (Sequence[int | ValueLocation] | None): List of ValueLocation
            enums or 0/1 for nodal/cell-centered. Must be same length as
            passive_var_list if provided.
        share_var_from_zone (Sequence[int] | None): List of zone indices to share
            variables from.
        share_connectivity_from_zone (int): Zone index to share connectivity from.

    Raises:
        TecioError: On C library error.
    """
    num_nodes_c = ctypes.c_int32(num_nodes)
    num_elements_c = ctypes.c_int32(num_elements)
    num_nodes_per_element_c = ctypes.c_int32(num_nodes_per_element)
    solution_time_c = ctypes.c_double(solution_time)
    strand_id_c = ctypes.c_int32(strand_id)
    parent_zone_c = ctypes.c_int32(parent_zone)
    is_block_c = ctypes.c_int32(_to_int_value(datapacking))
    num_face_connections_c = ctypes.c_int32(num_face_connections)
    face_neighbor_mode_c = ctypes.c_int32(_to_int_value(face_neighbor_mode))
    share_connectivity_c = ctypes.c_int32(share_connectivity_from_zone)

    # Handle optional array parameters
    passive_array = _process_sequence(passive_var_list)
    passive_ptr = (
        ctypes.cast(passive_array, ctypes.POINTER(ctypes.c_int32))
        if passive_array
        else ctypes.POINTER(ctypes.c_int32)()
    )

    value_loc_array = _process_sequence(value_location)
    value_loc_ptr = (
        ctypes.cast(value_loc_array, ctypes.POINTER(ctypes.c_int32))
        if value_loc_array
        else ctypes.POINTER(ctypes.c_int32)()
    )

    share_var_array = _process_sequence(share_var_from_zone)
    share_var_ptr = (
        ctypes.cast(share_var_array, ctypes.POINTER(ctypes.c_int32))
        if share_var_array
        else ctypes.POINTER(ctypes.c_int32)()
    )

    ret = lib.tecznefemixed142(
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.byref(num_nodes_c),
        ctypes.byref(num_elements_c),
        ctypes.byref(num_nodes_per_element_c),
        ctypes.byref(solution_time_c),
        ctypes.byref(strand_id_c),
        ctypes.byref(parent_zone_c),
        ctypes.byref(is_block_c),
        ctypes.byref(num_face_connections_c),
        ctypes.byref(face_neighbor_mode_c),
        passive_ptr,
        value_loc_ptr,
        share_var_ptr,
        ctypes.byref(share_connectivity_c),
    )
    if ret != 0:
        raise TecioError(
            f"tecznefemixed142 Error: zone_title={zone_title!r}, return_code={ret}"
        )


# Data writing
def tecdat142(
    field_data: npt.ArrayLike,
    is_double: bool = True,
) -> None:
    """Write field data to the current zone (classic API).

    Args:
        field_data (npt.ArrayLike): Array of field values.
        is_double (bool): True for double precision, False for single.

    Raises:
        TecioError: On C library error.

    Important:
        Data must be written in the order specified by dataset variables and zone
        definition

    Hint:
        If variable is shared or passive, skip writing data for that variable

    Note:
        For ORDERED zones: data should be ordered with the I dimension varying
        fastest, then J, then K

    Note:
        For FE zones: data should be ordered by node index for nodal variables, and by
        element index for cell-centered variables
    """
    # Convert to appropriate numpy array
    if is_double:
        arr = np.ascontiguousarray(field_data, dtype=np.float64)
        data_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        is_double_c = ctypes.c_int32(1)
    else:
        arr = np.ascontiguousarray(field_data, dtype=np.float32)
        data_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        is_double_c = ctypes.c_int32(0)

    ret = lib.tecdat142(
        ctypes.c_int32(arr.size),
        data_ptr,
        is_double_c,
    )
    if ret != 0:
        raise TecioError(
            f"tecdat142 Error: n={arr.size}, is_double={is_double}, return_code={ret}"
        )


# Connectivity writing
def tecnode142(nodes: npt.ArrayLike) -> None:
    """Write node connectivity for an FE zone (classic API).

    Args:
        nodes (npt.ArrayLike): Array of 1-based node indices.

    Raises:
        TecioError: On C library error.

    Important:
        Node indices are 1-based (Fortran convention)

    Hint:
        - For FEBRICK elements: 8 nodes per element.
        - For FETETRAHEDRON elements: 4 nodes per element.
    """
    nodes_array = np.ascontiguousarray(nodes, dtype=np.int32)
    n = ctypes.c_int32(nodes_array.size)
    nodes_ptr = nodes_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    ret = lib.tecnode142(
        ctypes.byref(n),
        nodes_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecnode142 Error: n_nodes={nodes_array.size}, return_code={ret}"
        )


def tecface142(face_connections: npt.ArrayLike) -> None:
    """Write face-neighbor connections (classic API).

    Args:
        face_connections (npt.ArrayLike): Array of face connection data.

    Raises:
        TecioError: On C library error.

    Note:
        Used to specify face-to-face connectivity between zones.
    """
    face_conn_array = np.ascontiguousarray(face_connections, dtype=np.int32)
    face_conn_ptr = face_conn_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    ret = lib.tecface142(face_conn_ptr)
    if ret != 0:
        raise TecioError(f"tecface142 Error: return_code={ret}")


def tecpolyface142(
    face_node_counts: npt.ArrayLike,
    face_nodes: npt.ArrayLike,
    face_left_elems: npt.ArrayLike,
    face_right_elems: npt.ArrayLike,
) -> None:
    """Write face data for polygonal/polyhedral zones (classic API).

    Args:
        face_node_counts (npt.ArrayLike): Number of nodes per face.
        face_nodes (npt.ArrayLike): Concatenated face node indices.
        face_left_elems (npt.ArrayLike): Left element per face.
        face_right_elems (npt.ArrayLike): Right element per face.

    Raises:
        TecioError: On C library error.

    Note:
        ``face_node_counts`` has one entry per face.

    Note:
        ``face_nodes`` is a concatenated list of all face node indices.

    Hint:
        Element indices are 1-based; 0 indicates boundary.
    """
    face_node_counts_array = np.ascontiguousarray(face_node_counts, dtype=np.int32)
    face_nodes_array = np.ascontiguousarray(face_nodes, dtype=np.int32)
    face_left_elems_array = np.ascontiguousarray(face_left_elems, dtype=np.int32)
    face_right_elems_array = np.ascontiguousarray(face_right_elems, dtype=np.int32)

    num_faces = ctypes.c_int32(face_node_counts_array.size)
    face_node_counts_ptr = face_node_counts_array.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32)
    )
    face_nodes_ptr = face_nodes_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    face_left_elems_ptr = face_left_elems_array.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32)
    )
    face_right_elems_ptr = face_right_elems_array.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32)
    )

    ret = lib.tecpolyface142(
        ctypes.byref(num_faces),
        face_node_counts_ptr,
        face_nodes_ptr,
        face_left_elems_ptr,
        face_right_elems_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecpolyface142 Error: num_faces={face_node_counts_array.size}, "
            f"return_code={ret}"
        )


def tecpolybconn142(
    boundary_connection_counts: npt.ArrayLike,
    boundary_connection_elems: npt.ArrayLike,
    boundary_connection_zones: npt.ArrayLike | None = None,
) -> None:
    """Write boundary connections for poly zones (classic API).

    Args:
        boundary_connection_counts (npt.ArrayLike): Connections per
            boundary face.
        boundary_connection_elems (npt.ArrayLike): Element indices.
        boundary_connection_zones (npt.ArrayLike | None): Zone indices.

    Raises:
        TecioError: On C library error.

    Note:
        Used to specify connectivity to neighboring zones at boundaries.
    """
    bconn_counts_array = np.ascontiguousarray(
        boundary_connection_counts, dtype=np.int32
    )
    bconn_elems_array = np.ascontiguousarray(boundary_connection_elems, dtype=np.int32)

    num_boundary_faces = ctypes.c_int32(bconn_counts_array.size)
    bconn_counts_ptr = bconn_counts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    bconn_elems_ptr = bconn_elems_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    if boundary_connection_zones is not None:
        bconn_zones_array = np.ascontiguousarray(
            boundary_connection_zones, dtype=np.int16
        )
        bconn_zones_ptr = bconn_zones_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16)
        )
    else:
        bconn_zones_ptr = ctypes.POINTER(ctypes.c_int16)()

    ret = lib.tecpolybconn142(
        ctypes.byref(num_boundary_faces),
        bconn_counts_ptr,
        bconn_elems_ptr,
        bconn_zones_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecpolybconn142 Error: num_boundary_faces={bconn_counts_array.size}, "
            f"return_code={ret}"
        )


# Auxiliary data
def tecauxstr142(name: str, value: str) -> None:
    """Add dataset-level auxiliary data (classic API).

    Args:
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called after ``tecini142`` but before first ``teczne142``.
    """
    ret = lib.tecauxstr142(
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecauxstr142 Error: name={name!r}, value={value!r}, return_code={ret}"
        )


def tecvauxstr142(var: int, name: str, value: str) -> None:
    """Add variable-level auxiliary data (classic API).

    Args:
        var (int): 1-based variable index.
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called after ``tecini142`` but before first ``teczne142``.
    """
    var_c = ctypes.c_int32(var)

    ret = lib.tecvauxstr142(
        ctypes.byref(var_c),
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecvauxstr142 Error: var={var}, name={name!r}, value={value!r}, "
            f"return_code={ret}"
        )


def teczauxstr142(name: str, value: str) -> None:
    """Add zone-level auxiliary data (classic API).

    Args:
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called after ``teczne142`` for the current zone.
    """
    ret = lib.teczauxstr142(
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"teczauxstr142 Error: name={name!r}, value={value!r}, return_code={ret}"
        )


# User-defined data (custom records)
def tecusr142(user_rec: str) -> None:
    """Write a user-defined data record (classic API).

    Args:
        user_rec (str): User-defined data string.

    Raises:
        TecioError: On C library error.

    Note:
        Used to write custom data records into the file.
        Data is preserved but not interpreted by Tecplot
    """
    ret = lib.tecusr142(ctypes.c_char_p(user_rec.encode("utf-8")))
    if ret != 0:
        raise TecioError(f"tecusr142 Error: user_rec={user_rec!r}, return_code={ret}")
