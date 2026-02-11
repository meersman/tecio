from __future__ import annotations

import ctypes
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from . import tecutils

# Load tecio library
TECIO_LIB_PATH = tecutils.get_tecio_lib()
lib = ctypes.cdll.LoadLibrary(TECIO_LIB_PATH)


# --------------------------------------------------------------------
# ---- Tecio exception classes ---------------------------------------
# --------------------------------------------------------------------


class TecioError(RuntimeError):
    """Base exception for all libtecio C/C++ library errors."""


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


class FaceNeighborMode(Enum):
    LOCAL_ONE_TO_ONE = 0
    LOCAL_ONE_TO_MANY = 1
    GLOBAL_ONE_TO_ONE = 2
    GLOBAL_ONE_TO_MANY = 3


class DataFormat(Enum):
    POINT = 0
    BLOCK = 1


# --------------------------------------------------------------------
# ---- Helper functions ----------------------------------
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


def _to_int_value(value: Union[int, Enum]) -> int:
    """Convert Enum or int to int value."""
    if isinstance(value, Enum):
        return value.value
    return int(value)


def _process_sequence(
    seq: Optional[Sequence[Union[int, Enum]]],
) -> Optional[ctypes.Array]:
    """Convert sequence of int/Enum to ctypes array, handling None."""
    if seq is None:
        return None
    values = [_to_int_value(v) for v in seq]
    return (ctypes.c_int32 * len(values))(*values)


# --------------------------------------------------------------------
# ---- C library bindings: SZL Read ----------------------------------
# --------------------------------------------------------------------

# ---- Reading SZL files ---------------------------------------------
lib.tecFileReaderOpen.restype = ctypes.c_int32
lib.tecFileReaderOpen.argtypes = [
    ctypes.c_char_p,
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

# ---- Reading SZL zones ---------------------------------------------
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

# ---- Reading SZL variable data -------------------------------------
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

# ---- Reading SZL aux data ------------------------------------------
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

# --------------------------------------------------------------------
# ---- C library bindings: SZL Write ---------------------------------
# --------------------------------------------------------------------

# ---- Create SZL objects --------------------------------------------
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
lib.tecFileWriterClose.restype = ctypes.c_int32
lib.tecFileWriterClose.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
]
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
lib.tecZoneSetUnsteadyOptions.restype = ctypes.c_int32
lib.tecZoneSetUnsteadyOptions.argtypes = [
    ctypes.c_void_p,  # file_handle
    ctypes.c_int32,  # zone
    ctypes.c_double,  # solutionTime
    ctypes.c_int32,  # strand
]

# ---- Write variable value functions --------------------------------
lib.tecZoneVarWriteDoubleValues.restype = ctypes.c_int32
lib.tecZoneVarWriteDoubleValues.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,  # partition
    ctypes.c_int64,  # count
    ctypes.POINTER(ctypes.c_double),
]
lib.tecZoneVarWriteFloatValues.restype = ctypes.c_int32
lib.tecZoneVarWriteFloatValues.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_float),
]
lib.tecZoneVarWriteInt32Values.restype = ctypes.c_int32
lib.tecZoneVarWriteInt32Values.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int32),
]
lib.tecZoneVarWriteInt16Values.restype = ctypes.c_int32
lib.tecZoneVarWriteInt16Values.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int16),
]
lib.tecZoneVarWriteUInt8Values.restype = ctypes.c_int32
lib.tecZoneVarWriteUInt8Values.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint8),
]

# --------------------------------------------------------------------
# ---- C library bindings: PLT Functions -----------------------------
# --------------------------------------------------------------------

# ---- File initialization and finalization --------------------------
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

# ---- Zone creation -------------------------------------------------
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
    ctypes.POINTER(ctypes.c_int32),  # NumFaces
    ctypes.POINTER(ctypes.c_int32),  # NumElements
    ctypes.POINTER(ctypes.c_int32),  # TotalNumFaceNodes
    ctypes.POINTER(ctypes.c_double),  # SolutionTime
    ctypes.POINTER(ctypes.c_int32),  # StrandID
    ctypes.POINTER(ctypes.c_int32),  # ParentZone
    ctypes.POINTER(ctypes.c_int32),  # IsBlock
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

# ---- Partitioned zone creation -------------------------------------
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

# ---- Data writing --------------------------------------------------
lib.tecdat142.restype = ctypes.c_int32
lib.tecdat142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # N (number of values)
    ctypes.c_void_p,  # FieldData (void pointer for flexibility)
    ctypes.POINTER(ctypes.c_int32),  # IsDouble (1=double, 0=float)
]

# ---- Connectivity writing ------------------------------------------
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

# ---- Auxiliary data ------------------------------------------------
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

# ---- MPI initialization (for parallel I/O) -------------------------
lib.tecmpiinit142.restype = ctypes.c_int32
lib.tecmpiinit142.argtypes = [
    ctypes.POINTER(ctypes.c_int32),  # Communicator
    ctypes.POINTER(ctypes.c_int32),  # MainRank
]

# ---- User-defined data (custom records) ----------------------------
lib.tecusr142.restype = ctypes.c_int32
lib.tecusr142.argtypes = [
    ctypes.c_char_p,  # UserRec
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
    - TecioError if the underlying lib.tecFileReaderOpen returns non-zero.
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


def tec_file_get_type(handle: ctypes.c_void_p) -> FileType:
    """
    Get the FileType for an opened SZL file.

    Inputs:
    - handle: ctypes.c_void_p returned by tec_file_reader_open.

    Returns:
    - FileType enum indicating FULL, GRID or SOLUTION.

    Raises:
    - TecioError on non-zero return code from lib.
    """
    file_type = ctypes.c_int32(0)

    ret = lib.tecFileGetType(handle, ctypes.byref(file_type))
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

    ret = lib.tecDataSetGetTitle(handle, ctypes.byref(title))
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

    ret = lib.tecDataSetGetNumVars(handle, ctypes.byref(num_vars))
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

    ret = lib.tecDataSetGetNumZones(handle, ctypes.byref(num_zones))
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

    ret = lib.tecZoneGetIJK(
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

    ret = lib.tecZoneGetTitle(
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

    ret = lib.tecZoneGetType(
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

    ret = lib.tecZoneIsEnabled(
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

    ret = lib.tecZoneGetSolutionTime(
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

    ret = lib.tecZoneGetStrandID(
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
    ret = lib.tecZoneNodeMapIs64Bit(
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

    ret = lib.tecVarGetName(handle, ctypes.c_int32(var_index), ctypes.byref(var_name))
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

    ret = lib.tecVarIsEnabled(
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

    ret = lib.tecZoneVarGetValueLocation(
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

    ret = lib.tecZoneVarGetSharedZone(
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

    ret = lib.tecZoneVarGetNumValues(
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

    ret = lib.tecDataSetAuxDataGetNumItems(handle, ctypes.byref(num_auxdata_items))
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

    ret = lib.tecDataSetAuxDataGetItem(
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

    ret = lib.tecVarAuxDataGetNumItems(
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

    ret = lib.tecVarAuxDataGetItem(
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

    ret = lib.tecZoneAuxDataGetNumItems(
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

    ret = lib.tecZoneAuxDataGetItem(
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


# ---- Initialization and File Handling ------------------------------
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
        raise TypeError("file_type must be a libtecio.FileType enum")

    handle = ctypes.c_void_p()
    ft_int = int(file_type.value)

    ret = lib.tecFileWriterOpen(
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
    ret = lib.tecFileWriterClose(ctypes.byref(handle))
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
                raise TypeError("All var_types entries must be libtecio.DataType enums")
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
                    "All value_locations entries must be libtecio.ValueLocation enums"
                )
            vl_list.append(int(v.value))
        arr = (ctypes.c_int32 * len(vl_list))(*vl_list)
        value_locations_ptr = arr

    ret = lib.tecZoneCreateIJK(
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


def tec_zone_create_fe(
    handle: ctypes.c_void_p,
    zone_title: str,
    zone_type: Union[int, ZoneType],
    num_nodes: int,
    num_cells: int,
    var_types: Optional[Sequence[DataType]] = None,
    var_sharing: Optional[Sequence[int]] = None,
    value_locations: Optional[Sequence[ValueLocation]] = None,
) -> int:
    """
    Create an FE zone for writing.

    Inputs:
    - handle: ctypes.c_void_p writer handle
    - zone_title: zone title string
    - zone_type: ZoneType Enum type
    - num_nodes: integer number of nodes
    - num_cells: integer number of cells
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
                raise TypeError("All var_types entries must be libtecio.DataType enums")
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
                    "All value_locations entries must be libtecio.ValueLocation enums"
                )
            vl_list.append(int(v.value))
        arr = (ctypes.c_int32 * len(vl_list))(*vl_list)
        value_locations_ptr = arr

    ret = lib.tecZoneCreateFE(
        handle,
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int32(_to_int_value(zone_type)),
        ctypes.c_int64(num_nodes),
        ctypes.c_int64(num_cells),
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
            f"tecZoneCreateIJK Error: zone_title={zone_title!r}, ZoneType={zone_type!r}, NODES={num_nodes}, ELEMENTS={num_elements}, "
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

    ret = lib.tecZoneVarWriteFloatValues(
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


# --------------------------------------------------------------------
# ---- Wrappers for C functions: PLT ---------------------------------
# --------------------------------------------------------------------


# ---- File initialization and finalization --------------------------
def tecini142(
    title: str,
    variables: str,
    fname: str,
    scratch_dir: str = ".",
    file_format: Union[int, FileFormat] = FileFormat.PLT,
    file_type: Union[int, FileType] = FileType.FULL,
    debug: Union[int, Debug] = Debug.FALSE,
    vis_double: Union[int, DataType] = DataType.DOUBLE,
) -> None:
    """
    Initialize a Tecplot data file.

    Inputs:
    - title: Dataset title
    - variables: Space or comma-separated variable names
    - fname: Output file name (.plt or .szplt)
    - scratch_dir: Scratch directory for temporary files
    - file_format: FileFormat.PLT (0) or FileFormat.SZPLT (1)
    - file_type: FileType.FULL (0), GRID (1), or SOLUTION (2)
    - debug: Debug.FALSE (0) or Debug.TRUE (1)
    - vis_double: DataType.DOUBLE (1) or DataType.FLOAT (0)

    Returns:
    - None

    Raises:
    - TecioError if initialization fails

    Notes:
    - Must be called before any zone or data operations
    - Call tecend142() to finalize the file
    """
    file_format_c = ctypes.c_int32(_to_int_value(file_format))
    file_type_c = ctypes.c_int32(_to_int_value(file_type))
    debug_c = ctypes.c_int32(_to_int_value(debug))
    vis_double_c = ctypes.c_int32(
        1 if _to_int_value(vis_double) == DataType.DOUBLE.value else 0
    )

    ret = lib.tecini142(
        ctypes.c_char_p(title.encode("utf-8")),
        ctypes.c_char_p(variables.encode("utf-8")),
        ctypes.c_char_p(fname.encode("utf-8")),
        ctypes.c_char_p(scratch_dir.encode("utf-8")),
        ctypes.byref(file_format_c),
        ctypes.byref(file_type_c),
        ctypes.byref(debug_c),
        ctypes.byref(vis_double_c),
    )
    if ret != 0:
        raise TecioError(f"tecini142 Error: fname={fname!r}, return_code={ret}")


def tecend142() -> None:
    """
    Finalize and close the Tecplot data file.

    Returns:
    - None

    Raises:
    - TecioError if finalization fails

    Notes:
    - Must be called after all data has been written
    - Flushes all pending data and closes the file
    """
    ret = lib.tecend142()
    if ret != 0:
        raise TecioError(f"tecend142 Error: return_code={ret}")


def tecflush142(
    num_zones_to_retain: int = 0,
    zones_to_retain: Optional[Sequence[int]] = None,
) -> None:
    """
    Flush data to disk, optionally retaining zones in memory.

    Inputs:
    - num_zones_to_retain: Number of zones to keep in memory
    - zones_to_retain: List of zone indices to retain (1-based)

    Returns:
    - None

    Raises:
    - TecioError if flush fails

    Notes:
    - Used to reduce memory usage for large files
    - Retained zones can still be modified
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
    """
    Get the file handle for the current output file.

    Returns:
    - int: File handle

    Raises:
    - TecioError if operation fails

    Notes:
    - Used for advanced file operations
    """
    output_file_handle = ctypes.c_int32(0)

    ret = lib.tecfil142(ctypes.byref(output_file_handle))
    if ret != 0:
        raise TecioError(f"tecfil142 Error: return_code={ret}")

    return output_file_handle.value


def tecforeign142(output_foreign_byte_order: int) -> None:
    """
    Set foreign byte order for output.

    Inputs:
    - output_foreign_byte_order: 0=native, 1=foreign

    Returns:
    - None

    Raises:
    - TecioError if operation fails

    Notes:
    - Used to control endianness of output files
    """
    foreign_c = ctypes.c_int32(output_foreign_byte_order)

    ret = lib.tecforeign142(ctypes.byref(foreign_c))
    if ret != 0:
        raise TecioError(f"tecforeign142 Error: return_code={ret}")


# ---- Zone creation -------------------------------------------------
def teczne142(
    zone_title: str,
    zone_type: Union[int, ZoneType],
    imx: int,
    jmx: int,
    kmx: int,
    icell_max: int = 0,
    jcell_max: int = 0,
    kcell_max: int = 0,
    solution_time: float = 0.0,
    strand_id: int = 0,
    parent_zone: int = 0,
    data_format: Union[int, DataFormat] = DataFormat.BLOCK,
    num_face_connections: int = 0,
    face_neighbor_mode: Union[
        int, FaceNeighborMode
    ] = FaceNeighborMode.LOCAL_ONE_TO_ONE,
    total_num_face_nodes: int = 0,
    num_connected_boundary_faces: int = 0,
    total_num_boundary_connections: int = 0,
    passive_var_list: Optional[Sequence[Union[int, VarStatus]]] = None,
    value_location: Optional[Sequence[Union[int, ValueLocation]]] = None,
    share_var_from_zone: Optional[Sequence[int]] = None,
    share_connectivity_from_zone: int = 0,
) -> None:
    """
    Create a new zone in the Tecplot file.

    Inputs:
    - zone_title: Zone title
    - zone_type: ZoneType enum or int (0=ORDERED, 1=FELINESEG, etc.)
    - imx: I-dimension (or NumNodes for FE)
    - jmx: J-dimension (or NumElements for FE)
    - kmx: K-dimension
    - icell_max: I-cell dimension (for cell-centered data)
    - jcell_max: J-cell dimension (for cell-centered data)
    - kcell_max: K-cell dimension (for cell-centered data)
    - solution_time: Solution time for transient data
    - strand_id: Strand ID for transient data
    - parent_zone: Parent zone index (0=none)
    - data_format: DataFormat.BLOCK (1) or DataFormat.POINT (0)
    - num_face_connections: Number of face connections
    - face_neighbor_mode: FaceNeighborMode enum
    - total_num_face_nodes: Total face nodes (for poly zones)
    - num_connected_boundary_faces: Boundary faces (for poly zones)
    - total_num_boundary_connections: Boundary connections (for poly zones)
    - passive_var_list: List of VarStatus enums or 0/1 for passive variables
    - value_location: List of ValueLocation enums or 0/1 for nodal/cell-centered
    - share_var_from_zone: List of zone indices to share variables from
    - share_connectivity_from_zone: Zone index to share connectivity from

    Returns:
    - None

    Raises:
    - TecioError if zone creation fails

    Notes:
    - For ORDERED zones: imx, jmx, kmx are dimensions
    - For FE zones: imx=NumNodes, jmx=NumElements, kmx=0
    """
    zone_type_c = ctypes.c_int32(_to_int_value(zone_type))
    imx_c = ctypes.c_int32(imx)
    jmx_c = ctypes.c_int32(jmx)
    kmx_c = ctypes.c_int32(kmx)
    icell_max_c = ctypes.c_int32(icell_max)
    jcell_max_c = ctypes.c_int32(jcell_max)
    kcell_max_c = ctypes.c_int32(kcell_max)
    solution_time_c = ctypes.c_double(solution_time)
    strand_id_c = ctypes.c_int32(strand_id)
    parent_zone_c = ctypes.c_int32(parent_zone)
    is_block_c = ctypes.c_int32(_to_int_value(data_format))
    num_face_connections_c = ctypes.c_int32(num_face_connections)
    face_neighbor_mode_c = ctypes.c_int32(_to_int_value(face_neighbor_mode))
    total_num_face_nodes_c = ctypes.c_int32(total_num_face_nodes)
    num_connected_boundary_faces_c = ctypes.c_int32(num_connected_boundary_faces)
    total_num_boundary_connections_c = ctypes.c_int32(total_num_boundary_connections)
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

    ret = lib.teczne142(
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.byref(zone_type_c),
        ctypes.byref(imx_c),
        ctypes.byref(jmx_c),
        ctypes.byref(kmx_c),
        ctypes.byref(icell_max_c),
        ctypes.byref(jcell_max_c),
        ctypes.byref(kcell_max_c),
        ctypes.byref(solution_time_c),
        ctypes.byref(strand_id_c),
        ctypes.byref(parent_zone_c),
        ctypes.byref(is_block_c),
        ctypes.byref(num_face_connections_c),
        ctypes.byref(face_neighbor_mode_c),
        ctypes.byref(total_num_face_nodes_c),
        ctypes.byref(num_connected_boundary_faces_c),
        ctypes.byref(total_num_boundary_connections_c),
        passive_ptr,
        value_loc_ptr,
        share_var_ptr,
        ctypes.byref(share_connectivity_c),
    )
    if ret != 0:
        raise TecioError(
            f"teczne142 Error: zone_title={zone_title!r}, return_code={ret}"
        )


def tecpolyzne142(
    zone_title: str,
    zone_type: Union[int, ZoneType],
    num_nodes: int,
    num_faces: int,
    num_elements: int,
    total_num_face_nodes: int,
    solution_time: float = 0.0,
    strand_id: int = 0,
    parent_zone: int = 0,
    data_format: Union[int, DataFormat] = DataFormat.BLOCK,
    num_connected_boundary_faces: int = 0,
    total_num_boundary_connections: int = 0,
    passive_var_list: Optional[Sequence[Union[int, VarStatus]]] = None,
    value_location: Optional[Sequence[Union[int, ValueLocation]]] = None,
    share_var_from_zone: Optional[Sequence[int]] = None,
    share_connectivity_from_zone: int = 0,
) -> None:
    """
    Create a polygonal or polyhedral zone.

    Inputs:
    - zone_title: Zone title
    - zone_type: ZoneType.FEPOLYGON or ZoneType.FEPOLYHEDRON
    - num_nodes: Number of nodes
    - num_faces: Number of faces
    - num_elements: Number of elements
    - total_num_face_nodes: Total number of face nodes
    - solution_time: Solution time for transient data
    - strand_id: Strand ID for transient data
    - parent_zone: Parent zone index (0=none)
    - data_format: DataFormat.BLOCK (1) or DataFormat.POINT (0)
    - num_connected_boundary_faces: Number of boundary faces
    - total_num_boundary_connections: Total boundary connections
    - passive_var_list: List of VarStatus enums or 0/1 for passive variables
    - value_location: List of ValueLocation enums or 0/1 for nodal/cell-centered
    - share_var_from_zone: List of zone indices to share variables from
    - share_connectivity_from_zone: Zone index to share connectivity from

    Returns:
    - None

    Raises:
    - TecioError if zone creation fails
    """
    zone_type_c = ctypes.c_int32(_to_int_value(zone_type))
    num_nodes_c = ctypes.c_int32(num_nodes)
    num_faces_c = ctypes.c_int32(num_faces)
    num_elements_c = ctypes.c_int32(num_elements)
    total_num_face_nodes_c = ctypes.c_int32(total_num_face_nodes)
    solution_time_c = ctypes.c_double(solution_time)
    strand_id_c = ctypes.c_int32(strand_id)
    parent_zone_c = ctypes.c_int32(parent_zone)
    is_block_c = ctypes.c_int32(_to_int_value(data_format))
    num_connected_boundary_faces_c = ctypes.c_int32(num_connected_boundary_faces)
    total_num_boundary_connections_c = ctypes.c_int32(total_num_boundary_connections)
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

    ret = lib.tecpolyzne142(
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.byref(zone_type_c),
        ctypes.byref(num_nodes_c),
        ctypes.byref(num_faces_c),
        ctypes.byref(num_elements_c),
        ctypes.byref(total_num_face_nodes_c),
        ctypes.byref(solution_time_c),
        ctypes.byref(strand_id_c),
        ctypes.byref(parent_zone_c),
        ctypes.byref(is_block_c),
        ctypes.byref(num_connected_boundary_faces_c),
        ctypes.byref(total_num_boundary_connections_c),
        passive_ptr,
        value_loc_ptr,
        share_var_ptr,
        ctypes.byref(share_connectivity_c),
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
    data_format: Union[int, DataFormat] = DataFormat.BLOCK,
    num_face_connections: int = 0,
    face_neighbor_mode: Union[
        int, FaceNeighborMode
    ] = FaceNeighborMode.LOCAL_ONE_TO_ONE,
    passive_var_list: Optional[Sequence[Union[int, VarStatus]]] = None,
    value_location: Optional[Sequence[Union[int, ValueLocation]]] = None,
    share_var_from_zone: Optional[Sequence[int]] = None,
    share_connectivity_from_zone: int = 0,
) -> None:
    """
    Create a mixed FE zone.

    Inputs:
    - zone_title: Zone title
    - num_nodes: Number of nodes
    - num_elements: Number of elements
    - num_nodes_per_element: Number of nodes per element
    - solution_time: Solution time for transient data
    - strand_id: Strand ID for transient data
    - parent_zone: Parent zone index (0=none)
    - data_format: DataFormat.BLOCK (1) or DataFormat.POINT (0)
    - num_face_connections: Number of face connections
    - face_neighbor_mode: FaceNeighborMode enum
    - passive_var_list: List of VarStatus enums or 0/1 for passive variables
    - value_location: List of ValueLocation enums or 0/1 for nodal/cell-centered
    - share_var_from_zone: List of zone indices to share variables from
    - share_connectivity_from_zone: Zone index to share connectivity from

    Returns:
    - None

    Raises:
    - TecioError if zone creation fails
    """
    num_nodes_c = ctypes.c_int32(num_nodes)
    num_elements_c = ctypes.c_int32(num_elements)
    num_nodes_per_element_c = ctypes.c_int32(num_nodes_per_element)
    solution_time_c = ctypes.c_double(solution_time)
    strand_id_c = ctypes.c_int32(strand_id)
    parent_zone_c = ctypes.c_int32(parent_zone)
    is_block_c = ctypes.c_int32(_to_int_value(data_format))
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


# ---- Partitioned zone creation -------------------------------------
def tecijkptn142(
    partition_owner_zone: int,
    imin: int,
    jmin: int,
    kmin: int,
    imax: int,
    jmax: int,
    kmax: int,
) -> None:
    """
    Create an ordered partition for an existing zone.

    Inputs:
    - partition_owner_zone: Zone index of the owner zone
    - imin: Minimum I index
    - jmin: Minimum J index
    - kmin: Minimum K index
    - imax: Maximum I index
    - jmax: Maximum J index
    - kmax: Maximum K index

    Returns:
    - None

    Raises:
    - TecioError if partition creation fails

    Notes:
    - Used for parallel I/O to partition large zones
    """
    partition_owner_zone_c = ctypes.c_int32(partition_owner_zone)
    imin_c = ctypes.c_int32(imin)
    jmin_c = ctypes.c_int32(jmin)
    kmin_c = ctypes.c_int32(kmin)
    imax_c = ctypes.c_int32(imax)
    jmax_c = ctypes.c_int32(jmax)
    kmax_c = ctypes.c_int32(kmax)

    ret = lib.tecijkptn142(
        ctypes.byref(partition_owner_zone_c),
        ctypes.byref(imin_c),
        ctypes.byref(jmin_c),
        ctypes.byref(kmin_c),
        ctypes.byref(imax_c),
        ctypes.byref(jmax_c),
        ctypes.byref(kmax_c),
    )
    if ret != 0:
        raise TecioError(
            f"tecijkptn142 Error: partition_owner_zone={partition_owner_zone}, "
            f"return_code={ret}"
        )


def tecfeptn142(
    partition_owner_zone: int,
    num_nodes: int,
    num_elements: int,
) -> None:
    """
    Create an FE partition for an existing zone.

    Inputs:
    - partition_owner_zone: Zone index of the owner zone
    - num_nodes: Number of nodes in this partition
    - num_elements: Number of elements in this partition

    Returns:
    - None

    Raises:
    - TecioError if partition creation fails
    """
    partition_owner_zone_c = ctypes.c_int32(partition_owner_zone)
    num_nodes_c = ctypes.c_int32(num_nodes)
    num_elements_c = ctypes.c_int32(num_elements)

    ret = lib.tecfeptn142(
        ctypes.byref(partition_owner_zone_c),
        ctypes.byref(num_nodes_c),
        ctypes.byref(num_elements_c),
    )
    if ret != 0:
        raise TecioError(
            f"tecfeptn142 Error: partition_owner_zone={partition_owner_zone}, "
            f"return_code={ret}"
        )


def tecfemixedptn142(
    partition_owner_zone: int,
    num_nodes: int,
    num_elements: int,
    num_nodes_per_element: int,
) -> None:
    """
    Create a mixed FE partition for an existing zone.

    Inputs:
    - partition_owner_zone: Zone index of the owner zone
    - num_nodes: Number of nodes in this partition
    - num_elements: Number of elements in this partition
    - num_nodes_per_element: Number of nodes per element

    Returns:
    - None

    Raises:
    - TecioError if partition creation fails
    """
    partition_owner_zone_c = ctypes.c_int32(partition_owner_zone)
    num_nodes_c = ctypes.c_int32(num_nodes)
    num_elements_c = ctypes.c_int32(num_elements)
    num_nodes_per_element_c = ctypes.c_int32(num_nodes_per_element)

    ret = lib.tecfemixedptn142(
        ctypes.byref(partition_owner_zone_c),
        ctypes.byref(num_nodes_c),
        ctypes.byref(num_elements_c),
        ctypes.byref(num_nodes_per_element_c),
    )
    if ret != 0:
        raise TecioError(
            f"tecfemixedptn142 Error: partition_owner_zone={partition_owner_zone}, "
            f"return_code={ret}"
        )


# ---- Data writing --------------------------------------------------
def tecdat142(
    field_data: npt.ArrayLike,
    is_double: bool = True,
) -> None:
    """
    Write field data to the current zone.

    Inputs:
    - field_data: Array of field values (numpy array, list, or tuple)
    - is_double: True for double precision, False for single precision

    Returns:
    - None

    Raises:
    - TecioError if data writing fails

    Notes:
    - Data must be written in the order specified by zone format (BLOCK or POINT)
    - For BLOCK format: write all values for variable 1, then variable 2, etc.
    - For POINT format: write all variables for point 1, then point 2, etc.
    """
    # Convert to appropriate numpy array
    if is_double:
        arr = np.ascontiguousarray(field_data, dtype=np.float64)
        data_ptr = arr.ctypes.data_as(ctypes.c_void_p)
    else:
        arr = np.ascontiguousarray(field_data, dtype=np.float32)
        data_ptr = arr.ctypes.data_as(ctypes.c_void_p)

    n = ctypes.c_int32(arr.size)
    is_double_c = ctypes.c_int32(1 if is_double else 0)

    ret = lib.tecdat142(
        ctypes.byref(n),
        data_ptr,
        ctypes.byref(is_double_c),
    )
    if ret != 0:
        raise TecioError(
            f"tecdat142 Error: n={arr.size}, is_double={is_double}, return_code={ret}"
        )


# ---- Connectivity writing ------------------------------------------
def tecnod142(connectivity: Sequence[int]) -> None:
    """
    Write node connectivity for FE zone (deprecated, use tecnode142).

    Inputs:
    - connectivity: Array of node indices (1-based)

    Returns:
    - None

    Raises:
    - TecioError if connectivity writing fails

    Notes:
    - Deprecated: Use tecnode142() or tecznemap142() instead
    """
    conn_array = (ctypes.c_int32 * len(connectivity))(*connectivity)
    conn_ptr = ctypes.cast(conn_array, ctypes.POINTER(ctypes.c_int32))

    ret = lib.tecnod142(conn_ptr)
    if ret != 0:
        raise TecioError(f"tecnod142 Error: return_code={ret}")


def tecnode142(connectivity: npt.ArrayLike) -> None:
    """
    Write node connectivity for FE zone.

    Inputs:
    - connectivity: Array of node indices (1-based)

    Returns:
    - None

    Raises:
    - TecioError if connectivity writing fails

    Notes:
    - Node indices are 1-based (Fortran convention)
    - For FEBRICK elements: 8 nodes per element
    - For FETETRAHEDRON elements: 4 nodes per element
    - etc.
    """
    conn_array = np.ascontiguousarray(connectivity, dtype=np.int32)
    n = ctypes.c_int32(conn_array.size)
    conn_ptr = conn_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    ret = lib.tecnode142(
        ctypes.byref(n),
        conn_ptr,
    )
    if ret != 0:
        raise TecioError(f"tecnode142 Error: n={conn_array.size}, return_code={ret}")


def tecznemap142(node_map: npt.ArrayLike) -> None:
    """
    Write node map for FE zone (alternative to tecnode142).

    Inputs:
    - node_map: Array of node indices (1-based)

    Returns:
    - None

    Raises:
    - TecioError if node map writing fails

    Notes:
    - Similar to tecnode142 but with different internal handling
    """
    node_map_array = np.ascontiguousarray(node_map, dtype=np.int32)
    n = ctypes.c_int32(node_map_array.size)
    node_map_ptr = node_map_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    ret = lib.tecznemap142(
        ctypes.byref(n),
        node_map_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecznemap142 Error: n={node_map_array.size}, return_code={ret}"
        )


def tecface142(face_connections: npt.ArrayLike) -> None:
    """
    Write face neighbor connections.

    Inputs:
    - face_connections: Array of face connection data

    Returns:
    - None

    Raises:
    - TecioError if face connection writing fails

    Notes:
    - Used to specify face-to-face connectivity between zones
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
    """
    Write face data for polygonal/polyhedral zones.

    Inputs:
    - face_node_counts: Number of nodes for each face
    - face_nodes: Node indices for all faces (concatenated)
    - face_left_elems: Left element index for each face
    - face_right_elems: Right element index for each face

    Returns:
    - None

    Raises:
    - TecioError if face data writing fails

    Notes:
    - face_node_counts has one entry per face
    - face_nodes is concatenated list of all face node indices
    - Element indices are 1-based; 0 indicates boundary
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
    boundary_connection_zones: Optional[npt.ArrayLike] = None,
) -> None:
    """
    Write boundary connections for polygonal/polyhedral zones.

    Inputs:
    - boundary_connection_counts: Number of connections per boundary face
    - boundary_connection_elems: Element indices for boundary connections
    - boundary_connection_zones: Zone indices for boundary connections (optional)

    Returns:
    - None

    Raises:
    - TecioError if boundary connection writing fails

    Notes:
    - Used to specify connectivity to neighboring zones at boundaries
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


# ---- Auxiliary data ------------------------------------------------
def tecauxstr142(name: str, value: str) -> None:
    """
    Add dataset-level auxiliary data.

    Inputs:
    - name: Auxiliary data name
    - value: Auxiliary data value

    Returns:
    - None

    Raises:
    - TecioError if auxiliary data writing fails

    Notes:
    - Must be called after tecini142() but before first teczne142()
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
    """
    Add variable-level auxiliary data.

    Inputs:
    - var: Variable index (1-based)
    - name: Auxiliary data name
    - value: Auxiliary data value

    Returns:
    - None

    Raises:
    - TecioError if auxiliary data writing fails

    Notes:
    - Must be called after tecini142() but before first teczne142()
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
    """
    Add zone-level auxiliary data.

    Inputs:
    - name: Auxiliary data name
    - value: Auxiliary data value

    Returns:
    - None

    Raises:
    - TecioError if auxiliary data writing fails

    Notes:
    - Must be called after teczne142() for the current zone
    """
    ret = lib.teczauxstr142(
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"teczauxstr142 Error: name={name!r}, value={value!r}, return_code={ret}"
        )


# ---- MPI initialization (for parallel I/O) -------------------------
def tecmpiinit142(communicator: int, main_rank: int) -> None:
    """
    Initialize MPI for parallel I/O.

    Inputs:
    - communicator: MPI communicator handle
    - main_rank: Rank of the main process

    Returns:
    - None

    Raises:
    - TecioError if MPI initialization fails

    Notes:
    - Must be called before tecini142() for parallel I/O
    - Only required for parallel file writing
    """
    communicator_c = ctypes.c_int32(communicator)
    main_rank_c = ctypes.c_int32(main_rank)

    ret = lib.tecmpiinit142(
        ctypes.byref(communicator_c),
        ctypes.byref(main_rank_c),
    )
    if ret != 0:
        raise TecioError(
            f"tecmpiinit142 Error: communicator={communicator}, "
            f"main_rank={main_rank}, return_code={ret}"
        )


# ---- User-defined data (custom records) ----------------------------
def tecusr142(user_rec: str) -> None:
    """
    Write user-defined data record.

    Inputs:
    - user_rec: User-defined data string

    Returns:
    - None

    Raises:
    - TecioError if user record writing fails

    Notes:
    - Used to write custom data records into the file
    - Data is preserved but not interpreted by Tecplot
    """
    ret = lib.tecusr142(ctypes.c_char_p(user_rec.encode("utf-8")))
    if ret != 0:
        raise TecioError(f"tecusr142 Error: user_rec={user_rec!r}, return_code={ret}")
