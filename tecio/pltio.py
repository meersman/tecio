from __future__ import annotations

import ctypes
from enum import Enum
from typing import Optional, Sequence, Union

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
    """Base exception for all pltio C/C++ library errors."""


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
# ---- Helper functions ----------------------------------------------
# --------------------------------------------------------------------


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
