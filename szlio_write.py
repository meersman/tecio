from __future__ import annotations

import ctypes
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

import tecutils
# use the canonical Enums from the read-side module
from szlio import DataType, FileType, ValueLocation, ZoneType

# Load tecio library (uses same discovery helper as read-side code)
TECIO_LIB_PATH = tecutils.get_tecio_lib()
tecio = ctypes.cdll.LoadLibrary(TECIO_LIB_PATH)


class TecioError(RuntimeError):
    """Runtime error for tecio write wrappers."""


# ---- C function prototypes (writer API) -----------------------------
tecio.tecFileWriterOpen.restype = ctypes.c_int32
tecio.tecFileWriterOpen.argtypes = (
    ctypes.c_char_p,  # fileName
    ctypes.c_char_p,  # dataSetTitle
    ctypes.c_char_p,  # varNames (comma separated)
    ctypes.c_int32,  # useSZL (1)
    ctypes.c_int32,  # fileType
    ctypes.c_int32,  # reserved / options
    ctypes.c_void_p,  # gridFileHandle (optional)
    ctypes.POINTER(ctypes.c_void_p),  # out fileHandle
)

tecio.tecFileWriterClose.restype = ctypes.c_int32
tecio.tecFileWriterClose.argtypes = (ctypes.POINTER(ctypes.c_void_p),)

tecio.tecZoneCreateIJK.restype = ctypes.c_int32
tecio.tecZoneCreateIJK.argtypes = (
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
)

tecio.tecZoneSetUnsteadyOptions.restype = ctypes.c_int32
tecio.tecZoneSetUnsteadyOptions.argtypes = (
    ctypes.c_void_p,  # file_handle
    ctypes.c_int32,  # zone
    ctypes.c_double,  # solutionTime
    ctypes.c_int32,  # strand
)

# write variable value functions
tecio.tecZoneVarWriteDoubleValues.restype = ctypes.c_int32
tecio.tecZoneVarWriteDoubleValues.argtypes = (
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,  # partition
    ctypes.c_int64,  # count
    ctypes.POINTER(ctypes.c_double),
)

tecio.tecZoneVarWriteFloatValues.restype = ctypes.c_int32
tecio.tecZoneVarWriteFloatValues.argtypes = (
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_float),
)

tecio.tecZoneVarWriteInt32Values.restype = ctypes.c_int32
tecio.tecZoneVarWriteInt32Values.argtypes = (
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int32),
)

tecio.tecZoneVarWriteInt16Values.restype = ctypes.c_int32
tecio.tecZoneVarWriteInt16Values.argtypes = (
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int16),
)

tecio.tecZoneVarWriteUInt8Values.restype = ctypes.c_int32
tecio.tecZoneVarWriteUInt8Values.argtypes = (
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_uint8),
)


# ---- helper to prepare numpy arrays for ctypes -----------------------
def _prepare_array_for_ctypes(
    values: npt.ArrayLike, np_dtype, ctype
) -> tuple[ctypes.POINTER, int, npt.NDArray]:
    """
    Ensure 'values' is a contiguous numpy array with dtype np_dtype and
    return (ctypes_ptr, count, backing_array).
    """
    arr = np.ascontiguousarray(values, dtype=np_dtype)
    count = int(arr.size)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctype))
    return ptr, count, arr


# ---- Low-level wrappers (updated to accept numpy array-likes) -------------
def tec_file_writer_open(
    file_name: str,
    dataset_title: str,
    var_names_csv: str,
    file_type: FileType,
    use_szl: int = 1,
    grid_file_handle: Optional[ctypes.c_void_p] = None,
) -> ctypes.c_void_p:
    """Open a writer handle. Returns ctypes.c_void_p handle.

    file_type must be a szlio.FileType enum.
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
    """Close writer handle in-place (expects ctypes.c_void_p)."""
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
    """Create ordered zone (I,J,K). Returns zone index (int).

    var_types must be a sequence of szlio.DataType enums (if provided).
    value_locations must be a sequence of szlio.ValueLocation enums (if provided).
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
    ret = tecio.tecZoneSetUnsteadyOptions(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_double(solution_time),
        ctypes.c_int32(strand),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneSetUnsteadyOptions Error: zone={zone}, strand={strand}, solution_time={solution_time}, return_code={ret}"
        )


def tec_zone_var_write_double_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
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
            f"tecZoneVarWriteDoubleValues Error: zone={zone}, var={var}, count={count}, return_code={ret}"
        )


def tec_zone_var_write_float_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
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
            f"tecZoneVarWriteFloatValues Error: zone={zone}, var={var}, count={count}, return_code={ret}"
        )


def tec_zone_var_write_int32_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: Sequence[int]
) -> None:
    count = len(values)
    vals = (ctypes.c_int32 * count)(*list(values))
    ret = tecio.tecZoneVarWriteInt32Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        vals,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteInt32Values Error: zone={zone}, var={var}, count={count}, return_code={ret}"
        )


def tec_zone_var_write_int16_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: Sequence[int]
) -> None:
    count = len(values)
    vals = (ctypes.c_int16 * count)(*list(values))
    ret = tecio.tecZoneVarWriteInt16Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        vals,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteInt16Values Error: zone={zone}, var={var}, count={count}, return_code={ret}"
        )


def tec_zone_var_write_uint8_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: Sequence[int]
) -> None:
    count = len(values)
    vals = (ctypes.c_uint8 * count)(*list(values))
    ret = tecio.tecZoneVarWriteUInt8Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        vals,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteUInt8Values Error: zone={zone}, var={var}, count={count}, return_code={ret}"
        )
