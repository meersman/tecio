r"""
:mod:`datwriter`: Tecplot ASCII DAT file writer
================================================

This module provides :class:`Write`, a context-manager-based writer for
Tecplot 360 ASCII data files (``.dat``).  The interface mirrors the binary
SZL and PLT writers so that calling code can swap file formats with minimal
changes::

    with Write("output.dat", title="My Dataset",
                   variables=["X", "Y", "Z", "Pressure"],
                   file_type="FULL") as writer:
        # optional dataset and variable auxiliary data
        writer.write_dataset_auxdata("Common.ReferencePressure", "101325.0")
        writer.write_var_auxdata(3, "Common.PressureRef", "101325.0")

        # ordered zone
        writer.write_ordered_zone(
            title="Grid",
            ijk=(100, 50, 1),
            var_data=[x_arr, y_arr, z_arr, p_arr],
            solution_time=1.0,
            strand_id=1,
        )

        # finite-element zone
        writer.write_fe_zone(
            title="Surface",
            zone_type="FETRIANGLE",
            num_points=n_pts,
            num_elements=n_tri,
            var_data=[x_arr, y_arr, z_arr, p_arr],
            connectivity=conn,          # (N, nodes_per_elem) int array
        )

File structure written
----------------------
::

    TITLE = "<title>"
    [FILETYPE = GRID | SOLUTION]   (omitted when FULL – the default)
    VARIABLES = "v1" "v2" ...
    DATASETAUXDATA <name> = "<value>"    (zero or more)
    VARAUXDATA  <1-based-index> <name> = "<value>"  (zero or more)
    ZONE T="<title>", ZONETYPE=ORDERED, I=i, J=j, K=k,
         DATAPACKING=BLOCK, SOLUTIONTIME=t, STRANDID=s
    <var0 data block>
    <var1 data block>
    ...
    [ZONE ...]

Numbers per line
----------------
The writer emits :attr:`Write.VALUES_PER_LINE` values per line (default
10) for variable data and connectivity, matching common Tecplot practice.

Format specification reference
-------------------------------
Tecplot 360 Data Format Guide 2025 R2, "ASCII Data" chapter (pp. 143–197).

Supported zone types
--------------------
* Ordered (``ORDERED``)
* ``FELINESEG``
* ``FETRIANGLE``
* ``FEQUADRILATERAL``
* ``FETETRAHEDRON``
* ``FEBRICK``
* ``FEPOLYGON`` (variable nodes per face; pass *face_node_counts* and
  *face_nodes*)
* ``FEPOLYHEDRON`` (variable faces per element; pass *face_node_counts*,
  *face_nodes*, *face_left_elems*, *face_right_elems*)

Passive and shared variables
-----------------------------
Pass ``passive_vars`` (list of 0-based variable indices) or
``shared_vars`` (dict mapping 0-based index → 1-based source zone number,
or the string ``"FECONNECT"`` to share only the connectivity list) to the
zone-writing methods.

Notes
-----
* All data arrays are accepted as any array-like convertible to
  :class:`numpy.ndarray`.
* Variable values are written in BLOCK format (one variable at a time),
  which is by far the most efficient layout for loading into Tecplot 360.
* Floating-point values are written with :attr:`Write.FLOAT_FMT`
  significant figures (default ``".9g"``).
* The class writes to a :func:`io.BufferedWriter`-backed text file
  (``newline="\\n"``) so that line endings are always ``LF`` on every
  platform.
"""

# Standard library
import io
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Third-party
import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Zone-type keyword strings accepted by :class:`Write`.
ORDERED_ZONE_TYPE: str = "ORDERED"

#: Valid ``file_type`` values for :class:`Write`.
#: ``"FULL"`` is the default and need not be written to the file header.
VALID_FILE_TYPES: frozenset = frozenset({"FULL", "GRID", "SOLUTION"})

#: Mapping from zone-type name → nodes per element (0 = variable).
_NODES_PER_ELEM: Dict[str, int] = {
    "FELINESEG": 2,
    "FETRIANGLE": 3,
    "FEQUADRILATERAL": 4,
    "FETETRAHEDRON": 4,
    "FEBRICK": 8,
    "FEPOLYGON": 0,       # variable; face-based
    "FEPOLYHEDRON": 0,    # variable; face-based
}

# Zone types whose connectivity is face-based rather than element-node-based.
_FACE_BASED_ZONE_TYPES: frozenset = frozenset({"FEPOLYGON", "FEPOLYHEDRON"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quote(s: str) -> str:
    r"""Wrap *s* in double quotes, escaping any embedded double-quote with ``\``."""
    return '"' + s.replace('"', '\\"') + '"'


def _needs_quoting(name: str) -> bool:
    """Return ``True`` if *name* contains spaces or special characters."""
    special = set(' \t\n\r,=')
    return bool(special.intersection(name))


def _fmt_name(name: str) -> str:
    """Quote *name* only when necessary."""
    if _needs_quoting(name):
        return _quote(name)
    return name


class Write:
    r"""Context-manager writer for Tecplot 360 ASCII (``.dat``) files.

    Parameters
    ----------
    path:
        Destination file path.
    title:
        Dataset title written to the file header.
    variables:
        Ordered list of variable names.  Must not be empty.
    file_type:
        One of ``"FULL"`` (default), ``"GRID"``, or ``"SOLUTION"``.
        ``"FULL"`` contains both grid and solution data and is the most
        common case; the keyword is **omitted** from the header when
        ``"FULL"`` is used so that older Tecplot versions that do not
        recognise the keyword continue to work.  ``"GRID"`` and
        ``"SOLUTION"`` write the ``FILETYPE`` keyword explicitly and are
        used when splitting grid and solution into separate files for
        transient datasets.
    float_fmt:
        ``%``-style or ``str.format``-compatible format string used to
        render every floating-point value.  Defaults to ``".9g"`` which
        gives up to 9 significant digits (Tecplot's own default).
    values_per_line:
        Number of values emitted per line for variable data and
        connectivity lists.  Default is ``10``.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from datwriter import Write

        x = np.linspace(0.0, 1.0, 5)
        y = np.zeros(5)

        with Write("out.dat", title="Demo", variables=["X", "Y"]) as w:
            w.write_dataset_auxdata("Author", "PyTecplot")
            w.write_ordered_zone("Line", (5, 1, 1), [x, y])

    Grid / solution split for transient data::

        with Write("grid.dat", title="Grid", variables=["X", "Y", "Z"],
                       file_type="GRID") as w:
            w.write_ordered_zone("Base", (100, 50, 1), [x, y, z])

        with Write("sol.dat", title="Solution t=1",
                       variables=["X", "Y", "Z", "Pressure"],
                       file_type="SOLUTION") as w:
            w.write_ordered_zone("Base", (100, 50, 1),
                                 [None, None, None, p],
                                 shared_vars={0: 1, 1: 1, 2: 1})
    """

    #: Default float format string (significant digits).
    FLOAT_FMT: str = ".9g"

    #: Default number of values per output line.
    VALUES_PER_LINE: int = 10

    # ------------------------------------------------------------------
    # Construction / context management
    # ------------------------------------------------------------------

    def __init__(
        self,
        path: str,
        title: str = "",
        variables: Optional[Sequence[str]] = None,
        file_type: str = "FULL",
        float_fmt: Optional[str] = None,
        values_per_line: Optional[int] = None,
    ) -> None:
        if variables is None or len(variables) == 0:
            raise ValueError("Write requires at least one variable name.")

        file_type_upper = file_type.upper()
        if file_type_upper not in VALID_FILE_TYPES:
            raise ValueError(
                f"Invalid file_type {file_type!r}.  "
                f"Expected one of {sorted(VALID_FILE_TYPES)}."
            )

        self._path: str = path
        self._title: str = title
        self._variables: List[str] = list(variables)
        self._file_type: str = file_type_upper
        self._float_fmt: str = float_fmt if float_fmt is not None else self.FLOAT_FMT
        self._vpl: int = (
            values_per_line if values_per_line is not None else self.VALUES_PER_LINE
        )
        self._fp: Optional[io.TextIOWrapper] = None
        self._header_written: bool = False

    def __enter__(self) -> "Write":
        # Open with explicit LF line endings and UTF-8 encoding.
        self._fp = open(  # noqa: WPS515
            self._path, "w", encoding="utf-8", newline="\n"
        )
        self._write_file_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        # Propagate exceptions.
        return False

    # ------------------------------------------------------------------
    # Auxiliary data
    # ------------------------------------------------------------------

    def write_dataset_auxdata(self, name: str, value: str) -> None:
        r"""Write a dataset-level auxiliary data record.

        Must be called *after* entering the context (i.e. after
        ``__enter__``) and *before* or *between* zone records.  Tecplot
        360 accepts ``DATASETAUXDATA`` records anywhere in the file but
        conventionally they appear just after the file header.

        Parameters
        ----------
        name:
            Auxiliary data key (e.g. ``"Common.ReferencePressure"``).
        value:
            Auxiliary data value string.

        Example output::

            DATASETAUXDATA Common.ReferencePressure = "101325.0"
        """
        self._ensure_open()
        self._fp.write(f"DATASETAUXDATA {name} = {_quote(value)}\n")

    def write_var_auxdata(
        self, var_index: int, name: str, value: str
    ) -> None:
        r"""Write a variable-level auxiliary data record.

        Parameters
        ----------
        var_index:
            **0-based** variable index.  Converted to 1-based internally.
        name:
            Auxiliary data key.
        value:
            Auxiliary data value string.

        Example output::

            VARAUXDATA 4 Common.PressureRef = "101325.0"
        """
        self._ensure_open()
        one_based = var_index + 1
        self._fp.write(f"VARAUXDATA {one_based} {name} = {_quote(value)}\n")

    # ------------------------------------------------------------------
    # Zone writers
    # ------------------------------------------------------------------

    def write_ordered_zone(
        self,
        title: str,
        ijk: Tuple[int, int, int],
        var_data: Sequence[ArrayLike],
        *,
        solution_time: float = 0.0,
        strand_id: int = -1,
        var_location: Optional[Dict[int, str]] = None,
        passive_vars: Optional[Sequence[int]] = None,
        shared_vars: Optional[Dict[int, Union[int, str]]] = None,
        zone_auxdata: Optional[Dict[str, str]] = None,
    ) -> None:
        r"""Write an ordered (structured) zone.

        Parameters
        ----------
        title:
            Zone title.
        ijk:
            ``(I, J, K)`` dimensions.  Use ``K=1`` for 2-D, ``J=K=1``
            for 1-D data.
        var_data:
            Sequence of array-like objects, one per variable.  Pass
            ``None`` in the sequence for passive or shared variables.
        solution_time:
            Solution time associated with this zone.
        strand_id:
            Strand ID for transient data.  ``-1`` means no strand.
        var_location:
            Optional mapping of 0-based variable index → ``"NODAL"`` or
            ``"CELLCENTERED"``.  Variables not listed default to
            ``"NODAL"``.
        passive_vars:
            0-based indices of variables to mark as *passive* (the value
            array may be ``None``).
        shared_vars:
            0-based index → 1-based source zone number.  When provided
            the variable is not written; Tecplot reads it from the
            referenced zone.
        zone_auxdata:
            Optional dict of zone-level auxiliary data to emit after the
            ``ZONE`` header line.
        """
        self._ensure_open()
        i, j, k = ijk
        zone_type = ORDERED_ZONE_TYPE
        header = self._build_zone_header(
            title=title,
            zone_type=zone_type,
            i=i,
            j=j,
            k=k,
            solution_time=solution_time,
            strand_id=strand_id,
            var_location=var_location,
            passive_vars=passive_vars,
            shared_vars=shared_vars,
        )
        self._fp.write(header + "\n")
        self._write_zone_auxdata(zone_auxdata)
        self._write_var_blocks(
            var_data=var_data,
            passive_vars=passive_vars,
            shared_vars=shared_vars,
        )

    def write_fe_zone(
        self,
        title: str,
        zone_type: str,
        num_points: int,
        num_elements: int,
        var_data: Sequence[ArrayLike],
        connectivity: Optional[ArrayLike] = None,
        *,
        solution_time: float = 0.0,
        strand_id: int = -1,
        var_location: Optional[Dict[int, str]] = None,
        passive_vars: Optional[Sequence[int]] = None,
        shared_vars: Optional[Dict[int, Union[int, str]]] = None,
        shared_connectivity: Optional[int] = None,
        zone_auxdata: Optional[Dict[str, str]] = None,
        # Polygon / polyhedron face-connectivity arguments
        face_node_counts: Optional[ArrayLike] = None,
        face_nodes: Optional[ArrayLike] = None,
        face_left_elems: Optional[ArrayLike] = None,
        face_right_elems: Optional[ArrayLike] = None,
    ) -> None:
        r"""Write a finite-element zone.

        Parameters
        ----------
        title:
            Zone title.
        zone_type:
            One of ``"FELINESEG"``, ``"FETRIANGLE"``,
            ``"FEQUADRILATERAL"``, ``"FETETRAHEDRON"``, ``"FEBRICK"``,
            ``"FEPOLYGON"``, ``"FEPOLYHEDRON"``.
        num_points:
            Number of nodes (data points).
        num_elements:
            Number of elements (cells / faces).
        var_data:
            One array-like per variable; ``None`` entries for passive /
            shared variables.
        connectivity:
            For classic element types (not poly*): integer array of shape
            ``(num_elements, nodes_per_elem)`` using **1-based** node
            numbers.  May be ``None`` when using *shared_connectivity*.
        solution_time:
            Solution time for transient data.
        strand_id:
            Strand ID; ``-1`` disables.
        var_location:
            0-based index → ``"NODAL"`` or ``"CELLCENTERED"``.
        passive_vars:
            0-based indices of passive variables.
        shared_vars:
            0-based index → 1-based source zone number for shared
            variables.
        shared_connectivity:
            1-based zone number whose connectivity list is shared.
        zone_auxdata:
            Zone-level auxiliary data dict.
        face_node_counts:
            ``FEPOLYGON`` / ``FEPOLYHEDRON``: 1-D integer array of face
            node counts.
        face_nodes:
            Flat 1-D integer array of 1-based node numbers for each face.
        face_left_elems:
            ``FEPOLYHEDRON``: 1-D integer array of 1-based element
            numbers to the left of each face (0 for boundary).
        face_right_elems:
            ``FEPOLYHEDRON``: 1-D integer array of 1-based element
            numbers to the right of each face (0 for boundary).
        """
        self._ensure_open()
        zone_type_upper = zone_type.upper()
        if zone_type_upper not in _NODES_PER_ELEM:
            raise ValueError(
                f"Unknown FE zone type: {zone_type!r}.  "
                f"Expected one of {list(_NODES_PER_ELEM)}"
            )

        is_face_based = zone_type_upper in _FACE_BASED_ZONE_TYPES
        num_faces: Optional[int] = None
        if is_face_based:
            if face_node_counts is None or face_nodes is None:
                raise ValueError(
                    f"Zone type {zone_type!r} requires face_node_counts "
                    "and face_nodes."
                )
            num_faces = int(np.asarray(face_node_counts).size)

        header = self._build_zone_header(
            title=title,
            zone_type=zone_type_upper,
            num_points=num_points,
            num_elements=num_elements,
            num_faces=num_faces,
            solution_time=solution_time,
            strand_id=strand_id,
            var_location=var_location,
            passive_vars=passive_vars,
            shared_vars=shared_vars,
            shared_connectivity=shared_connectivity,
        )
        self._fp.write(header + "\n")
        self._write_zone_auxdata(zone_auxdata)

        # Variable data blocks
        self._write_var_blocks(
            var_data=var_data,
            passive_vars=passive_vars,
            shared_vars=shared_vars,
        )

        # Connectivity
        if not is_face_based:
            if shared_connectivity is None:
                if connectivity is None:
                    raise ValueError(
                        "connectivity is required for non-face-based FE zones "
                        "unless shared_connectivity is specified."
                    )
                self._write_connectivity(
                    connectivity=connectivity,
                    num_elements=num_elements,
                    nodes_per_elem=_NODES_PER_ELEM[zone_type_upper],
                )
        else:
            # Polygon / polyhedron face data
            self._write_face_based_connectivity(
                zone_type=zone_type_upper,
                face_node_counts=face_node_counts,
                face_nodes=face_nodes,
                face_left_elems=face_left_elems,
                face_right_elems=face_right_elems,
            )

    # ------------------------------------------------------------------
    # Private helpers – header construction
    # ------------------------------------------------------------------

    def _write_file_header(self) -> None:
        """Emit TITLE, optional FILETYPE, and VARIABLES lines.

        ``FILETYPE`` is written only for ``GRID`` and ``SOLUTION`` files.
        ``FULL`` (the default) is deliberately omitted so that files remain
        compatible with older Tecplot versions that do not recognise the
        keyword.
        """
        fp = self._fp
        fp.write(f'TITLE = {_quote(self._title)}\n')
        # Emit FILETYPE only when it is not the default (FULL).
        if self._file_type != "FULL":
            fp.write(f"FILETYPE = {self._file_type}\n")
        # Build VARIABLES line; quote names that contain spaces.
        var_strs = " ".join(_quote(v) for v in self._variables)
        fp.write(f"VARIABLES = {var_strs}\n")
        self._header_written = True

    def _build_zone_header(
        self,
        title: str,
        zone_type: str,
        i: int = 0,
        j: int = 0,
        k: int = 0,
        num_points: int = 0,
        num_elements: int = 0,
        num_faces: Optional[int] = None,
        solution_time: float = 0.0,
        strand_id: int = -1,
        var_location: Optional[Dict[int, str]] = None,
        passive_vars: Optional[Sequence[int]] = None,
        shared_vars: Optional[Dict[int, Union[int, str]]] = None,
        shared_connectivity: Optional[int] = None,
    ) -> str:
        """Build and return the ``ZONE`` header string (without trailing \\n)."""
        parts: List[str] = [f"ZONE T={_quote(title)}"]

        parts.append(f"ZONETYPE={zone_type}")

        if zone_type == ORDERED_ZONE_TYPE:
            parts.append(f"I={i}")
            parts.append(f"J={j}")
            parts.append(f"K={k}")
        else:
            parts.append(f"N={num_points}")
            parts.append(f"E={num_elements}")
            if num_faces is not None:
                parts.append(f"FACES={num_faces}")

        parts.append("DATAPACKING=BLOCK")

        if strand_id >= 0:
            parts.append(f"STRANDID={strand_id}")
            parts.append(f"SOLUTIONTIME={format(solution_time, self._float_fmt)}")
        elif solution_time != 0.0:
            parts.append(f"SOLUTIONTIME={format(solution_time, self._float_fmt)}")

        # Variable locations
        if var_location:
            loc_parts = self._build_varloc_param(var_location)
            if loc_parts:
                parts.append(loc_parts)

        # Passive variables (1-based list)
        if passive_vars:
            one_based = [str(v + 1) for v in passive_vars]
            parts.append(f"PASSIVEVARLIST=[{','.join(one_based)}]")

        # Shared variables
        if shared_vars:
            sv_parts = self._build_shared_var_param(shared_vars)
            if sv_parts:
                parts.append(sv_parts)

        # Shared connectivity
        if shared_connectivity is not None:
            parts.append(f"CONNECTIVITYSHAREZONE={shared_connectivity}")

        # Tecplot uses comma+space as separators on the ZONE header line.
        return ", ".join(parts)

    def _build_varloc_param(
        self, var_location: Dict[int, str]
    ) -> str:
        """Build the ``VARLOCATION`` parameter string."""
        # Only emit entries that differ from NODAL (the default).
        cell_centered = sorted(
            v for v, loc in var_location.items()
            if loc.upper() == "CELLCENTERED"
        )
        if not cell_centered:
            return ""
        one_based = [str(v + 1) for v in cell_centered]
        return f"VARLOCATION=([{','.join(one_based)}]=CELLCENTERED)"

    def _build_shared_var_param(
        self, shared_vars: Dict[int, Union[int, str]]
    ) -> str:
        """Build the ``VARSHARELIST`` parameter string."""
        # Entries are: 1-based-index=source_zone (integer) or FECONNECT.
        entries: List[str] = []
        for var_idx, src in sorted(shared_vars.items()):
            one_based = var_idx + 1
            if isinstance(src, str):
                entries.append(f"{one_based}={src}")
            else:
                entries.append(f"{one_based}={src}")
        if not entries:
            return ""
        return f"VARSHARELIST=([{','.join(entries)}])"

    # ------------------------------------------------------------------
    # Private helpers – data writing
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._fp is None or self._fp.closed:
            raise IOError(
                "Write is not open.  Use it as a context manager: "
                "with Write(...) as w: ..."
            )

    def _write_zone_auxdata(
        self, zone_auxdata: Optional[Dict[str, str]]
    ) -> None:
        """Write zero or more ``AUXDATA`` records inside a zone."""
        if not zone_auxdata:
            return
        for name, value in zone_auxdata.items():
            self._fp.write(f"AUXDATA {name} = {_quote(value)}\n")

    def _write_var_blocks(
        self,
        var_data: Sequence[Optional[ArrayLike]],
        passive_vars: Optional[Sequence[int]],
        shared_vars: Optional[Dict[int, Union[int, str]]],
    ) -> None:
        """Write one BLOCK-format data block per active variable."""
        passive_set: frozenset = frozenset(passive_vars or [])
        shared_set: frozenset = frozenset((shared_vars or {}).keys())

        for var_idx, data in enumerate(var_data):
            if var_idx in passive_set or var_idx in shared_set:
                # Passive / shared variables produce no data in this zone.
                continue
            if data is None:
                raise ValueError(
                    f"var_data[{var_idx}] is None but variable {var_idx} is "
                    "not in passive_vars or shared_vars."
                )
            arr = np.asarray(data).ravel()
            self._write_values(arr)

    def _write_values(self, arr: NDArray) -> None:
        r"""Write a flat 1-D array to the file, *values_per_line* per line."""
        fp = self._fp
        vpl = self._vpl
        fmt = self._float_fmt
        n = len(arr)
        is_float = np.issubdtype(arr.dtype, np.floating)

        for start in range(0, n, vpl):
            chunk = arr[start: start + vpl]
            if is_float:
                line = " ".join(format(v, fmt) for v in chunk)
            else:
                # Integer-typed arrays (e.g. connectivity)
                line = " ".join(str(int(v)) for v in chunk)
            fp.write(line + "\n")

    def _write_connectivity(
        self,
        connectivity: ArrayLike,
        num_elements: int,
        nodes_per_elem: int,
    ) -> None:
        r"""Write a classic (non-face-based) connectivity list.

        Parameters
        ----------
        connectivity:
            Shape ``(num_elements, nodes_per_elem)`` using 1-based node
            indices.  May also be a flat array of length
            ``num_elements * nodes_per_elem``.
        num_elements:
            Number of elements.
        nodes_per_elem:
            Nodes per element implied by the zone type.
        """
        conn = np.asarray(connectivity, dtype=np.intp)
        conn = conn.reshape(num_elements, nodes_per_elem)
        fp = self._fp
        for row in conn:
            fp.write(" ".join(str(int(n)) for n in row) + "\n")

    def _write_face_based_connectivity(
        self,
        zone_type: str,
        face_node_counts: ArrayLike,
        face_nodes: ArrayLike,
        face_left_elems: Optional[ArrayLike],
        face_right_elems: Optional[ArrayLike],
    ) -> None:
        """Write FEPOLYGON / FEPOLYHEDRON face connectivity data."""
        fnc = np.asarray(face_node_counts, dtype=np.intp).ravel()
        fn = np.asarray(face_nodes, dtype=np.intp).ravel()

        # Face node counts
        self._write_values(fnc)
        # Face nodes
        self._write_values(fn)

        if zone_type == "FEPOLYHEDRON":
            if face_left_elems is None or face_right_elems is None:
                raise ValueError(
                    "FEPOLYHEDRON zones require face_left_elems and "
                    "face_right_elems."
                )
            fle = np.asarray(face_left_elems, dtype=np.intp).ravel()
            fre = np.asarray(face_right_elems, dtype=np.intp).ravel()
            self._write_values(fle)
            self._write_values(fre)
