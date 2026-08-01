r"""Slice a Tecplot data file along IJK indices and/or solution time.

Structured CFD datasets frequently contain more spatial resolution or time steps than
are necessary for a given analysis.  Reducing the data to a relevant subset (like every
other grid point, a specific index range along one axis, or a window of solution times)
would otherwise require loading the full file in Tecplot or writing a dedicated script.
``tecslice`` applies these reductions directly to the file using a compact
colon-notation that mirrors Python slice syntax, supporting both structured (ordered)
zone thinning along IJK axes and time-step windowing across all zone types.

All slice arguments use the form ``start:end:skip``, where any component may be omitted.
IJK indices are one-based and inclusive at both ends; solution-time ``start`` and
``end`` are float values, and ``skip`` is an integer stride applied after the time
window is filtered.

.. list-table:: Slice notation reference
   :header-rows: 1
   :widths: 20 25 55

   * - Input
     - Equivalent
     - Meaning
   * - ``5``
     - ``[5:]``
     - From index 5 to end
   * - ``:10``
     - ``[:10]``
     - From start to index 10
   * - ``::2``
     - ``[::2]``
     - Every 2nd point or step
   * - ``2:10``
     - ``[2:10]``
     - Index 2 through 10 inclusive
   * - ``2:10:3``
     - ``[2:10:3]``
     - Every 3rd point from index 2 through 10
   * - ``::-1``
     - ``[::-1]``
     - Reverse the axis (e.g. mirror)

:Usage:

.. code:: bash

    tecslice [-h] -o PATH [-f] [-i [start]:[end]:[skip]] [-j [start]:[end]:[skip]] [-k
             [start]:[end]:[skip]] [-t [start]:[end]:[skip]] [--strand-id ID] FILE

:Positional Arguments:
    ``FILE``
        Path to the input Tecplot file (``.plt``, ``.szplt``, or ``.dat``) to slice.

:Options:
    ``-o PATH``, ``--output PATH``
        Output file path. Required. The extension controls the output format:
        ``.szplt``, ``.plt``, or ``.dat``.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

    ``-i [start]:[end]:[skip]``
        Slice along the I axis of ordered zones. Any component may be omitted. ``skip``
        may be negative to reverse the axis.

    ``-j [start]:[end]:[skip]``
        Slice along the J axis of ordered zones. Any component may be omitted. ``skip``
        may be negative to reverse the axis.

    ``-k [start]:[end]:[skip]``
        Slice along the K axis of ordered zones. Any component may be omitted. ``skip``
        may be negative to reverse the axis.

    ``-t [start]:[end]:[skip]``
        Slice solution times. ``start`` and ``end`` are float time values (inclusive);
        ``skip`` is an integer stride applied after the time window is
        filtered. Strand-0 zones are always written unchanged.

    ``--strand-id INT``
        Restrict time slicing to a single strand ID. Defaults to all strands with ID
        greater than zero.

:Returns:
    A new Tecplot file written to the output path containing only the requested subset
    of grid points and/or time steps. Exit code is ``0`` on success and non-zero if the
    input file cannot be read, an invalid slice is supplied, or the output file already
    exists and ``--force`` is not set.

Examples:
    Thin a structured grid to every other I point::

        $ tecslice -i ::2 -o thinned.szplt flow.szplt

    Extract a sub-block: I=2..10, J up to 5::

        $ tecslice -i 2:10 -j :5 -o sub.szplt flow.szplt

    Reverse the I axis::

        $ tecslice -i ::-1 -o mirrored.szplt flow.szplt

    Extract a solution time window::

        $ tecslice -t 0.5:2.0 -o window.szplt transient.szplt

    Keep every 3rd time step of strand 1 only::

        $ tecslice -t ::3 --strand-id 1 -o sparse.szplt transient.szplt

    Call directly from a Python session::

        import tecio.cli.tecslice.main as tecslice

        tecslice(["-i", "::2", "-o", "thinned.szplt", "flow.szplt"])

See Also:
    * :mod:`tecio.cli.tecextract`: Extract a subset of zones or variables by index
      rather than by positional slice.
    * :mod:`tecio.cli.tecsplit`: Split a file into separate grid and solution files.
    * :mod:`tecio.cli.tecmerge`: Merge zones from multiple files into a single output.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .. import open as tecio_open
from ..libtecio import ZoneType

# ---------------------------------------------------------------------------
# Slice spec types
# ---------------------------------------------------------------------------


class IjkSliceSpec(NamedTuple):
    """Parsed IJK slice: 1-based inclusive integer start/end, integer skip."""

    start: int | None
    end: int | None
    skip: int | None


class TimeSliceSpec(NamedTuple):
    """Parsed time slice: float start/end, integer skip."""

    start: float | None
    end: float | None
    skip: int | None


# ---------------------------------------------------------------------------
# Slice spec parsers
# ---------------------------------------------------------------------------


def _parse_ijk_slice(value: str) -> IjkSliceSpec:
    """Parse a colon-notation IJK slice string into an :class:`IjkSliceSpec`.

    Accepts any of::

        N          ->  start=N, end=None, skip=None
        :N         ->  start=None, end=N, skip=None
        ::N        ->  start=None, end=None, skip=N
        N:M        ->  start=N, end=M, skip=None
        N:M:S      ->  start=N, end=M, skip=S

    All values are integers.  ``skip`` may be negative (reversal).

    Args:
        value: Raw string from the command line.

    Returns:
        :class:`IjkSliceSpec` with parsed components.

    Raises:
        argparse.ArgumentTypeError: On invalid input.

    """
    parts = value.split(":")

    if len(parts) > 3:
        raise argparse.ArgumentTypeError(
            f"Too many ':' separators in slice {value!r}.  "
            "Expected at most start:end:skip."
        )

    try:
        if len(parts) == 1:
            # Bare integer -> treat as start.
            return IjkSliceSpec(
                start=int(parts[0]) if parts[0] else None,
                end=None,
                skip=None,
            )
        if len(parts) == 2:
            return IjkSliceSpec(
                start=int(parts[0]) if parts[0] else None,
                end=int(parts[1]) if parts[1] else None,
                skip=None,
            )
        # len == 3
        return IjkSliceSpec(
            start=int(parts[0]) if parts[0] else None,
            end=int(parts[1]) if parts[1] else None,
            skip=int(parts[2]) if parts[2] else None,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid IJK slice {value!r}: {exc}.  All components must be integers."
        ) from exc


def _parse_time_slice(value: str) -> TimeSliceSpec:
    """Parse a colon-notation time slice string into a :class:`TimeSliceSpec`.

    Same colon notation as IJK, but ``start`` and ``end`` are floats and
    ``skip`` is an integer stride.

    Args:
        value: Raw string from the command line.

    Returns:
        :class:`TimeSliceSpec` with parsed components.

    Raises:
        argparse.ArgumentTypeError: On invalid input.

    """
    parts = value.split(":")

    if len(parts) > 3:
        raise argparse.ArgumentTypeError(
            f"Too many ':' separators in time slice {value!r}.  "
            "Expected at most start:end:skip."
        )

    try:
        if len(parts) == 1:
            return TimeSliceSpec(
                start=float(parts[0]) if parts[0] else None,
                end=None,
                skip=None,
            )
        if len(parts) == 2:
            return TimeSliceSpec(
                start=float(parts[0]) if parts[0] else None,
                end=float(parts[1]) if parts[1] else None,
                skip=None,
            )
        # len == 3
        return TimeSliceSpec(
            start=float(parts[0]) if parts[0] else None,
            end=float(parts[1]) if parts[1] else None,
            skip=int(parts[2]) if parts[2] else None,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid time slice {value!r}: {exc}.  "
            "start/end must be floats, skip must be an integer."
        ) from exc


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecslice",
        description=(
            # -|--------------------|---------------------------------------------|
            "Slice a Tecplot file along IJK indices (ordered zones) and/or\n"
            "solution time (all zone types).\n"
            "Slice notation: start:end:skip (any component may be omitted)."
        ),
        epilog=(
            # -|--------------------|---------------------------------------------|
            "IJK indices are 1-based and inclusive. skip may be negative.\n"
            "Time start/end are float values; skip is an integer stride.\n\n"
            "Example usage:\n"
            "  Every other I point\n"
            "    $ tecslice -i ::2 -o thinned.szplt flow.szplt\n"
            "  I=2..10, J up to 5\n"
            "    $ tecslice -i 2:10 -j :5 -o sub.szplt flow.szplt\n"
            "  Reverse I (mirror)\n"
            "    $ tecslice -i ::-1 -o mirrored.szplt flow.szplt\n"
            "  Time window 0.5 to 2.0\n"
            "    $ tecslice -t 0.5:2.0 -o win.szplt transient.szplt\n"
            "  Every 3rd step of strand 1\n"
            "    $ tecslice -t ::3 --strand-id 1 -o sparse.szplt transient.szplt\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )

    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Output file path.  The extension controls the output format "
            "(.szplt, .plt, .dat)."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )

    # --- IJK slice flags -----------------------------------------------------
    ijk = parser.add_argument_group(
        "IJK slicing",
        (
            "Colon-notation slices for structured axes.  "
            "Indices are 1-based and inclusive.  "
            "Applied only to ordered (structured) zones."
        ),
    )
    for axis in ("i", "j", "k"):
        ijk.add_argument(
            f"-{axis}",
            type=_parse_ijk_slice,
            default=None,
            metavar="[start]:[end]:[skip]",
            help=(
                f"Slice along the {axis.upper()} axis.  "
                "Any component may be omitted.  "
                "skip may be negative to reverse the axis."
            ),
        )

    # --- Solution-time slice flags -------------------------------------------
    tslice = parser.add_argument_group(
        "solution-time slicing",
        (
            "Applied per strand to all zone types.  "
            "Strand-0 zones are always written unchanged."
        ),
    )
    tslice.add_argument(
        "-t",
        type=_parse_time_slice,
        default=None,
        metavar="[start]:[end]:[skip]",
        help=(
            "Slice solution times.  start and end are float time values "
            "(inclusive); skip is an integer stride applied after the "
            "time window filter."
        ),
    )
    tslice.add_argument(
        "--strand-id",
        type=int,
        default=None,
        metavar="ID",
        dest="strand_id",
        help=(
            "Apply time slicing to this strand ID only.  "
            "Default: all strands with ID > 0."
        ),
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Slice conversion helpers
# ---------------------------------------------------------------------------


def _ijk_spec_to_slice(spec: IjkSliceSpec | None) -> slice:
    """Convert an :class:`IjkSliceSpec` to a Python :class:`slice`.

    Applies the 1-based inclusive -> 0-based exclusive conversion.
    A ``None`` spec (axis not specified) returns ``slice(None)`` (full axis).

    Args:
        spec: Parsed spec, or ``None`` if the axis flag was not given.

    Returns:
        Python :class:`slice` ready for use with NumPy.

    """
    if spec is None:
        return slice(None)

    # 1-based inclusive start -> 0-based: always subtract 1.
    py_start = (spec.start - 1) if spec.start is not None else None

    # 1-based inclusive end -> Python exclusive stop.
    #
    # Positive step: 1-based end N -> py_stop N.
    #   arr[:N] (0-based) gives elements 0..N-1 = 1-based 1..N.  Correct.
    #
    # Negative step: 1-based end N -> py_stop N-1.
    #   arr[s:N-1:-1] includes 0-based index N-1 = 1-based N.  Correct.
    #   Special case: 1-based end=1 -> 0-based index 0 -> py_stop=None
    #   (arr[s:0:-1] would miss index 0; arr[s:None:-1] includes it).
    if spec.end is not None:
        is_neg = spec.skip is not None and spec.skip < 0
        py_end = (None if spec.end == 1 else spec.end - 1) if is_neg else spec.end
    else:
        py_end = None

    return slice(py_start, py_end, spec.skip)


def _has_ijk(args: argparse.Namespace) -> bool:
    """Return ``True`` if any IJK slice was provided."""
    return any(getattr(args, ax) is not None for ax in ("i", "j", "k"))


def _has_time(args: argparse.Namespace) -> bool:
    """Return ``True`` if any time/strand slice was provided."""
    return args.t is not None or args.strand_id is not None


# ---------------------------------------------------------------------------
# Solution-time filter
# ---------------------------------------------------------------------------


def _build_time_filter(
    zones: list[Any],
    t_spec: TimeSliceSpec | None,
    target_strand: int | None,
) -> set[int]:
    """Return the set of 0-based zone indices to include in the output.

    Zones with strand ID == 0 are always included.  For each eligible
    strand the zones are sorted by solution time, the ``[start, end]``
    window is applied (inclusive), then the ``skip`` stride is applied to
    the windowed list.

    Args:
        zones:         List of all ``ReadZone`` objects.
        t_spec:        Parsed time slice spec, or ``None``.
        target_strand: Restrict time slicing to this strand; ``None`` for
                       all strands > 0.

    Returns:
        Set of 0-based zone indices to write.

    """
    t_start = t_spec.start if t_spec is not None else None
    t_end = t_spec.end if t_spec is not None else None
    t_skip = t_spec.skip if t_spec is not None else None

    keep: set[int] = set()

    # Bucket zones by strand id.
    strand_buckets: dict[int, list[tuple[float, int]]] = {}
    for zi, zone in enumerate(zones):
        sid = zone.strand_id
        strand_buckets.setdefault(sid, []).append((zone.solution_time, zi))

    for sid, entries in strand_buckets.items():
        # Strand 0 is stationary data -- always keep.
        if sid == 0:
            keep.update(zi for _, zi in entries)
            continue

        # Pass through strands not targeted by slicing.
        if target_strand is not None and sid != target_strand:
            keep.update(zi for _, zi in entries)
            continue

        # Sort by solution time, apply window, apply stride.
        entries_sorted = sorted(entries, key=lambda x: x[0])

        if t_start is not None or t_end is not None:
            entries_sorted = [
                (t, zi)
                for t, zi in entries_sorted
                if (t_start is None or t >= t_start) and (t_end is None or t <= t_end)
            ]

        if t_skip is not None and t_skip > 1:
            entries_sorted = entries_sorted[::t_skip]

        keep.update(zi for _, zi in entries_sorted)

    return keep


# ---------------------------------------------------------------------------
# Zone writing helpers
# ---------------------------------------------------------------------------


def _build_protected_set(
    zones: list[Any],
    keep_indices: set[int],
) -> set[int]:
    """Return 0-based zone indices that must be written as protected zones.

    A zone is "protected" if it is excluded by the time filter but is
    referenced as a variable-sharing source by at least one zone that will
    be written.  Protected zones are written with all non-shared variables
    set passive and their strand ID forced to 0 so they act as inert grid
    anchors without contributing to the time series.

    Args:
        zones:        All source ``ReadZone`` objects.
        keep_indices: 0-based zone indices selected by the time filter.

    Returns:
        Set of 0-based zone indices to write as protected zones.

    """
    # Collect every sharing source referenced by a kept zone.
    required_sources: set[int] = set()
    for zi in keep_indices:
        for var in zones[zi].variable:
            sv = var.shared_zone  # 1-based, or None
            if sv is not None:
                required_sources.add(sv - 1)  # -> 0-based, to match keep_indices

    # A zone needs protection only if it is required but not already kept.
    return required_sources - keep_indices


def _collect_zone_arrays(
    zone: Any,
    num_vars: int,
) -> tuple[list[np.ndarray], list[Any], list[bool], list[int]]:
    """Read all variable arrays and metadata from *zone*.

    Sharing references are passed through verbatim as 1-based zone indices.
    Zone numbering is preserved in the output (every zone is written in source
    order — either full, protected, or skipped), so no remapping is needed.

    Args:
        zone:     Source ``ReadZone``.
        num_vars: Number of variables in the dataset.

    Returns:
        ``(data, locs, passive_vars, var_sharing)`` where each list has
        length ``num_vars``.  Passive and shared entries have empty arrays.

    """
    data: list[np.ndarray] = []
    locs: list[Any] = []
    passive_vars: list[bool] = []
    var_sharing: list[int] = []

    for j in range(num_vars):
        var = zone.variable[j]
        passive_vars.append(var.is_passive())
        sv = var.shared_zone  # 1-based source zone index, or None
        share_int = sv if sv is not None else 0

        var_sharing.append(share_int)
        locs.append(var.value_location)

        if var.is_passive() or share_int != 0:
            data.append(np.array([], dtype=np.float32))
            continue

        arr = var.values
        if arr is None or arr.size == 0:
            passive_vars[-1] = True
            data.append(np.array([], dtype=np.float32))
        else:
            data.append(arr)

    return data, locs, passive_vars, var_sharing


def _filter_for_writer(
    data: list[np.ndarray],
    locs: list[Any],
    passive_vars: list[bool],
    var_sharing: list[int],
) -> tuple[list[np.ndarray], list[Any]]:
    """Return only the arrays and locations the Write API expects.

    The Write API receives only active, non-shared variable arrays.

    """
    writer_data = [
        arr
        for arr, is_p, sv in zip(data, passive_vars, var_sharing, strict=False)
        if not is_p and sv == 0
    ]
    writer_locs = [
        loc
        for loc, is_p, sv in zip(locs, passive_vars, var_sharing, strict=False)
        if not is_p and sv == 0
    ]
    return writer_data, writer_locs


def _write_zone_verbatim(
    writer: Any,
    zone: Any,
    num_vars: int,
) -> None:
    """Copy a zone to *writer* without modification."""
    data, locs, passive_vars, var_sharing = _collect_zone_arrays(zone, num_vars)
    writer_data, writer_locs = _filter_for_writer(data, locs, passive_vars, var_sharing)

    zone_aux: dict[str, str] | None = (
        dict(zone.auxdata.items()) if len(zone.auxdata) > 0 else None
    )

    kw: dict[str, Any] = dict(
        title=zone.title,
        value_locations=writer_locs,
        passive_vars=passive_vars,
        var_sharing=var_sharing,
        solution_time=zone.solution_time,
        strand_id=zone.strand_id,
        aux=zone_aux,
    )

    if zone.zone_type == ZoneType.ORDERED:
        writer.write_ijk_zone(data=writer_data, **kw)
    else:
        con_sharing = zone.shared_connectivity
        writer.write_fe_zone(
            zone_type=zone.zone_type,
            data=writer_data,
            node_map=None if con_sharing else zone.node_map,
            con_sharing=con_sharing,
            **kw,
        )


def _write_zone_protected(
    writer: Any,
    zone: Any,
    num_vars: int,
) -> None:
    """Write a zone as a protected grid anchor.

    All variables that are not already passive or shared are set passive so
    that no solution data is carried into the output.  The strand ID is
    forced to 0 so the zone is treated as stationary and does not appear in
    any time-series animation.  Sharing references are preserved verbatim so
    that zones which share coordinates from this zone continue to work.

    Args:
        writer:   Open ``Write`` instance.
        zone:     Source ``ReadZone``.
        num_vars: Number of variables in the dataset.

    """
    data, locs, passive_vars, var_sharing = _collect_zone_arrays(zone, num_vars)

    # Force every non-shared variable to passive.
    passive_vars = [
        True if sv == 0 else is_p
        for is_p, sv in zip(passive_vars, var_sharing, strict=False)
    ]

    # No active non-shared data to pass to the writer.
    writer_data: list[np.ndarray] = []
    writer_locs: list[Any] = []

    zone_aux: dict[str, str] | None = (
        dict(zone.auxdata.items()) if len(zone.auxdata) > 0 else None
    )

    kw: dict[str, Any] = dict(
        title=zone.title,
        value_locations=writer_locs,
        passive_vars=passive_vars,
        var_sharing=var_sharing,
        solution_time=zone.solution_time,
        strand_id=0,  # force stationary so it is invisible in time animation
        aux=zone_aux,
    )

    if zone.zone_type == ZoneType.ORDERED:
        writer.write_ijk_zone(data=writer_data, **kw)
    else:
        con_sharing = zone.shared_connectivity
        writer.write_fe_zone(
            zone_type=zone.zone_type,
            data=writer_data,
            node_map=None if con_sharing else zone.node_map,
            con_sharing=con_sharing,
            **kw,
        )


def _slice_and_write_ordered(
    writer: Any,
    zone: Any,
    num_vars: int,
    sl_i: slice,
    sl_j: slice,
    sl_k: slice,
) -> bool:
    """Apply IJK slices to an ordered zone and write it.

    Sharing references are passed through verbatim (zone numbering is
    preserved because every zone is written in source order).

    Args:
        writer:   Open ``Write`` instance.
        zone:     Source ordered ``ReadZone``.
        num_vars: Number of variables.
        sl_i:     Python slice for I axis.
        sl_j:     Python slice for J axis.
        sl_k:     Python slice for K axis.

    Returns:
        ``True`` if the zone was written, ``False`` if the slice was empty.

    """
    from ..libtecio import ValueLocation  # local import to avoid circular

    ni, nj, nk = zone.dimensions

    # Compute output dimensions.
    ni_out = len(range(*sl_i.indices(ni)))
    nj_out = len(range(*sl_j.indices(nj)))
    nk_out = len(range(*sl_k.indices(nk)))

    if ni_out == 0 or nj_out == 0 or nk_out == 0:
        return False

    data, locs, passive_vars, var_sharing = _collect_zone_arrays(zone, num_vars)

    sliced_data: list[np.ndarray] = []
    for j, (arr, is_p, sv) in enumerate(
        zip(data, passive_vars, var_sharing, strict=False)
    ):
        if is_p or sv != 0 or arr.size == 0:
            sliced_data.append(arr)
            continue

        var = zone.variable[j]

        if var.value_location == ValueLocation.CELL_CENTERED:
            # Cell array has shape (I-1, J-1, K-1).
            # Cell c sits between nodes c and c+1.  For a nodal selection
            # with stride s, the fully-bounded cells are exactly the
            # lower-node indices of each consecutive selected-node pair:
            #   positive step -> selected_nodes[:-1]  (each is the lower node)
            #   negative step -> selected_nodes[1:]   (each is the lower node)
            # In slice terms:
            #   positive: cell slice = slice(node_start, node_stop - step, step)
            #   negative: cell slice = slice(node_start + step,
            #                               node_stop + step or None, step)
            cc_slices: list[slice] = []
            for sl, n_nodal in zip((sl_i, sl_j, sl_k), (ni, nj, nk), strict=False):
                start, stop, step = sl.indices(n_nodal)
                if step > 0:
                    cc_slices.append(slice(start, stop - step, step))
                else:
                    raw_stop = stop + step
                    cc_slices.append(
                        slice(start + step, None if raw_stop < 0 else raw_stop, step)
                    )

            grid = arr.reshape(
                max(ni - 1, 1), max(nj - 1, 1), max(nk - 1, 1), order="F"
            )
            sliced = grid[cc_slices[0], cc_slices[1], cc_slices[2]]
        else:
            grid = arr.reshape(ni, nj, nk, order="F")
            sliced = grid[sl_i, sl_j, sl_k]

        sliced_data.append(np.ascontiguousarray(sliced))

    writer_data, writer_locs = _filter_for_writer(
        sliced_data, locs, passive_vars, var_sharing
    )

    zone_aux: dict[str, str] | None = (
        dict(zone.auxdata.items()) if len(zone.auxdata) > 0 else None
    )

    writer.write_ijk_zone(
        data=writer_data,
        title=zone.title,
        value_locations=writer_locs,
        passive_vars=passive_vars,
        var_sharing=var_sharing,
        solution_time=zone.solution_time,
        strand_id=zone.strand_id,
        aux=zone_aux,
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Slice a Tecplot file along IJK indices and/or solution time.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    dst = Path(args.output)
    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    do_ijk = _has_ijk(args)
    do_time = _has_time(args)

    if not do_ijk and not do_time:
        print(
            "Warning: no slice flags provided.  Output will be a verbatim copy.",
            file=sys.stderr,
        )

    # Convert IJK specs to Python slices once.
    sl_i = _ijk_spec_to_slice(args.i)
    sl_j = _ijk_spec_to_slice(args.j)
    sl_k = _ijk_spec_to_slice(args.k)

    # Validate time skip.
    if args.t is not None and args.t.skip is not None and args.t.skip < 1:
        print(
            f"Error: time skip must be >= 1, got {args.t.skip}.",
            file=sys.stderr,
        )
        return 1

    try:
        with tecio_open(str(src), "r") as reader:
            num_vars: int = reader.num_vars
            var_names: list[str] = reader.variables

            # Materialise zone list to build the time filter before opening
            # the writer (which requires the variable list at open time for
            # SZL format).
            all_zones: list[Any] = list(reader.zone)

            # Build time keep-set.
            if do_time:
                keep_indices: set[int] = _build_time_filter(
                    zones=all_zones,
                    t_spec=args.t,
                    target_strand=args.strand_id,
                )
            else:
                keep_indices = set(range(len(all_zones)))

            # --- Summarise what will happen -----------------------------------
            print(f"Input  : {src}  ({len(all_zones)} zone(s))")
            print(f"Output : {dst}")

            if do_ijk:

                def _fmt(spec: IjkSliceSpec | None, ax: str) -> str:
                    if spec is None:
                        return f"{ax}[:]"
                    parts = [
                        str(spec.start) if spec.start is not None else "",
                        str(spec.end) if spec.end is not None else "",
                        str(spec.skip) if spec.skip is not None else "",
                    ]
                    # Trim trailing empty parts.
                    while parts and parts[-1] == "":
                        parts.pop()
                    return f"{ax}[{':'.join(parts)}]"

                print(
                    f"IJK    : "
                    f"{_fmt(args.i, 'I')}  "
                    f"{_fmt(args.j, 'J')}  "
                    f"{_fmt(args.k, 'K')}"
                )

            if do_time:
                t = args.t
                t_desc = (
                    (
                        f"[{t.start if t is not None and t.start is not None else '-inf'}"  # noqa: E501
                        f":"
                        f"{t.end if t is not None and t.end is not None else '+inf'}"
                        f":{t.skip if t is not None and t.skip is not None else '1'}]"
                    )
                    if t is not None
                    else "[::]"
                )
                print(
                    f"Time   : {t_desc}  "
                    f"(keeping {len(keep_indices)} of {len(all_zones)} zone(s))"
                )
                if args.strand_id is not None:
                    print(f"Strand : {args.strand_id} only")

            with tecio_open(
                str(dst),
                "w",
                title=reader.title,
                variables=var_names,
                file_type=reader.file_type,
            ) as writer:
                # Forward dataset-level aux data.
                if len(reader.auxdata) > 0:
                    writer.add_auxdataset_dict(dict(reader.auxdata.items()))

                # Forward variable-level aux data.
                auxvar: dict[int, dict[str, str]] = {}
                for i in range(num_vars):
                    var_aux = reader.get_var_auxdata(i + 1)
                    if len(var_aux) > 0:
                        auxvar[i + 1] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                # Zones excluded by the time filter that are sharing sources
                # for kept zones are written as "protected" anchors: all
                # non-shared variables passive, strand ID forced to 0.
                # Because every zone is written in source order (full, protected,
                # or skipped), zone numbering is identical between source and
                # output and sharing references need no remapping.
                protected_indices: set[int] = (
                    _build_protected_set(all_zones, keep_indices) if do_time else set()
                )
                if protected_indices:
                    print(
                        f"Protected (sharing sources): "
                        f"{sorted(zi + 1 for zi in protected_indices)} "
                        f"zone(s) written as passive grid anchors."
                    )

                zones_written = 0
                zones_protected = 0

                for zi, zone in enumerate(all_zones):
                    zt = zone.zone_type

                    if zt in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
                        print(
                            f"Warning: zone {zi + 1} ('{zone.title}') is "
                            f"{zt.name} -- skipping.",
                            file=sys.stderr,
                        )
                        continue

                    if zi in protected_indices:
                        # Write as a passive grid anchor to preserve sharing.
                        _write_zone_protected(writer, zone, num_vars)
                        zones_written += 1
                        zones_protected += 1
                        continue

                    if zi not in keep_indices:
                        continue

                    if do_ijk and zt == ZoneType.ORDERED:
                        written = _slice_and_write_ordered(
                            writer,
                            zone,
                            num_vars,
                            sl_i,
                            sl_j,
                            sl_k,
                        )
                        if not written:
                            print(
                                f"Warning: zone {zi + 1} ('{zone.title}') "
                                "IJK slice produced empty dimensions -- "
                                "skipping.",
                                file=sys.stderr,
                            )
                            continue

                    elif do_ijk and zt != ZoneType.ORDERED:
                        print(
                            f"Warning: zone {zi + 1} ('{zone.title}') is "
                            f"{zt.name} (unstructured) -- IJK slice ignored, "
                            "written verbatim.",
                            file=sys.stderr,
                        )
                        _write_zone_verbatim(writer, zone, num_vars)

                    else:
                        _write_zone_verbatim(writer, zone, num_vars)

                    zones_written += 1

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    msg = f"Done. {zones_written} zone(s) written to: {dst}"
    if zones_protected:
        msg += f" ({zones_protected} protected grid anchor(s))"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
