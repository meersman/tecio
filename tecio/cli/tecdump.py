r"""Dump the full contents of a Tecplot data file to standard output.

Inspecting the contents of a binary Tecplot file (variable names, zone metadata,
auxiliary data, and raw nodal or element values) ordinarily requires either opening the
file in Tecplot itself or writing a dedicated script against the TecIO API.  For a quick
sanity check during a simulation or post-processing workflow, ``tecdump`` serializes
every record in the file to plain text, providing a complete and human-readable
representation of the file contents directly in the terminal.  The output is structured
hierarchically: file header and auxiliary data first, followed by per-zone headers and,
optionally, the underlying variable and connectivity arrays.

By default all zones and variables are printed. However, the scope can be narrowed to a
single zone or variable using the ``-zone`` and ``-variable`` flags, or the variable
arrays can be suppressed entirely to focus on structural metadata.

:Positional Arguments:
    ``filename``
        Path to the Tecplot binary file (``.plt`` or ``.szplt``)
        to inspect.

:Options:
    ``--ignore-zones``
        Print only the file-level header and auxiliary data, then exit.  Zone records
        and variable arrays are suppressed entirely. Useful for confirming file-level
        metadata such as solver, strand IDs, or global auxiliary entries without the
        overhead of parsing zone data.

    ``--ignore-vars``
        Print the file header and all zone headers, then exit. Variable arrays and
        connectivity tables are suppressed. Use this to survey zone names, types, and
        dimensions across a large file without printing any data.

    ``-zone INDEX``
        Restrict output to the zone at the given one-based index. All other zones are
        skipped. If omitted, every zone is printed.

    ``-variable INDEX``
        Restrict variable array output to the variable at the given one-based index. All
        other variable arrays are skipped. If omitted, every variable is printed.

    ``-maxvals INT``
        Maximum number of values to print from any variable or connectivity array before
        truncating with a summary line.  Defaults to a small number suitable for a quick
        preview; set to a large value (e.g. ``1000000``) to print an array in full.

:Returns:
    Output is written to standard output. Exit code is ``0`` on success and non-zero if
    the file cannot be read or an invalid index is supplied.

Examples:
    Print the complete contents of a file, truncating long arrays at the default limit::

        $ tecdump flow.plt

    Print only file and zone headers with no variable data::

        $ tecdump --ignore-vars flow.szplt

    Inspect a single zone/variable combination in full::

        $ tecdump -zone 1 -variable 3 -maxvals 1000000 flow.plt

See Also:
    * :mod:`tecio.cli.tecstats` - Compute per-zone min, max, and mean statistics without
      printing raw values.
    * :mod:`tecio.cli.tecfix`: Rewrite a file with invalid variable
      arrays set to passive once bad values have been identified.
"""

import argparse
from collections.abc import Sequence

import numpy as np

from .. import open as tecio_open
from ..libtecio import ZoneType


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump all contents of a Tecplot file.",
        epilog=(
            "Example usage:\n"
            "  Print all contents of file\n"
            "    $ tecdump <input file>\n"
            "  Print all values from zone 1 variable 3 (set maxval to large number)\n"
            "    $ tecdump -zone 1 -variable 3 -maxvals 1e6 <input file>\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument(
        "filename",
        help="Tecplot file to print all contents.",
        type=str,
    )
    parser.add_argument(
        "--ignore-zones",
        help="Print only file header and exit. No zone or variable records printed.",
        action="store_false",
        default=True,
        dest="print_zones",
    )
    parser.add_argument(
        "--ignore-vars",
        help=(
            "Print file header and zone headers then exits. No variable records "
            "printed."
        ),
        action="store_false",
        default=True,
        dest="print_vars",
    )
    parser.add_argument(
        "-zone",
        help="Zone number to dump data from. Default is all zones.",
        type=int,
        default=None,
        metavar="INDEX",
    )
    parser.add_argument(
        "-variable",
        help="Variable number to dump data from. Default is all variables.",
        type=int,
        default=None,
        metavar="INDEX",
    )
    parser.add_argument(
        "-maxvals",
        help=(
            "Max number of values to print for variable and connectivity arrays before "
            "truncating"
        ),
        type=int,
        default=100,
        metavar="INT",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print all available info for the given SZPLT file."""
    # Get command line input
    args = _parse_args(argv)

    # Set numpy array print option
    np.set_printoptions(threshold=args.maxvals)

    # Create tec reader object
    with tecio_open(args.filename, "r") as tec:
        print("\nFile Record")
        print("=" * 70)
        print(f"File Type         : {tec.file_type}")
        print(f"Dataset Title     : {tec.title}")
        print(f"Num Vars          : {tec.num_vars}")
        print(f"Variables         : {tec.variables}")
        print(f"Num Zones         : {tec.num_zones}")
        print(f"Dataset Aux Items : {tec.num_auxdata_items}")

        # Print dataset-level auxiliary data if available
        print("\n\nDataset Auxiliary Data")
        print("-" * 70)
        if len(tec.auxdata) > 0:
            for name, value in tec.auxdata.items():
                print(f"  {name:>15} : {value}")

        # Print variable-level auxiliary data if available
        print("\n\nVariable Auxiliary Data")
        print("-" * 70)
        for i in range(tec.num_vars):
            var_aux = tec.get_var_auxdata(i + 1)
            if len(var_aux) > 0:
                print(f"Var {i + 1:3} Aux Data  : {dict(var_aux)}")
                for name, value in var_aux.items():
                    print(f"  {name:>15} : {value}")

        # Print zone record
        if args.print_zones:
            print("\n\nZone Record")
            print("-" * 70)
            for i in range(tec.num_zones):
                if (args.zone is None) or (i + 1 == args.zone):
                    zone = tec.zone[i]
                    print(f"\nZone {i + 1:3}")
                    print(f"  Title           : {zone.title}")
                    print(f"  Zone Type       : {zone.zone_type}")
                    if zone.zone_type == ZoneType.ORDERED:
                        print(f"  I,J,K           : {zone.dimensions}")
                    else:
                        print(f"  Num Nodes       : {zone.num_nodes}")
                        print(f"  Num Elements    : {zone.num_elements}")
                    print(f"  Is Enabled      : {zone.is_enabled()}")
                    print(f"  Solution Time   : {zone.solution_time}")
                    print(f"  Strand ID       : {zone.strand_id}")

                    # Print zone-level auxiliary data
                    if len(zone.auxdata) > 0:
                        print(f"  Zone Aux Data   : {dict(zone.auxdata)}")
                        for name, value in zone.auxdata.items():
                            print(f"  {name:>15} : {value}")

                    # Show node map for FE zones
                    if zone.zone_type != ZoneType.ORDERED:
                        print(f"  Nodes Per Cell  : {zone.nodes_per_cell}")
                        print(f"  Node Map Shape  : {zone.node_map.shape}")
                        value_str = np.array2string(
                            zone.node_map, prefix="  Connectivity    : ", separator=", "
                        )
                        print(f"  Connectivity    : {value_str}")

                    # Print variable record
                    if args.print_vars:
                        for j in range(tec.num_vars):
                            if (args.variable is None) or (j + 1 == args.variable):
                                var = zone.variable[j]
                                print(f"  Variable {j + 1:3}")
                                print(f"    Name          : {var.name}")
                                print(f"    Data Type     : {var.data_type}")
                                print(f"    Location      : {var.value_location}")
                                print(f"    Is Enabled    : {var.is_enabled()}")
                                print(f"    Is Passive    : {var.is_passive()}")
                                print(f"    Shared Zone   : {var.shared_zone}")
                                print(f"    Num Values    : {var.num_values}")

                                # Check if variable is shared or passive (no data)
                                if (var.shared_zone is None) and (not var.is_passive()):
                                    print(f"    Array shape   : {var.values.shape}")
                                    # Get first 100 values or all if fewer than 100
                                    value_str = np.array2string(
                                        var.values,
                                        prefix="    Values        : ",
                                        separator=", ",
                                    )
                                    print(f"    Values        : {value_str}")

    return 0


if __name__ == "__main__":
    main()
