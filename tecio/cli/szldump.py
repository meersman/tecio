"""Command line interface to dump all contents of a szlfile.

(like a super verbose szlpltview)
"""

import argparse

import numpy as np

from .. import szlfile


def main():
    """Print all available info for the given SZPLT file."""
    # Get command line input
    parser = argparse.ArgumentParser(description="Dump all contents of a SZPLT file.")
    parser.add_argument(
        "filename",
        help="SZPLT file to print all contents.",
        type=str,
    )
    parser.add_argument(
        "--ignore-zones",
        help="Print only file header and exit. No zone or variable records printed.",
        action="store_false",
        default=True,
        dest="print_zones"
    )
    parser.add_argument(
        "--ignore-vars",
        help=(
            "Print file header and zone headers then exits. No variable records "
            "printed."
        ),
        action="store_false",
        default=True,
        dest="print_vars"
    )
    parser.add_argument(
        "-maxvals",
        help=(
            "Max number of values to print for variable and connectivity arrays before "
            "truncating"
        ),
        type=int,
        default=100,
    )

    args = parser.parse_args()

    # Set numpy array print option
    np.set_printoptions(threshold=args.maxvals)

    # Create szl reader object
    szl = szlfile.Read(args.filename)

    print("\nFile Record")
    print("="*70)
    print(f"File Type         : {szl.file_type}")
    print(f"Dataset Title     : {szl.title}")
    print(f"Num Vars          : {szl.num_vars}")
    print(f"Variables         : {szl.var_list}")
    print(f"Num Zones         : {szl.num_zones}")
    print(f"Dataset Aux Items : {szl.num_auxdata_items}")

    # Print dataset-level auxiliary data if available
    print("\n\nDataset Auxiliary Data")
    print("-"*70)
    if len(szl.auxdata) > 0:
        print(f"Dataset Aux Data  : {dict(szl.auxdata)}")
        for name, value in szl.auxdata.items():
            print(f"  {name:>15} : {value}")

    # Print variable-level auxiliary data if available
    print("\n\nVariable Auxiliary Data")
    print("-"*70)
    for i in range(szl.num_vars):
        var_aux = szl.get_var_auxdata(i+1)
        if len(var_aux) > 0:
            print(f"Var {i+1:3} Aux Data  : {dict(var_aux)}")
            for name, value in var_aux.items():
                print(f"  {name:>15} : {value}")

    # Print zone record
    if args.print_zones:
        print("\n\nZone Record")
        print("-"*70)
        for i in range(szl.num_zones):
            zone = szl.zones[i]
            print(f"\nZone {i+1:3}")
            print(f"  Title           : {zone.title}")
            print(f"  Zone Type       : {zone.zone_type}")
            print(f"  Is Enabled      : {zone.is_enabled()}")
            print(f"  Solution Time   : {zone.solution_time}")
            print(f"  Strand ID       : {zone.strand_id}")

            # Print zone-level auxiliary data
            if len(zone.auxdata) > 0:
                print(f"  Zone Aux Data   : {dict(zone.auxdata)}")
                for name, value in zone.auxdata.items():
                    print(f"  {name:>15} : {value}")

            # Print variable record
            if args.print_vars:
                for j in range(szl.num_vars):
                    var = zone.variables[j]
                    print(f"  Variable {j+1:3}")
                    print(f"    Name          : {var.name}")
                    print(f"    Data Type     : {var.data_type}")
                    print(f"    Is Enabled    : {var.is_enabled()}")
                    print(f"    Location      : {var.value_location}")
                    print(f"    Is Passive    : {var.is_passive()}")
                    print(f"    Shared Zone   : {var.shared_zone}")
                    print(f"    Num Values    : {var.num_values}")

                    # Get first 100 values or all if fewer than 100
                    value_str = np.array2string(
                        var.values,
                        prefix="    Values        : ", separator=", "
                    )
                    print(f"    Values        : {value_str}")

            # Show node map for FE zones
            if zone.zone_type != szlfile.ZoneType.ORDERED:
                print(f"  Node Map Shape  : {zone.node_map.shape}")
                value_str = np.array2string(
                    zone.node_map,
                    prefix="  Connectivity    : ", separator=", "
                )
                print(f"  Connectivity    : {value_str}")


if __name__ == "__main__":
    main()
