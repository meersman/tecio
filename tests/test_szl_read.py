#!/usr/bin/env python3
"""Lite test for szlfile read functions."""

from pathlib import Path

import numpy as np

from tecio import szl
from tecio.libtecio import ZoneType


test_dir = Path(tecio.__file__).parent.parent
input_file = test_dir / "tests" / "Onera.szplt"
input_file = input_file.as_posix()

# Set numpy array print option
np.set_printoptions(threshold=100)

# Create szl reader object
szl = szl.Read(input_file)

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
    if zone.zone_type != ZoneType.ORDERED:
        print(f"  Node Map Shape  : {zone.node_map.shape}")
        value_str = np.array2string(
            zone.node_map,
            prefix="  Connectivity    : ", separator=", "
        )
        print(f"  Connectivity    : {value_str}")
