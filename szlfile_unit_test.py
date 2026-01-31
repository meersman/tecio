#!/usr/bin/env python3

import szlfile

# Create szl reader object
szl = szlfile.Read("Onera.szplt")

print(f"file type: {szl.type}")
print(f"dataset title: {szl.title}")
print(f"num vars: {szl.num_vars}")
print(f"num zones: {szl.num_zones}")
print(f"aux items count: {szl.num_auxdata_items}")

# Test dataset-level auxiliary data
print("\n=== Dataset Auxiliary Data ===")
if len(szl.auxdata) > 0:
    print(f"Dataset aux data: {dict(szl.auxdata)}")
    for name, value in szl.auxdata.items():
        print(f"  {name}: {value}")
else:
    print("No dataset auxiliary data")

# Test variable-level auxiliary data
print("\n=== Variable Auxiliary Data ===")
for i in range(szl.num_vars):
    var_aux = szl.get_var_auxdata(i + 1)
    if len(var_aux) > 0:
        print(f"Variable {i+1} aux data: {dict(var_aux)}")
        for name, value in var_aux.items():
            print(f"  {name}: {value}")

# Test zones
print("\n=== Zone Information ===")
for i in range(szl.num_zones):
    zone = szl.zones[i]
    print(f"\nZone {i+1}:")
    print(f"  type: {zone.type}")
    print(f"  title: {zone.title}")
    print(f"  is enabled: {zone.is_enabled()}")
    print(f"  solution time: {zone.solution_time}")
    print(f"  strand id: {zone.strand_id}")

    # Test zone-level auxiliary data
    if len(zone.auxdata) > 0:
        print(f"  Zone auxiliary data:")
        for name, value in zone.auxdata.items():
            print(f"    {name}: {value}")

            # Demonstrate type conversion methods
            int_val = zone.auxdata.as_int(name)
            float_val = zone.auxdata.as_float(name)
            bool_val = zone.auxdata.as_bool(name)

            if int_val is not None:
                print(f"      as int: {int_val}")
            if float_val is not None:
                print(f"      as float: {float_val}")
            if bool_val is not None:
                print(f"      as bool: {bool_val}")

    # Test variables
    for j in range(szl.num_vars):
        var = zone.variables[j]
        print(f"  Variable {j+1}:")
        print(f"    name: {var.name}")
        print(f"    type: {var.type}")
        print(f"    is enabled: {var.is_enabled()}")
        print(f"    location: {var.value_location}")
        print(f"    is passive: {var.is_passive()}")
        print(f"    shared zone: {var.shared_zone}")
        print(f"    num values: {var.num_values}")

        # Get first 10 values or all if fewer than 10
        num_to_show = min(10, var.num_values)
        values = var.get_values((1, num_to_show + 1))
        print(f"    first {num_to_show} values: {values}")

    # Show node map for FE zones
    if zone.type != szlfile.ZoneType.ORDERED:
        print(f"  Node map shape: {zone.node_map.shape}")
        print(f"  First element connectivity: {zone.node_map[0]}")
