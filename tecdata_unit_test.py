#!/usr/bin/env python3
"""
Unit tests for TecData class demonstrating in-memory mutable data structure.
"""

import numpy as np

from szlio import DataType, ValueLocation, ZoneType
from tecdata import TecData


def test_empty_dataset():
    """Test creating an empty dataset."""
    print("=" * 70)
    print("TEST: Empty dataset")
    print("=" * 70)

    data = TecData()

    print(f"Title: '{data.title}'")
    print(f"Number of variables: {data.num_vars}")
    print(f"Number of zones: {data.num_zones}")
    
    # Demonstrate mutability
    data.title = "My New Title"
    data.num_zones = 5
    print(f"\nAfter mutation:")
    print(f"Title: '{data.title}'")
    print(f"Number of zones: {data.num_zones}")
    print()


def test_full_load():
    """Test loading entire dataset into memory."""
    print("=" * 70)
    print("TEST: Full load - TecData('Onera.szplt')")
    print("=" * 70)

    data = TecData("Onera.szplt")

    print(f"Title: {data.title}")
    print(f"File Type: {data.file_type.name}")
    print(f"Number of variables: {data.num_vars}")
    print(f"Number of zones: {data.num_zones}")
    
    # Check that num_vars matches length
    print(f"\nConsistency check:")
    print(f"  data.num_vars = {data.num_vars}")
    print(f"  len(data.variables) = {len(data.variables)}")
    print(f"  data.num_zones = {data.num_zones}")
    print(f"  len(data.zones) = {len(data.zones)}")
    
    print(f"\nVariables (with all metadata loaded):")
    for i, var in enumerate(data.variables):
        print(f"  [{i}] {var.name:20s} {var.data_type.name:8s} {var.value_location.name}")
        if var.auxdata:
            print(f"      Auxiliary data: {var.auxdata}")

    print(f"\nZones (with all data in memory):")
    for i, zone in enumerate(data.zones):
        print(f"  [{i}] {zone.title}")
        print(f"      Type: {zone.zone_type.name}")
        print(f"      Dimensions: {zone.dimensions}")
        print(f"      Points: {zone.num_points}")
        print(f"      Elements: {zone.num_elements}")

        # Verify all data is actually loaded (not on-demand)
        loaded_count = sum(1 for j in range(data.num_vars) if zone.has_variable_data(j))
        print(f"      Loaded {loaded_count}/{data.num_vars} variables in memory")
    print()


def test_zone_filter():
    """Test loading specific zones only."""
    print("=" * 70)
    print("TEST: Zone filter - TecData('Onera.szplt', zones=[0, 1, 2])")
    print("=" * 70)

    data = TecData("Onera.szplt", zones=[0, 1, 2])

    print(f"Title: {data.title}")
    print(f"Number of variables: {data.num_vars}")
    print(f"Number of zones: {data.num_zones} (filtered from full dataset)")

    print(f"\nLoaded zones:")
    for i, zone in enumerate(data.zones):
        print(f"  [{i}] {zone.title}")
        print(f"      Dimensions: {zone.dimensions}")
    print()


def test_var_filter_by_index():
    """Test loading specific variables by index."""
    print("=" * 70)
    print("TEST: Variable filter by index - TecData('Onera.szplt', vars=[0, 1, 2])")
    print("=" * 70)

    data = TecData("Onera.szplt", vars=[0, 1, 2])

    print(f"Title: {data.title}")
    print(f"Number of variables: {data.num_vars} (filtered)")
    print(f"Number of zones: {data.num_zones}")

    print(f"\nLoaded variables:")
    for i, var in enumerate(data.variables):
        print(f"  [{i}] {var.name}")

    print(f"\nZone data check:")
    for i, zone in enumerate(data.zones):
        print(f"  Zone [{i}] {zone.title}:")
        for j in range(data.num_vars):
            if zone.has_variable_data(j):
                var_data = zone.get_variable_data(j)
                print(
                    f"    Variable [{j}] {data.variables[j].name}: "
                    f"shape={var_data.shape}, dtype={var_data.dtype}"
                )
    print()


def test_var_filter_by_name():
    """Test loading specific variables by name."""
    print("=" * 70)
    print("TEST: Variable filter by name - TecData('Onera.szplt', vars=['X', 'Y', 'Z'])")
    print("=" * 70)

    data = TecData("Onera.szplt", vars=["X", "Y", "Z"])

    print(f"Title: {data.title}")
    print(f"Number of variables: {data.num_vars} (filtered)")
    print(f"Number of zones: {data.num_zones}")

    print(f"\nLoaded variables:")
    for i, var in enumerate(data.variables):
        print(f"  [{i}] {var.name}")

    print(f"\nFirst zone data statistics:")
    zone = data.zones[0]
    for j in range(data.num_vars):
        if zone.has_variable_data(j):
            var_data = zone.get_variable_data(j)
            print(
                f"  {data.variables[j].name}: "
                f"min={var_data.min():.6f}, max={var_data.max():.6f}"
            )
    print()


def test_zone_and_var_filter():
    """Test loading specific zones and variables."""
    print("=" * 70)
    print("TEST: Zone + Var filter - TecData('Onera.szplt', zones=[0], vars=[0, 1, 2])")
    print("=" * 70)

    data = TecData("Onera.szplt", zones=[0], vars=[0, 1, 2])

    print(f"Title: {data.title}")
    print(f"Number of variables: {data.num_vars} (filtered)")
    print(f"Number of zones: {data.num_zones} (filtered)")

    print(f"\nLoaded variables:")
    for i, var in enumerate(data.variables):
        print(f"  [{i}] {var.name}")

    print(f"\nLoaded zones:")
    for i, zone in enumerate(data.zones):
        print(f"  [{i}] {zone.title}")
        print(f"      Num points: {zone.num_points}")
    print()


def test_metadata_only():
    """Test loading metadata without zone data."""
    print("=" * 70)
    print("TEST: Metadata only - TecData('Onera.szplt', zones=[])")
    print("=" * 70)

    data = TecData("Onera.szplt", zones=[])

    print(f"Title: {data.title}")
    print(f"File Type: {data.file_type.name}")
    print(f"Number of variables: {data.num_vars}")
    print(f"Number of zones: {data.num_zones} (no zones loaded)")

    print(f"\nVariables (all metadata loaded):")
    for i, var in enumerate(data.variables):
        print(f"  [{i}] {var.name:20s} {var.data_type.name:8s} {var.value_location.name}")

    print(f"\nDataset Auxiliary data:")
    for name, value in data.auxdata.items():
        print(f"  {name}: {value}")
    
    # Demonstrate mutability
    print(f"\nDemonstrate mutability:")
    print(f"  Original num_zones: {data.num_zones}")
    data.num_zones = 10
    print(f"  After data.num_zones = 10: {data.num_zones}")
    print()


def test_create_and_write():
    """Test creating dataset from scratch and writing to file."""
    print("=" * 70)
    print("TEST: Create dataset from scratch and write")
    print("=" * 70)

    # Create empty dataset
    data = TecData()
    data.title = "Synthetic Test Data"

    # Add variables
    x_idx = data.add_variable("X")
    y_idx = data.add_variable("Y")
    p_idx = data.add_variable("Pressure")

    print(f"Created variables: {[v.name for v in data.variables]}")

    # Create a zone
    zone = data.add_zone(
        title="Test Grid",
        zone_type=ZoneType.ORDERED,
        dimensions=(10, 10, 1),
        solution_time=1.0,
        strand_id=1,
    )

    print(f"Created zone: {zone.title}, dimensions: {zone.dimensions}")

    # Generate synthetic data
    nx, ny = 10, 10
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Set coordinate data (float64)
    zone.set_variable_data(x_idx, X.ravel())
    zone.set_variable_data(y_idx, Y.ravel())

    # Set pressure field (float32 - demonstrating mixed data types)
    P = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    zone.set_variable_data(p_idx, P.ravel().astype(np.float32))

    print(f"Set data for {data.num_vars} variables")
    
    # Show what will be written
    print("\n" + data.summary())

    # Write to file - all inference happens automatically
    output_file = "test_output.szplt"
    data.write_szl(output_file)
    print(f"\nWrote dataset to: {output_file}")
    print("  FileType, ZoneType, DataType, and ValueLocation automatically inferred!")

    # Verify by reading it back
    data_check = TecData(output_file)
    print(f"\nVerification - read back from file:")
    print(f"  Title: {data_check.title}")
    print(f"  Variables: {[v.name for v in data_check.variables]}")
    print(f"  Zones: {[z.title for z in data_check.zones]}")
    
    # Check data types preserved
    zone_check = data_check.zones[0]
    for i, var in enumerate(data_check.variables):
        if zone_check.has_variable_data(i):
            var_data = zone_check.get_variable_data(i)
            print(f"  {var.name}: dtype={var_data.dtype}, shape={var_data.shape}")
    print()


def test_copy_with_filter():
    """Test loading, filtering, and re-writing dataset."""
    print("=" * 70)
    print("TEST: Load, filter, and write - automatic inference")
    print("=" * 70)

    # Load only coordinates
    data = TecData("Onera.szplt", vars=["X", "Y", "Z"])

    print(f"Loaded {data.num_vars} variables: {[v.name for v in data.variables]}")
    print(f"Loaded {data.num_zones} zones")
    
    # Show summary before writing
    print("\n" + data.summary())

    # Write filtered dataset - everything inferred automatically
    output_file = "onera_coords_only.szplt"
    data.write_szl(output_file)
    print(f"\nWrote filtered dataset to: {output_file}")
    print("  All metadata automatically inferred from TecData object!")

    # Verify
    data_check = TecData(output_file)
    print(f"\nVerification:")
    print(f"  Variables: {[v.name for v in data_check.variables]}")
    print(f"  Zones: {data_check.num_zones}")
    print()


def test_round_trip():
    """Test complete round-trip: load -> modify -> write -> verify."""
    print("=" * 70)
    print("TEST: Round-trip with automatic inference")
    print("=" * 70)

    # Load original
    original = TecData("Onera.szplt", zones=[0], vars=["X", "Y", "Z"])
    print("Loaded original dataset")
    print(f"  Variables: {[v.name for v in original.variables]}")
    print(f"  Zones: {original.num_zones}")
    
    # Get some data statistics
    zone = original.zones[0]
    x_data = zone.get_variable_data(0)
    print(f"\nOriginal X data:")
    print(f"  dtype: {x_data.dtype}")
    print(f"  min: {x_data.min():.6f}, max: {x_data.max():.6f}")

    # Write to new file (all automatic)
    output_file = "onera_roundtrip.szplt"
    original.write_szl(output_file)
    print(f"\nWrote to: {output_file}")

    # Read back
    readback = TecData(output_file)
    print(f"\nRead back from file:")
    print(f"  Variables: {[v.name for v in readback.variables]}")
    print(f"  Zones: {readback.num_zones}")

    # Verify data integrity
    zone_rb = readback.zones[0]
    x_data_rb = zone_rb.get_variable_data(0)
    print(f"\nRead-back X data:")
    print(f"  dtype: {x_data_rb.dtype}")
    print(f"  min: {x_data_rb.min():.6f}, max: {x_data_rb.max():.6f}")
    
    # Check if data matches
    if np.allclose(x_data, x_data_rb):
        print(f"\n✓ Data integrity verified - round-trip successful!")
    else:
        print(f"\n✗ Data mismatch detected!")
    print()


def test_mutability():
    """Test that all properties are mutable."""
    print("=" * 70)
    print("TEST: Mutability - all properties are read/write")
    print("=" * 70)
    
    # Load data
    data = TecData("Onera.szplt", zones=[0], vars=["X", "Y"])
    
    print("Original state:")
    print(f"  title: {data.title}")
    print(f"  num_vars: {data.num_vars}")
    print(f"  num_zones: {data.num_zones}")
    print(f"  variable[0].name: {data.variables[0].name}")
    print(f"  zone[0].title: {data.zones[0].title}")
    
    # Mutate everything
    data.title = "Modified Title"
    data.num_vars = 99  # Can be different from len(variables) if needed
    data.num_zones = 77
    data.variables[0].name = "X_Modified"
    data.variables[0].data_type = DataType.FLOAT
    data.zones[0].title = "Modified Zone"
    data.zones[0].solution_time = 3.14159
    
    print("\nAfter mutation:")
    print(f"  title: {data.title}")
    print(f"  num_vars: {data.num_vars}")
    print(f"  num_zones: {data.num_zones}")
    print(f"  variable[0].name: {data.variables[0].name}")
    print(f"  variable[0].data_type: {data.variables[0].data_type.name}")
    print(f"  zone[0].title: {data.zones[0].title}")
    print(f"  zone[0].solution_time: {data.zones[0].solution_time}")
    
    # Mutate data arrays
    zone = data.zones[0]
    x_data = zone.get_variable_data(0)
    print(f"\nOriginal X data range: [{x_data.min():.6f}, {x_data.max():.6f}]")
    
    # Modify in place
    x_data[:] = x_data * 2.0
    print(f"After x_data *= 2.0: [{x_data.min():.6f}, {x_data.max():.6f}]")
    
    # Or replace entirely
    new_data = np.random.random(zone.num_points)
    zone.set_variable_data(0, new_data)
    x_data_new = zone.get_variable_data(0)
    print(f"After replacement: [{x_data_new.min():.6f}, {x_data_new.max():.6f}]")
    print()


def main():
    """Run all tests."""
    tests = [
        test_empty_dataset,
        test_full_load,
        test_zone_filter,
        test_var_filter_by_index,
        test_var_filter_by_name,
        test_zone_and_var_filter,
        test_metadata_only,
        test_mutability,
        test_create_and_write,
        test_copy_with_filter,
        test_round_trip,
    ]

    print("\n" + "=" * 70)
    print("TECDATA UNIT TESTS")
    print("=" * 70 + "\n")

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}\n")
            import traceback

            traceback.print_exc()
            print()

    print("=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
