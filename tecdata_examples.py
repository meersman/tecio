#!/usr/bin/env python3
"""
Examples of using TecData for various workflows.
"""

from tecdata import TecData, TecZone, TecVariable
from szlio import ZoneType, ValueLocation
import numpy as np


# =============================================================================
# Example 1: Load from SZL file and normalize pressure
# =============================================================================

def example_normalize_pressure():
    """Load SZL file and normalize pressure by reference value."""
    print("=" * 70)
    print("Example 1: Normalize Pressure")
    print("=" * 70)
    
    # Load all data into memory
    data = TecData.from_szl_file("flow.szplt", load_data=True)
    
    print(f"Loaded: {data}")
    print(f"Variables: {[v.name for v in data.variables]}")
    
    # Find pressure variable
    p_idx = data.get_variable_index("Pressure")
    if p_idx == -1:
        print("Pressure variable not found!")
        return
    
    # Normalize by reference pressure
    p_ref = 101325.0  # Pa
    data.normalize_variable(p_idx, p_ref)
    
    print(f"Normalized pressure by {p_ref} Pa")
    
    # Optionally rename variable
    data.variables[p_idx].name = "Pressure_Nondimensional"
    
    # TODO: Write back to file
    # data.write_szl("flow_normalized.szplt")


# =============================================================================
# Example 2: Compute Mach number from velocity and temperature
# =============================================================================

def example_compute_mach():
    """Compute Mach number from velocity components and temperature."""
    print("\n" + "=" * 70)
    print("Example 2: Compute Mach Number")
    print("=" * 70)
    
    # Load file
    data = TecData.from_szl_file("flow.szplt", load_data=True)
    
    # Find velocity and temperature variables
    u_idx = data.get_variable_index("U-Velocity")
    v_idx = data.get_variable_index("V-Velocity")
    w_idx = data.get_variable_index("W-Velocity")
    t_idx = data.get_variable_index("Temperature")
    
    if any(idx == -1 for idx in [u_idx, v_idx, w_idx, t_idx]):
        print("Required variables not found!")
        return
    
    # Add Mach number variable
    mach_idx = data.add_variable("Mach", ValueLocation.NODAL)
    
    # Compute for each zone
    gamma = 1.4
    R = 287.05  # J/(kg·K) for air
    
    for zone in data.zones:
        u = zone.get_variable_data(u_idx)
        v = zone.get_variable_data(v_idx)
        w = zone.get_variable_data(w_idx)
        T = zone.get_variable_data(t_idx)
        
        velocity_mag = np.sqrt(u**2 + v**2 + w**2)
        speed_of_sound = np.sqrt(gamma * R * T)
        mach = velocity_mag / speed_of_sound
        
        zone.set_variable_data(mach_idx, mach)
    
    print(f"Computed Mach number for {len(data.zones)} zones")
    
    # Check results
    for i, zone in enumerate(data.zones):
        mach_data = zone.get_variable_data(mach_idx)
        print(f"  Zone {i+1}: Mach range [{mach_data.min():.3f}, {mach_data.max():.3f}]")


# =============================================================================
# Example 3: Combine multiple files and compute recovery temperature
# =============================================================================

def example_combine_files():
    """Combine zones from multiple files and compute derived quantities."""
    print("\n" + "=" * 70)
    print("Example 3: Combine Multiple Files")
    print("=" * 70)
    
    # Create empty dataset
    combined = TecData(title="Combined Surface Heating Analysis")
    
    # Load multiple files (assuming they have same variables)
    file_paths = ["surface1.szplt", "surface2.szplt", "surface3.szplt"]
    
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        data = TecData.from_szl_file(file_path, load_data=True)
        
        # First file: copy variable definitions
        if len(combined.variables) == 0:
            combined.variables = data.variables.copy()
        
        # Add zones from this file
        combined.zones.extend(data.zones)
    
    print(f"Combined {len(combined.zones)} zones from {len(file_paths)} files")
    
    # Compute recovery temperature
    q_idx = combined.get_variable_index("HeatFlux")
    T_idx = combined.get_variable_index("Temperature")
    
    if q_idx != -1 and T_idx != -1:
        recovery_idx = combined.add_variable("RecoveryTemperature")
        h = 1000.0  # Heat transfer coefficient (example)
        
        for zone in combined.zones:
            q = zone.get_variable_data(q_idx)
            T = zone.get_variable_data(T_idx)
            T_recovery = T + q / h
            zone.set_variable_data(recovery_idx, T_recovery)
        
        print(f"Computed recovery temperature for all zones")


# =============================================================================
# Example 4: Create dataset from scratch
# =============================================================================

def example_create_from_scratch():
    """Create a Tecplot dataset from scratch with synthetic data."""
    print("\n" + "=" * 70)
    print("Example 4: Create Dataset from Scratch")
    print("=" * 70)
    
    # Create empty dataset
    data = TecData(title="Synthetic Test Data")
    
    # Add variables
    x_idx = data.add_variable("X")
    y_idx = data.add_variable("Y")
    z_idx = data.add_variable("Z")
    p_idx = data.add_variable("Pressure")
    
    # Create an ordered zone (10 x 10 x 10 grid)
    zone = data.add_zone(
        title="Structured Grid",
        zone_type=ZoneType.ORDERED,
        dimensions=(10, 10, 10)
    )
    
    # Generate synthetic data
    nx, ny, nz = 10, 10, 10
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten for zone data
    zone.set_variable_data(x_idx, X.ravel())
    zone.set_variable_data(y_idx, Y.ravel())
    zone.set_variable_data(z_idx, Z.ravel())
    
    # Synthetic pressure field: P = sin(2πx) * cos(2πy) * exp(-z)
    P = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.exp(-Z)
    zone.set_variable_data(p_idx, P.ravel())
    
    print(f"Created {zone}")
    print(f"  Points: {zone.num_points}")
    print(f"  Elements: {zone.num_elements}")


# =============================================================================
# Example 5: Partial loading for large files
# =============================================================================

def example_partial_load():
    """Load only specific zones and variables for memory efficiency."""
    print("\n" + "=" * 70)
    print("Example 5: Partial Loading")
    print("=" * 70)
    
    # Load only zones 1-3 and variables 1, 2, 5
    data = TecData.from_szl_file(
        "large_file.szplt",
        load_data=True,
        zones=[1, 2, 3],
        variables=[1, 2, 5]
    )
    
    print(f"Loaded subset: {data}")
    print(f"Variables: {[v.name for v in data.variables]}")
    print(f"Zones: {len(data.zones)}")


# =============================================================================
# Example 6: Using velocity magnitude helper
# =============================================================================

def example_velocity_magnitude():
    """Compute velocity magnitude using built-in helper."""
    print("\n" + "=" * 70)
    print("Example 6: Velocity Magnitude")
    print("=" * 70)
    
    data = TecData.from_szl_file("flow.szplt", load_data=True)
    
    # Find velocity components
    u_idx = data.get_variable_index("U-Velocity")
    v_idx = data.get_variable_index("V-Velocity")
    w_idx = data.get_variable_index("W-Velocity")
    
    if any(idx == -1 for idx in [u_idx, v_idx, w_idx]):
        print("Velocity components not found!")
        return
    
    # Compute magnitude
    vmag_idx = data.compute_magnitude(
        (u_idx, v_idx, w_idx),
        result_name="Velocity_Magnitude"
    )
    
    print(f"Computed velocity magnitude at variable index {vmag_idx}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run examples (comment out as needed)
    
    # example_normalize_pressure()
    # example_compute_mach()
    # example_combine_files()
    example_create_from_scratch()
    # example_partial_load()
    # example_velocity_magnitude()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
