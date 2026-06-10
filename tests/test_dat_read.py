#!/usr/bin/env python3
"""Lite test for datfile read functions."""

import textwrap
from pathlib import Path

import numpy as np

import tecio
from tecio.libtecio import DataPacking, ValueLocation, ZoneType


def test_dat_read():
    """Print out all headers and data fields for test dat file."""
    test_dir = Path(tecio.__file__).parent.parent
    input_file = test_dir / "tests" / "Onera.dat"
    input_file = input_file.as_posix()

    # Set numpy array print option
    np.set_printoptions(threshold=100)

    # Create dat reader object
    dat = tecio.open(input_file, "r")

    print("\nFile Record")
    print("=" * 70)
    print(f"File Type         : {dat.file_type}")
    print(f"Dataset Title     : {dat.title}")
    print(f"Num Vars          : {dat.num_vars}")
    print(f"Variables         : {dat.variables}")
    print(f"Num Zones         : {dat.num_zones}")
    print(f"Dataset Aux Items : {dat.num_auxdata_items}")

    # Print dataset-level auxiliary data if available
    print("\n\nDataset Auxiliary Data")
    print("-" * 70)
    if len(dat.auxdata) > 0:
        print(f"Dataset Aux Data  : {dict(dat.auxdata)}")
        for name, value in dat.auxdata.items():
            print(f"  {name:>15} : {value}")

    # Print variable-level auxiliary data if available
    print("\n\nVariable Auxiliary Data")
    print("-" * 70)
    for i in range(dat.num_vars):
        var_aux = dat.get_var_auxdata(i + 1)
        if len(var_aux) > 0:
            print(f"Var {i + 1:3} Aux Data  : {dict(var_aux)}")
            for name, value in var_aux.items():
                print(f"  {name:>15} : {value}")

    # Print zone record
    print("\n\nZone Record")
    print("-" * 70)
    for i in range(dat.num_zones):
        zone = dat.zone[i]
        print(f"\nZone {i + 1:3}")
        print(f"  Title           : {zone.title}")
        print(f"  Zone Type       : {zone.zone_type}")
        print(f"  Datapacking     : {zone.datapacking}")
        print(f"  Is Enabled      : {zone.is_enabled()}")
        print(f"  Solution Time   : {zone.solution_time}")
        print(f"  Strand ID       : {zone.strand_id}")

        # Print zone-level auxiliary data
        if len(zone.auxdata) > 0:
            print(f"  Zone Aux Data   : {dict(zone.auxdata)}")
            for name, value in zone.auxdata.items():
                print(f"  {name:>15} : {value}")

        # Print variable record
        for j in range(dat.num_vars):
            var = zone.variable[j]
            print(f"  Variable {j + 1:3}")
            print(f"    Name          : {var.name}")
            print(f"    Data Type     : {var.data_type}")
            print(f"    Is Enabled    : {var.is_enabled()}")
            print(f"    Location      : {var.value_location}")
            print(f"    Is Passive    : {var.is_passive()}")
            print(f"    Shared Zone   : {var.shared_zone}")
            print(f"    Num Values    : {var.num_values}")

            # Get first 100 values or all if fewer than 100
            value_str = np.array2string(
                var.values, prefix="    Values        : ", separator=", "
            )
            print(f"    Values        : {value_str}")

        # Show node map for FE zones
        if zone.zone_type != ZoneType.ORDERED:
            print(f"  Node Map Shape  : {zone.node_map.shape}")
            value_str = np.array2string(
                zone.node_map, prefix="  Connectivity    : ", separator=", "
            )
            print(f"  Connectivity    : {value_str}")


def test_dat_read_point_ordered():
    """Read a hand-authored DATAPACKING=POINT ordered zone from a temp file.

    Exercises the POINT reader path directly with a minimal DAT string that
    mimics what third-party tools write.  Values are checked exactly since the
    test data is small enough for exact ASCII representation.
    """
    import os
    import tempfile

    # Minimal POINT-format DAT with three variables and four nodes.
    # Row order: one row per node, all variable values on the same line.
    point_dat = textwrap.dedent("""\
        TITLE     = "point_test"
        VARIABLES = "x" "y" "p"
        ZONE T="Zone1"
         STRANDID=0, SOLUTIONTIME=0.0
         I=4, J=1, K=1, ZONETYPE=Ordered
         DATAPACKING=POINT
        0.0\t1.0\t10.0
        1.0\t2.0\t20.0
        2.0\t3.0\t30.0
        3.0\t4.0\t40.0
        """)
    expected_x = np.array([0.0, 1.0, 2.0, 3.0])
    expected_y = np.array([1.0, 2.0, 3.0, 4.0])
    expected_p = np.array([10.0, 20.0, 30.0, 40.0])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(point_dat)
        tmp_path = fh.name

    try:
        dat = tecio.open(tmp_path, "r")

        print("\nPOINT Ordered Zone Read")
        print("=" * 70)
        assert dat.num_vars == 3, f"Expected 3 vars, got {dat.num_vars}"
        assert dat.num_zones == 1, f"Expected 1 zone, got {dat.num_zones}"

        zone = dat.zone[0]
        assert zone.zone_type == ZoneType.ORDERED
        assert zone.datapacking == DataPacking.POINT
        assert zone.dimensions == (4, 1, 1)

        np.testing.assert_allclose(zone.variable[0].values.ravel(), expected_x)
        np.testing.assert_allclose(zone.variable[1].values.ravel(), expected_y)
        np.testing.assert_allclose(zone.variable[2].values.ravel(), expected_p)

        print(f"Variables         : {dat.variables}")
        print(f"Zone type         : {zone.zone_type}")
        print(f"x values          : {zone.variable[0].values.ravel()}")
        print(f"y values          : {zone.variable[1].values.ravel()}")
        print(f"p values          : {zone.variable[2].values.ravel()}")
        print("PASS")

    finally:
        os.unlink(tmp_path)


def test_dat_read_point_fe():
    """Read a hand-authored DATAPACKING=POINT FE triangle zone from a temp file.

    Verifies that the POINT reader correctly separates the nodal data rows
    from the connectivity block that follows.
    """
    import os
    import tempfile

    # FETRIANGLE with 4 nodes, 2 elements, 3 variables.
    # Data rows: one per node.  Connectivity rows: one per element (1-based).
    point_dat = textwrap.dedent("""\
        TITLE     = "point_fe_test"
        VARIABLES = "x" "y" "c"
        ZONE T="FE_Tri_Point"
         STRANDID=0, SOLUTIONTIME=0.0
         Nodes=4, Elements=2, ZONETYPE=FETriangle
         DATAPACKING=POINT
        0.0\t0.0\t0.1
        1.0\t0.0\t0.2
        1.0\t1.0\t0.3
        0.0\t1.0\t0.4
        1 2 3
        1 3 4
        """)
    expected_x = np.array([0.0, 1.0, 1.0, 0.0])
    expected_c = np.array([0.1, 0.2, 0.3, 0.4])
    expected_conn = np.array([[1, 2, 3], [1, 3, 4]], dtype=np.int64)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(point_dat)
        tmp_path = fh.name

    try:
        dat = tecio.open(tmp_path, "r")

        print("\nPOINT FE Triangle Zone Read")
        print("=" * 70)
        assert dat.num_vars == 3
        assert dat.num_zones == 1

        zone = dat.zone[0]
        assert zone.zone_type == ZoneType.FETRIANGLE
        assert zone.datapacking == DataPacking.POINT
        assert zone.num_nodes == 4
        assert zone.num_elements == 2

        np.testing.assert_allclose(zone.variable[0].values, expected_x)
        np.testing.assert_allclose(zone.variable[2].values, expected_c)
        np.testing.assert_array_equal(zone.node_map, expected_conn)

        print(f"Zone type         : {zone.zone_type}")
        print(f"Num nodes         : {zone.num_nodes}")
        print(f"Num elements      : {zone.num_elements}")
        print(f"x values          : {zone.variable[0].values}")
        print(f"c values          : {zone.variable[2].values}")
        print(f"Node map          :\n{zone.node_map}")
        print("PASS")

    finally:
        os.unlink(tmp_path)


def test_dat_read_point_mixed_cc():
    """Read a POINT zone with mixed nodal and cell-centred variables.

    Verifies the two-section layout: nodal rows (one per node, nodal vars
    only) followed by CC rows (one per element, CC vars only).
    """
    import os
    import tempfile

    # Ordered 1-D zone: I=4 nodes, 3 cells.
    # Variables: x (nodal), p (nodal), rho (cell-centred).
    # Nodal section: 4 rows × 2 columns.
    # CC section:    3 rows × 1 column.
    point_dat = textwrap.dedent("""\
        TITLE     = "point_cc_test"
        VARIABLES = "x" "p" "rho"
        ZONE T="CC_Point"
         STRANDID=0, SOLUTIONTIME=0.0
         I=4, J=1, K=1, ZONETYPE=Ordered
         DATAPACKING=POINT
         VARLOCATION=([3]=CELLCENTERED)
        0.0\t1.0
        1.0\t2.0
        2.0\t3.0
        3.0\t4.0
        10.0
        20.0
        30.0
        """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dat", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(point_dat)
        tmp_path = fh.name

    try:
        dat = tecio.open(tmp_path, "r")

        print("\nPOINT Mixed Nodal + CC Zone Read")
        print("=" * 70)
        assert dat.num_vars == 3
        zone = dat.zone[0]
        assert zone.datapacking == DataPacking.POINT

        x_var = zone.variable[0]
        p_var = zone.variable[1]
        rho_var = zone.variable[2]

        assert x_var.value_location == ValueLocation.NODAL
        assert p_var.value_location == ValueLocation.NODAL
        assert rho_var.value_location == ValueLocation.CELL_CENTERED

        np.testing.assert_allclose(x_var.values.ravel(), [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(p_var.values.ravel(), [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(rho_var.values.ravel(), [10.0, 20.0, 30.0])

        print(f"x   (nodal, 4)    : {x_var.values.ravel()}")
        print(f"p   (nodal, 4)    : {p_var.values.ravel()}")
        print(f"rho (CC, 3)       : {rho_var.values.ravel()}")
        print("PASS")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_dat_read()
    test_dat_read_point_ordered()
    test_dat_read_point_fe()
    test_dat_read_point_mixed_cc()
