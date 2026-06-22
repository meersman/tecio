#!/usr/bin/env python3
"""pytest tests for :class:`tecio.dat.Read` -- DAT ASCII reader.

Pattern per test:
    1. Open an existing or hand-authored ``.dat`` file via tecio.open(..., "r")
    2. Assert on metadata, container types (ZoneList / VariableList), and
       variable values

Run directly:

    $ python tests/test_dat_read.py -v

Keep output files for Tecplot 360 inspection:

    $ python tests/test_dat_read.py -v --keep-files
"""

# ruff: noqa: E501, SIM117

import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import tecio
from tecio.libtecio import DataPacking, ValueLocation, ZoneType


# ===========================================================================
# Full-file dump
# ===========================================================================


class TestReadDump:
    """Dump every header and data field for the shared Onera DAT fixture."""

    def test_read_dump(self, tests_dir: Path) -> None:
        """Print out all headers and data fields for the Onera DAT fixture.

        Demonstrates:
        - Top-level metadata: file_type, title, num_vars, variables, num_zones
        - Dataset- and variable-level auxiliary data access
        - Per-zone metadata: title, zone_type, datapacking, solution_time,
          strand_id, and zone-level aux data
        - Per-variable metadata: name, data_type, value_location,
          is_enabled/is_passive, shared_zone, num_values, and the raw array
        - Connectivity (node_map) for FE zones
        """
        np.set_printoptions(threshold=100)

        input_file = tests_dir / "Onera.dat"
        dat = tecio.open(str(input_file), "r")

        print("\nFile Record")
        print("=" * 70)
        print(f"File Type         : {dat.file_type}")
        print(f"Dataset Title     : {dat.title}")
        print(f"Num Vars          : {dat.num_vars}")
        print(f"Variables         : {dat.variables}")
        print(f"Num Zones         : {dat.num_zones}")
        print(f"Dataset Aux Items : {dat.num_auxdata_items}")

        print("\n\nDataset Auxiliary Data")
        print("-" * 70)
        if len(dat.auxdata) > 0:
            print(f"Dataset Aux Data  : {dict(dat.auxdata)}")
            for name, value in dat.auxdata.items():
                print(f"  {name:>15} : {value}")

        print("\n\nVariable Auxiliary Data")
        print("-" * 70)
        for i in range(dat.num_vars):
            var_aux = dat.get_var_auxdata(i + 1)
            if len(var_aux) > 0:
                print(f"Var {i + 1:3} Aux Data  : {dict(var_aux)}")
                for name, value in var_aux.items():
                    print(f"  {name:>15} : {value}")

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

            if len(zone.auxdata) > 0:
                print(f"  Zone Aux Data   : {dict(zone.auxdata)}")
                for name, value in zone.auxdata.items():
                    print(f"  {name:>15} : {value}")

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

                value_str = np.array2string(
                    var.values, prefix="    Values        : ", separator=", "
                )
                print(f"    Values        : {value_str}")

            if zone.zone_type != ZoneType.ORDERED:
                print(f"  Node Map Shape  : {zone.node_map.shape}")
                value_str = np.array2string(
                    zone.node_map, prefix="  Connectivity    : ", separator=", "
                )
                print(f"  Connectivity    : {value_str}")


# ===========================================================================
# DATAPACKING=POINT parsing
# ===========================================================================


class TestPointFormat:
    """Hand-authored ``DATAPACKING=POINT`` fixtures, mimicking third-party writers."""

    def test_read_point_ordered(self, output_path: Callable) -> None:
        """Read a hand-authored DATAPACKING=POINT ordered zone.

        Demonstrates:
        - The POINT reader path with a minimal DAT string, one row per node
          containing all variable values on the same line
        - Values are checked exactly since the test data is small enough for
          exact ASCII representation
        """
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

        path = output_path("test_dat_read_point_ordered.dat")
        path.write_text(point_dat, encoding="utf-8")

        dat = tecio.open(str(path), "r")

        assert dat.num_vars == 3, f"Expected 3 vars, got {dat.num_vars}"
        assert dat.num_zones == 1, f"Expected 1 zone, got {dat.num_zones}"

        zone = dat.zone[0]
        assert zone.zone_type == ZoneType.ORDERED
        assert zone.datapacking == DataPacking.POINT
        assert zone.dimensions == (4, 1, 1)

        np.testing.assert_allclose(zone.variable[0].values.ravel(), expected_x)
        np.testing.assert_allclose(zone.variable[1].values.ravel(), expected_y)
        np.testing.assert_allclose(zone.variable[2].values.ravel(), expected_p)

    def test_read_point_fe(self, output_path: Callable) -> None:
        """Read a hand-authored DATAPACKING=POINT FE triangle zone.

        Demonstrates:
        - The POINT reader correctly separates the nodal data rows from the
          connectivity block that follows
        """
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

        path = output_path("test_dat_read_point_fe.dat")
        path.write_text(point_dat, encoding="utf-8")

        dat = tecio.open(str(path), "r")

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

    def test_read_point_mixed_cc(self, output_path: Callable) -> None:
        """Read a POINT zone with mixed nodal and cell-centred variables.

        Demonstrates:
        - The two-section layout: nodal rows (one per node, nodal vars only)
          followed by CC rows (one per element, CC vars only)
        """
        # Ordered 1-D zone: I=4 nodes, 3 cells.
        # Variables: x (nodal), p (nodal), rho (cell-centred).
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
        path = output_path("test_dat_read_point_mixed_cc.dat")
        path.write_text(point_dat, encoding="utf-8")

        dat = tecio.open(str(path), "r")

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


# ===========================================================================
# ZoneList / VariableList containers
# ===========================================================================


class TestContainers:
    """``ZoneList`` / ``VariableList`` container behaviour."""

    def test_zone_and_variable_containers(self, output_path: Callable) -> None:
        """Verify ``zone``/``variable`` return the new container types.

        Demonstrates:
        - A hand-authored two-zone BLOCK file (rather than the opaque Onera
          fixture) so zone slicing can be checked against known titles
        - ``Read.zone`` is a :class:`tecio.ZoneList`: an int index returns a
          ``ReadZone``; a slice returns another ``ZoneList`` of the same kind
        - ``ReadZone.variable`` is a :class:`tecio.VariableList`: index by
          0-based position or by exact, case-sensitive name
        - Unknown name -> ``KeyError``; out-of-range index -> ``IndexError``
        """
        two_zone_dat = textwrap.dedent("""\
            TITLE     = "two_zone_test"
            VARIABLES = "x" "y" "p"
            ZONE T="Zone1"
             STRANDID=0, SOLUTIONTIME=0.0
             I=4, J=1, K=1, ZONETYPE=Ordered
             DATAPACKING=BLOCK
            0.0 1.0 2.0 3.0
            0.0 1.0 2.0 3.0
            10.0 20.0 30.0 40.0
            ZONE T="Zone2"
             STRANDID=0, SOLUTIONTIME=1.0
             I=4, J=1, K=1, ZONETYPE=Ordered
             DATAPACKING=BLOCK
            4.0 5.0 6.0 7.0
            4.0 5.0 6.0 7.0
            50.0 60.0 70.0 80.0
            """)
        path = output_path("test_dat_containers.dat")
        path.write_text(two_zone_dat, encoding="utf-8")

        dat = tecio.open(str(path), "r")

        assert dat.num_vars == 3
        assert dat.num_zones == 2

        assert isinstance(dat.zone, tecio.ZoneList)
        assert len(dat.zone) == 2

        sub = dat.zone[0:2]
        assert isinstance(sub, tecio.ZoneList)
        assert len(sub) == 2
        assert sub[0].title == "Zone1"
        assert sub[1].title == "Zone2"

        zone = dat.zone[0]

        assert isinstance(zone.variable, tecio.VariableList)
        assert len(zone.variable) == 3
        assert zone.variable.names() == ["x", "y", "p"]
        assert zone.variable[0].name == zone.variable["x"].name == "x"

        # Name lookup is exact and case-sensitive -- "x" exists, "X" does not.
        with pytest.raises(KeyError):
            zone.variable["X"]

        with pytest.raises(KeyError):
            zone.variable["__not_a_real_variable__"]

        with pytest.raises(IndexError):
            zone.variable[10]


# ===========================================================================
# ReadZone.get_array
# ===========================================================================


class TestGetArray:
    """``ReadZone.get_array`` scalar, list, and passive-variable behaviour."""

    def test_get_array_matches_variable_values(self, output_path: Callable) -> None:
        """Verify ``get_array`` parity with ``variable[...].values``.

        Demonstrates:
        - Scalar index / scalar name both return the same array as the
          explicit ``zone.variable[key].values`` path
        - A list of names returns a tuple, in order -- never a bare array,
          even for a single-element list
        - Bad keys raise the same errors as ``VariableList``
        """
        point_dat = textwrap.dedent("""\
            TITLE     = "get_array_test"
            VARIABLES = "x" "y" "p"
            ZONE T="Zone1"
             STRANDID=0, SOLUTIONTIME=0.0
             I=4, J=1, K=1, ZONETYPE=Ordered
             DATAPACKING=BLOCK
            0.0 1.0 2.0 3.0
            0.0 1.0 2.0 3.0
            10.0 20.0 30.0 40.0
            """)
        path = output_path("test_dat_get_array.dat")
        path.write_text(point_dat, encoding="utf-8")

        dat = tecio.open(str(path), "r")
        zone = dat.zone[0]

        np.testing.assert_allclose(
            zone.get_array("x").ravel(), zone.variable["x"].values.ravel()
        )
        np.testing.assert_allclose(
            zone.get_array(0).ravel(), zone.variable[0].values.ravel()
        )

        pair = zone.get_array(["x", "p"])
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        np.testing.assert_allclose(pair[0].ravel(), [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(pair[1].ravel(), [10.0, 20.0, 30.0, 40.0])

        # A single-element list still returns a 1-tuple.
        one = zone.get_array(["x"])
        assert isinstance(one, tuple)
        assert len(one) == 1

        with pytest.raises(KeyError):
            zone.get_array("__not_a_real_variable__")

        with pytest.raises(IndexError):
            zone.get_array(10)

    def test_get_array_passive(self, output_path: Callable) -> None:
        """Verify ``get_array`` returns ``None`` for a passive ASCII variable.

        Demonstrates:
        - The DAT zone header declares variable 3 (``p``) passive via
          ``PASSIVEVARLIST``, so no data block is written for it
        - ``get_array`` returns ``None`` for it, by index or by name
        - A list key mixing an active and a passive variable returns
          ``(array, None)``
        """
        passive_dat = textwrap.dedent("""\
            TITLE     = "dat_passive_test"
            VARIABLES = "x" "y" "p"
            ZONE T="ZonePassive"
             STRANDID=0, SOLUTIONTIME=0.0
             I=4, J=1, K=1, ZONETYPE=Ordered
             DATAPACKING=BLOCK
             PASSIVEVARLIST=[3]
            0.0 1.0 2.0 3.0
            0.0 1.0 2.0 3.0
            """)
        path = output_path("test_dat_get_array_passive.dat")
        path.write_text(passive_dat, encoding="utf-8")

        dat = tecio.open(str(path), "r")
        zone = dat.zone[0]

        assert zone.variable["p"].is_passive()
        assert zone.get_array("p") is None
        assert zone.get_array(2) is None

        np.testing.assert_allclose(zone.get_array("x").ravel(), [0.0, 1.0, 2.0, 3.0])

        mixed = zone.get_array(["x", "p"])
        assert isinstance(mixed, tuple)
        np.testing.assert_allclose(mixed[0].ravel(), [0.0, 1.0, 2.0, 3.0])
        assert mixed[1] is None


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
