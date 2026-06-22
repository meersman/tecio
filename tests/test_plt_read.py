#!/usr/bin/env python3
"""pytest tests for :class:`tecio.plt.Read` -- PLT high-level reader.

Pattern per test:
    1. Open an existing or hand-authored ``.plt`` file via tecio.open(..., "r")
    2. Assert on metadata, container types (ZoneList / VariableList), and
       variable values

Run directly:

    $ python tests/test_plt_read.py -v

Keep output files for Tecplot 360 inspection:

    $ python tests/test_plt_read.py -v --keep-files
"""

# ruff: noqa: E501, SIM117

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import tecio
from tecio.libtecio import ZoneType

# ===========================================================================
# Full-file dump
# ===========================================================================


class TestReadDump:
    """Dump every header and data field for the shared Onera PLT fixture."""

    def test_read_dump(self, tests_dir: Path) -> None:
        """Print out all headers and data fields for the Onera PLT fixture.

        Demonstrates:
        - Top-level metadata: file_type, title, num_vars, variables, num_zones
        - Dataset- and variable-level auxiliary data access
        - Per-zone metadata: title, zone_type, solution_time, strand_id, and
          zone-level aux data
        - Per-variable metadata: name, data_type, value_location,
          is_enabled/is_passive, shared_zone, num_values, and the raw array
        - Connectivity (node_map) for FE zones
        """
        np.set_printoptions(threshold=100)

        input_file = tests_dir / "Onera.plt"
        pltfile = tecio.open(str(input_file), "r")

        print("\nFile Record")
        print("=" * 70)
        print(f"File Type         : {pltfile.file_type}")
        print(f"Dataset Title     : {pltfile.title}")
        print(f"Num Vars          : {pltfile.num_vars}")
        print(f"Variables         : {pltfile.variables}")
        print(f"Num Zones         : {pltfile.num_zones}")
        print(f"Dataset Aux Items : {pltfile.num_auxdata_items}")

        print("\n\nDataset Auxiliary Data")
        print("-" * 70)
        if len(pltfile.auxdata) > 0:
            print(f"Dataset Aux Data  : {dict(pltfile.auxdata)}")
            for name, value in pltfile.auxdata.items():
                print(f"  {name:>15} : {value}")

        print("\n\nVariable Auxiliary Data")
        print("-" * 70)
        for i in range(pltfile.num_vars):
            var_aux = pltfile.get_var_auxdata(i + 1)
            if len(var_aux) > 0:
                print(f"Var {i + 1:3} Aux Data  : {dict(var_aux)}")
                for name, value in var_aux.items():
                    print(f"  {name:>15} : {value}")

        print("\n\nZone Record")
        print("-" * 70)
        for i in range(pltfile.num_zones):
            zone = pltfile.zone[i]
            print(f"\nZone {i + 1:3}")
            print(f"  Title           : {zone.title}")
            print(f"  Zone Type       : {zone.zone_type}")
            print(f"  Is Enabled      : {zone.is_enabled()}")
            print(f"  Solution Time   : {zone.solution_time}")
            print(f"  Strand ID       : {zone.strand_id}")

            if len(zone.auxdata) > 0:
                print(f"  Zone Aux Data   : {dict(zone.auxdata)}")
                for name, value in zone.auxdata.items():
                    print(f"  {name:>15} : {value}")

            for j in range(pltfile.num_vars):
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
# ZoneList / VariableList containers
# ===========================================================================


class TestContainers:
    """``ZoneList`` / ``VariableList`` container behaviour."""

    def test_zone_and_variable_containers(self, tests_dir: Path) -> None:
        """Verify ``zone``/``variable`` return the new container types.

        Demonstrates:
        - ``Read.zone`` is a :class:`tecio.ZoneList`: an int index returns a
          ``ReadZone``; a slice returns another ``ZoneList`` of the same kind
        - ``ReadZone.variable`` is a :class:`tecio.VariableList`: index by
          0-based position or by exact, case-sensitive name
        - Unknown name -> ``KeyError``; out-of-range index -> ``IndexError``
        """
        input_file = tests_dir / "Onera.plt"
        pltfile = tecio.open(str(input_file), "r")

        assert isinstance(pltfile.zone, tecio.ZoneList)
        assert len(pltfile.zone) == pltfile.num_zones

        if pltfile.num_zones >= 2:
            sub = pltfile.zone[0:2]
            assert isinstance(sub, tecio.ZoneList)
            assert len(sub) == 2
            assert sub[0].title == pltfile.zone[0].title
            assert sub[1].title == pltfile.zone[1].title

        zone = pltfile.zone[0]

        assert isinstance(zone.variable, tecio.VariableList)
        assert len(zone.variable) == pltfile.num_vars
        assert zone.variable.names() == pltfile.variables

        first_name = pltfile.variables[0]
        assert zone.variable[0].name == zone.variable[first_name].name == first_name

        # Name lookup is exact and case-sensitive.
        swapped = first_name.swapcase()
        if swapped != first_name:
            with pytest.raises(KeyError):
                zone.variable[swapped]

        with pytest.raises(KeyError):
            zone.variable["__not_a_real_variable__"]

        with pytest.raises(IndexError):
            zone.variable[pltfile.num_vars + 10]


# ===========================================================================
# ReadZone.get_array
# ===========================================================================


class TestGetArray:
    """``ReadZone.get_array`` scalar, list, and passive-variable behaviour."""

    def test_get_array_matches_variable_values(self, tests_dir: Path) -> None:
        """Verify ``get_array`` parity with ``variable[...].values``.

        Demonstrates:
        - Scalar index / scalar name both return the same array as the
          explicit ``zone.variable[key].values`` path
        - A list of names returns a tuple, in order -- never a bare array,
          even for a single-element list
        - Bad keys raise the same errors as ``VariableList``
        """
        input_file = tests_dir / "Onera.plt"
        pltfile = tecio.open(str(input_file), "r")
        zone = pltfile.zone[0]
        names = pltfile.variables

        for i, name in enumerate(names[: min(3, len(names))]):
            expected = zone.variable[i].values
            if expected is None:
                continue
            np.testing.assert_array_equal(zone.get_array(i), expected)
            np.testing.assert_array_equal(zone.get_array(name), expected)

        if len(names) >= 2:
            pair = zone.get_array(names[:2])
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            for arr, name in zip(pair, names[:2], strict=True):
                expected = zone.variable[name].values
                if expected is None:
                    assert arr is None
                else:
                    np.testing.assert_array_equal(arr, expected)

        one = zone.get_array([names[0]])
        assert isinstance(one, tuple)
        assert len(one) == 1

        with pytest.raises(KeyError):
            zone.get_array("__not_a_real_variable__")

        with pytest.raises(IndexError):
            zone.get_array(pltfile.num_vars + 10)

    def test_get_array_passive_round_trip(self, output_path: Callable) -> None:
        """Round-trip a tiny PLT file with one passive variable.

        Demonstrates:
        - ``get_array`` returns ``None`` for a passive variable, by index or
          by name
        - A list key mixing an active and a passive variable returns
          ``(array, None)``
        """
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])

        path = output_path("test_plt_get_array_passive.plt")
        with tecio.open(str(path), "w") as w:
            w.write_ijk_zone(
                data=[x, y],
                variables=["x", "y", "p"],
                passive_vars=[False, False, True],
            )

        assert path.exists()
        with tecio.open(str(path), "r") as r:
            zone = r.zone[0]

            assert zone.variable["p"].is_passive()
            assert zone.get_array("p") is None
            assert zone.get_array(2) is None

            np.testing.assert_allclose(zone.get_array("x").ravel(), x)

            mixed = zone.get_array(["x", "p"])
            assert isinstance(mixed, tuple)
            np.testing.assert_allclose(mixed[0].ravel(), x)
            assert mixed[1] is None


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
