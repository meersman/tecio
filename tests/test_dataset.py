#!/usr/bin/env python3
"""pytest tests for :class:`tecio.Dataset` (and its ``Zone`` / ``Variable``).

``Dataset`` is the mutable, in-memory mirror of the ``Dataset -> Zone -> Variable``
hierarchy that the readers expose and the writers consume.  This suite exercises:

1. **Construction** -- from flat ``{"name": array}`` dicts, from the
   :meth:`Zone.ijk_from_dict` / :meth:`Zone.fe_from_dict` constructors (and the
   :meth:`Dataset.add_ijk_zone` / :meth:`add_fe_zone` wrappers), and incrementally
   with ``add_zone`` / ``add_variable`` (which keeps every zone rectangular).
2. **Sharing** -- a shared variable holds a *reference* to its source variable and a
   zone that shares connectivity references its source zone.  ``values`` / ``node_map``
   read through to the source, ``shared_zone`` / ``shared_connectivity`` are 1-based
   indices derived from where the source sits (so they survive reordering), and
   :meth:`Dataset.branch_variables` / :meth:`branch_connectivity` turn shares into
   independent copies.
3. **Round-trips** -- writing a dataset out through ``tecio.open`` and reading it back
   with ``Dataset(path)``, verified consistent across every writer format (szl, plt,
   dat).  Round-trip value checks write at ``precision="double"`` so per-format
   precision defaults (covered by ``test_write.py``) don't enter into it.
4. **Real-file loading** -- parsing the shared ``Onera.*`` reference fixture: metadata,
   FE topology, zone-level auxiliary data, and ``zones=`` / ``variables=`` subsetting.

Test data comes from :mod:`create_test_data` (the same standardized generators the rest
of the suite uses) and the ``output_path`` / ``tests_dir`` fixtures from ``conftest.py``.

Run directly:

    $ python tests/test_dataset.py -v

Keep output files for Tecplot 360 inspection:

    $ python tests/test_dataset.py -v --keep-files

"""

# ruff: noqa: E501, SIM117

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from create_test_data import (
    create_FE_brick,
    create_FE_lineseg,
    create_FE_quad,
    create_FE_tet,
    create_FE_tri,
    create_ordered,
    scalar_field,
)

import tecio
from tecio import AuxData, Dataset, Variable, Zone
from tecio.libtecio import DataType, FileType, ValueLocation, ZoneType

# ======================================================================================
# Cross-format helpers
# ======================================================================================

#: Every writer format under test, and the file extension tecio.open() dispatches on.
FORMATS: list[str] = ["szl", "plt", "dat"]
_EXTENSIONS: dict[str, str] = {"szl": "szplt", "plt": "plt", "dat": "dat"}

#: Round-trip tolerance per format when writing at precision="double".
_RTOL: dict[str, float] = {"szl": 1e-10, "plt": 1e-10, "dat": 1e-7}


@pytest.fixture(params=FORMATS)
def fmt(request) -> str:
    """Parametrize a test across every writer format (szl, plt, dat)."""
    return request.param


@pytest.fixture
def onera_file(fmt: str, tests_dir: Path) -> Path:
    """Resolve the per-format Onera reference fixture, skipping if it is missing.

    Mirrors ``test_read.py``: a format whose ``Onera.<ext>`` file is absent is
    skipped rather than failed, so the suite degrades gracefully when only some
    of the three formats are checked in.
    """
    path = tests_dir / f"Onera.{_EXTENSIONS[fmt]}"
    if not path.exists():
        pytest.skip(f"No Onera fixture for format {fmt!r} at {path}")
    return path


def _path(output_path: Callable, fmt: str, name: str) -> Path:
    """Build an output path with the extension for *fmt*."""
    return output_path(f"{name}.{_EXTENSIONS[fmt]}")


def _write_read(ds: Dataset, path: Path, *, precision: str | None = "double", **kw) -> Dataset:
    """Write *ds* to *path* and read it back into a fresh :class:`Dataset`."""
    ds.write(path, precision=precision, **kw)
    assert path.exists()
    return Dataset(path)


# ======================================================================================
# Construction
# ======================================================================================


class TestConstruction:
    """Building datasets in memory and keeping them rectangular."""

    def test_empty_dataset(self) -> None:
        """A dataset with no source is empty but fully usable."""
        ds = Dataset(title="empty")
        assert ds.num_zones == 0
        assert ds.num_vars == 0
        assert ds.variables == []
        assert list(ds) == []

    def test_flat_dict_source(self) -> None:
        """``Dataset({name: array})`` becomes one ordered zone."""
        x, y, _ = create_ordered((5, 1, 1))
        ds = Dataset({"x": x.ravel(), "p": scalar_field(x, y).ravel()}, title="flat")
        assert ds.num_zones == 1
        assert ds.variables == ["x", "p"]
        assert ds.zone[0].zone_type == ZoneType.ORDERED
        np.testing.assert_array_equal(ds.zone[0].get_array("x"), x.ravel())

    def test_from_dict_classmethod(self) -> None:
        """:meth:`Dataset.from_dict` mirrors the flat-dict constructor."""
        ds = Dataset.from_dict({"x": np.arange(3.0)}, title="d")
        assert ds.title == "d"
        assert ds.num_zones == 1
        assert ds.variables == ["x"]

    def test_add_zone_reconciles_variables(self) -> None:
        """Adding zones with differing variables keeps every zone rectangular."""
        ds = Dataset()
        ds.add_ijk_zone({"x": np.arange(4.0), "y": np.arange(4.0)}, title="a")
        ds.add_ijk_zone({"x": np.arange(4.0), "z": np.arange(4.0)}, title="b")

        assert ds.variables == ["x", "y", "z"]
        for zone in ds.zone:
            assert [v.name for v in zone.variable] == ["x", "y", "z"]
        assert ds.zone[0].get_variable("z").is_passive()
        assert ds.zone[1].get_variable("y").is_passive()
        assert not ds.zone[0].get_variable("y").is_passive()

    def test_add_variable_propagates_dataset_wide(self) -> None:
        """A new variable appears (passive) in every existing zone."""
        ds = Dataset()
        ds.add_ijk_zone({"x": np.arange(3.0)}, title="a")
        ds.add_ijk_zone({"x": np.arange(3.0)}, title="b")
        idx = ds.add_variable("p")
        assert idx == 1
        assert ds.variables == ["x", "p"]
        assert all(z.get_variable("p").is_passive() for z in ds.zone)

    def test_add_variable_with_default_fills_existing_zones(self) -> None:
        """``default=`` fills existing zones with a constant instead of passive."""
        ds = Dataset()
        ds.add_ijk_zone({"x": np.arange(3.0)}, title="a")
        ds.add_variable("p", default=2.5)
        np.testing.assert_array_equal(ds.zone[0].get_array("p"), np.full(3, 2.5))

    def test_rename_variable_is_dataset_wide(self) -> None:
        """Renaming updates the dataset list and every zone consistently."""
        ds = Dataset()
        ds.add_ijk_zone({"x": np.arange(3.0), "p": np.arange(3.0)}, title="a")
        ds.add_ijk_zone({"x": np.arange(3.0), "p": np.arange(3.0)}, title="b")
        ds.rename_variable("p", "pressure")
        assert ds.variables == ["x", "pressure"]
        for zone in ds.zone:
            assert zone.get_variable("pressure").name == "pressure"

    def test_delete_variable_removes_everywhere(self) -> None:
        """Deleting removes the variable from the dataset and every zone."""
        ds = Dataset()
        ds.add_ijk_zone({"x": np.arange(3.0), "p": np.arange(3.0)}, title="a")
        ds.delete_variable("p")
        assert ds.variables == ["x"]
        assert len(ds.zone[0]) == 1

    def test_unsupported_source_raises_type_error(self) -> None:
        """An unrecognized source type is rejected with a clear message."""
        with pytest.raises(TypeError, match="Unsupported Dataset source"):
            Dataset(object())


# ======================================================================================
# Dict zone constructors
# ======================================================================================


class TestDictConstructors:
    """``Zone.ijk_from_dict`` / ``Zone.fe_from_dict`` and the Dataset wrappers."""

    def test_ijk_from_dict_infers_dimensions(self) -> None:
        """An ordered zone infers ``(i, j, k)`` from the arrays."""
        x, y, z = create_ordered((4, 3, 2))
        zone = Zone.ijk_from_dict({"x": x, "y": y, "z": z})
        assert zone.zone_type == ZoneType.ORDERED
        assert zone.dimensions == (4, 3, 2)
        assert zone.num_nodes == 24

    def test_ijk_from_dict_value_location_strings(self) -> None:
        """String value-locations are coerced (case-insensitively)."""
        x, _, _ = create_ordered((6, 1, 1))
        zone = Zone.ijk_from_dict(
            {"x": x.ravel(), "p": np.arange(5.0)},
            value_locations={"p": "cell_centered"},
        )
        assert zone.get_variable("p").value_location == ValueLocation.CELL_CENTERED
        assert zone.get_variable("x").value_location == ValueLocation.NODAL

    def test_fe_from_dict_lineseg(self) -> None:
        x, y, node_map = create_FE_lineseg()
        zone = Zone.fe_from_dict({"x": x, "y": y}, node_map)
        assert zone.zone_type == ZoneType.FELINESEG

    def test_fe_from_dict_triangle(self) -> None:
        x, y, node_map = create_FE_tri()
        zone = Zone.fe_from_dict({"x": x, "y": y}, node_map)
        assert zone.zone_type == ZoneType.FETRIANGLE

    def test_fe_from_dict_tetrahedron(self) -> None:
        """4 nodes-per-cell defaults to tetrahedron."""
        x, y, z, node_map = create_FE_tet()
        zone = Zone.fe_from_dict({"x": x, "y": y, "z": z}, node_map)
        assert zone.zone_type == ZoneType.FETETRAHEDRON

    def test_fe_from_dict_brick(self) -> None:
        x, y, z, _faces, node_map = create_FE_brick()
        zone = Zone.fe_from_dict({"x": x, "y": y, "z": z}, node_map)
        assert zone.zone_type == ZoneType.FEBRICK

    def test_fe_from_dict_quad_needs_explicit_type(self) -> None:
        """A 4-node quad must be requested explicitly (4 -> tet by default)."""
        x, y, node_map = create_FE_quad()
        zone = Zone.fe_from_dict(
            {"x": x, "y": y}, node_map, zone_type=ZoneType.FEQUADRILATERAL
        )
        assert zone.zone_type == ZoneType.FEQUADRILATERAL

    def test_fe_from_dict_unknown_nodes_per_cell_raises(self) -> None:
        """An un-inferable nodes-per-cell count is rejected."""
        with pytest.raises(ValueError, match="nodes-per-cell"):
            Zone.fe_from_dict({"x": np.arange(6.0)}, np.arange(1, 7).reshape(1, 6))

    def test_add_ijk_zone_returns_added_zone(self) -> None:
        """``add_ijk_zone`` builds, adds, and returns the zone."""
        ds = Dataset()
        z = ds.add_ijk_zone({"x": np.arange(3.0)}, title="a")
        assert z is ds.zone[0]
        assert z.dataset is ds

    def test_add_fe_zone_counts(self) -> None:
        """``add_fe_zone`` sets node/element counts from coords and node map."""
        ds = Dataset()
        x, y, node_map = create_FE_tri()
        z = ds.add_fe_zone({"x": x, "y": y}, node_map, title="mesh")
        assert z.zone_type == ZoneType.FETRIANGLE
        assert z.num_nodes == len(x)
        assert z.num_elements == node_map.shape[0]
        np.testing.assert_array_equal(z.node_map, node_map)


# ======================================================================================
# Read parity (Variable / Zone) -- in-memory
# ======================================================================================


class TestReadParity:
    """Reader-shaped accessors on the mutable container."""

    def test_variable_states(self) -> None:
        """active / passive report the expected flags and sizes."""
        active = Variable("x", np.arange(4.0))
        passive = Variable("p", is_passive=True)
        assert not active.is_passive() and active.is_enabled()
        assert active.num_values == 4 and active.shape == (4,)
        assert passive.is_passive() and not passive.is_enabled()
        assert passive.values is None and passive.num_values == 0

    def test_get_array_and_attribute_access(self) -> None:
        """``get_array`` and ``zone.<name>`` return the underlying data."""
        zone = Zone.ijk_from_dict({"x": np.arange(3.0), "p": np.arange(3.0) + 10})
        x, p = zone.get_array(["x", "p"])
        np.testing.assert_array_equal(x, np.arange(3.0))
        np.testing.assert_array_equal(zone.p, np.arange(3.0) + 10)  # __getattr__

    def test_get_array_list_returns_tuple(self) -> None:
        """A list key returns a tuple, even for a single name."""
        zone = Zone.ijk_from_dict({"x": np.arange(3.0)})
        one = zone.get_array(["x"])
        assert isinstance(one, tuple) and len(one) == 1

    def test_get_values_1based_slice(self) -> None:
        """``get_values`` supports a 1-based half-open slice of the flat array."""
        v = Variable("x", np.arange(10.0))
        np.testing.assert_array_equal(v.get_values((2, 5)), np.array([1.0, 2.0, 3.0]))

    def test_data_type_inference(self) -> None:
        """``data_type`` follows the array dtype unless overridden."""
        assert Variable("a", np.arange(3, dtype=np.float32)).data_type == DataType.FLOAT
        assert Variable("b", np.arange(3, dtype=np.float64)).data_type == DataType.DOUBLE
        assert Variable("c", np.arange(3, dtype=np.int32)).data_type == DataType.INT32

    def test_fe_counts_inferred_from_node_map(self) -> None:
        """Node/element counts fall back to the node map when not given."""
        x, y, z, node_map = create_FE_tet()
        zone = Zone.fe_from_dict({"x": x, "y": y, "z": z}, node_map)
        assert zone.num_nodes == int(node_map.max())
        assert zone.num_elements == node_map.shape[0]
        assert zone.nodes_per_cell == 4


# ======================================================================================
# Variable sharing (reference-based)
# ======================================================================================


class TestVariableSharing:
    """A shared variable references its source and reads through to it."""

    def _grid_and_solution(self):
        """Two ordered zones over the same grid; the second reuses coordinates."""
        x, y, z = create_ordered((5, 4, 1))
        ds = Dataset()
        src = ds.add_ijk_zone(
            {"x": x, "y": y, "z": z, "c": scalar_field(x, y, z)}, title="src"
        )
        dst = ds.add_ijk_zone({"c": scalar_field(x + 1, y, z)}, title="dst")
        return ds, src, dst, x

    def test_share_from_reads_through(self) -> None:
        """``share_from`` makes ``values`` resolve to the source array."""
        ds, src, dst, x = self._grid_and_solution()
        dst.get_variable("x").share_from(src.get_variable("x"))
        v = dst.get_variable("x")
        assert v.is_shared()
        assert v.shared_zone == 1  # 1-based index of the source zone
        assert v.source is src.get_variable("x")
        np.testing.assert_array_equal(v.values, x)
        np.testing.assert_array_equal(dst.get_array("x"), x)

    def test_shared_zone_setter_by_index(self) -> None:
        """Setting ``shared_zone`` to a 1-based index resolves via the dataset."""
        ds, src, dst, x = self._grid_and_solution()
        dst.get_variable("x").shared_zone = 1
        assert dst.get_variable("x").is_shared()
        np.testing.assert_array_equal(dst.get_array("x"), x)

    def test_shared_zone_setter_none_clears(self) -> None:
        """Setting ``shared_zone = None`` clears the share."""
        ds, src, dst, _ = self._grid_and_solution()
        v = dst.get_variable("x")
        v.share_from(src.get_variable("x"))
        v.shared_zone = None
        assert not v.is_shared() and v.shared_zone is None

    def test_shared_zone_by_index_on_detached_raises(self) -> None:
        """Resolving a share by index needs a dataset."""
        v = Variable("x", is_passive=True)
        with pytest.raises(ValueError, match="detached"):
            v.shared_zone = 1

    def test_share_survives_reordering(self) -> None:
        """The 1-based index recomputes when zones are reordered."""
        ds, src, dst, x = self._grid_and_solution()
        dst.get_variable("x").share_from(src.get_variable("x"))
        ds.zone.reverse()  # source moves from index 0 to index 1
        assert dst.get_variable("x").shared_zone == 2
        np.testing.assert_array_equal(dst.get_array("x"), x)

    def test_branch_variables_makes_independent_copy(self) -> None:
        """``branch_variables`` copies source data and clears the share."""
        ds, src, dst, x = self._grid_and_solution()
        dst.get_variable("x").share_from(src.get_variable("x"))
        ds.branch_variables()
        v = dst.get_variable("x")
        assert not v.is_shared() and v.shared_zone is None
        np.testing.assert_array_equal(v.values, x)
        assert v.values is not src.get_variable("x").values  # a real copy

    def test_copy_materializes_share(self) -> None:
        """``Variable.copy`` of a shared variable yields an independent copy."""
        ds, src, dst, x = self._grid_and_solution()
        dst.get_variable("x").share_from(src.get_variable("x"))
        c = dst.get_variable("x").copy()
        assert not c.is_shared()
        np.testing.assert_array_equal(c.values, x)

    def test_setting_values_clears_share(self) -> None:
        """Assigning an array to a shared variable makes it active again."""
        ds, src, dst, x = self._grid_and_solution()
        v = dst.get_variable("x")
        v.share_from(src.get_variable("x"))
        v.values = x + 7
        assert not v.is_shared()
        np.testing.assert_array_equal(v.values, x + 7)


# ======================================================================================
# Connectivity sharing (reference-based)
# ======================================================================================


class TestConnectivitySharing:
    """An FE zone can reference another zone's connectivity."""

    def _grid_and_solution(self):
        """A grid zone plus a solution zone that reuses its connectivity."""
        x, y, node_map = create_FE_tri()
        ds = Dataset()
        src = ds.add_fe_zone({"x": x, "y": y}, node_map, title="grid")
        dst = ds.add_zone(
            Zone("sol", ZoneType.FETRIANGLE, num_nodes=len(x), num_elements=node_map.shape[0])
        )
        dst.add_variable("x", x)
        dst.add_variable("y", y)
        return ds, src, dst, node_map

    def test_share_connectivity_reads_through(self) -> None:
        """``share_connectivity_from`` makes ``node_map`` resolve to the source."""
        ds, src, dst, node_map = self._grid_and_solution()
        dst.share_connectivity_from(src)
        assert dst.shares_connectivity()
        assert dst.shared_connectivity == 1
        assert dst.connectivity_source is src
        np.testing.assert_array_equal(dst.node_map, node_map)

    def test_shared_connectivity_setter_index(self) -> None:
        """Setting ``shared_connectivity`` to an index resolves via the dataset."""
        ds, src, dst, node_map = self._grid_and_solution()
        dst.shared_connectivity = 1
        assert dst.shares_connectivity()
        np.testing.assert_array_equal(dst.node_map, node_map)

    def test_connectivity_share_survives_reordering(self) -> None:
        """The connectivity index recomputes after reordering."""
        ds, src, dst, _ = self._grid_and_solution()
        dst.share_connectivity_from(src)
        ds.zone.reverse()
        assert dst.shared_connectivity == 2

    def test_branch_connectivity_makes_independent_node_map(self) -> None:
        """``branch_connectivity`` copies the node map and clears the share."""
        ds, src, dst, node_map = self._grid_and_solution()
        dst.share_connectivity_from(src)
        ds.branch_connectivity()
        assert not dst.shares_connectivity()
        assert dst.shared_connectivity is None
        np.testing.assert_array_equal(dst.node_map, node_map)
        assert dst.node_map is not src.node_map

    def test_assigning_node_map_clears_share(self) -> None:
        """Assigning a node map makes the zone own its connectivity."""
        ds, src, dst, node_map = self._grid_and_solution()
        dst.share_connectivity_from(src)
        dst.node_map = node_map + 0  # a fresh array
        assert not dst.shares_connectivity()


# ======================================================================================
# branch() convenience + removed paths
# ======================================================================================


class TestBranchAndRemovals:
    """The combined ``branch`` helper and the removed construction paths."""

    def test_branch_breaks_all_shares(self) -> None:
        """``branch`` clears both variable and connectivity shares."""
        x, y, node_map = create_FE_tri()
        ds = Dataset()
        src = ds.add_fe_zone({"x": x, "y": y}, node_map, title="grid")
        dst = ds.add_zone(
            Zone("sol", ZoneType.FETRIANGLE, num_nodes=len(x), num_elements=node_map.shape[0])
        )
        dst.add_variable("x", x)
        dst.add_variable("y", y)
        dst.get_variable("x").share_from(src.get_variable("x"))
        dst.share_connectivity_from(src)

        ds.branch()
        assert not dst.get_variable("x").is_shared()
        assert not dst.shares_connectivity()

    def test_dataframe_source_rejected(self) -> None:
        """A DataFrame-like object is no longer a valid construction source."""

        class FakeDF:
            columns = ["x"]

            def to_numpy(self):
                return np.arange(3.0)

        with pytest.raises(TypeError):
            Dataset(FakeDF())

    def test_from_dataframe_removed(self) -> None:
        """The ``from_dataframe`` constructor has been removed."""
        assert not hasattr(Dataset, "from_dataframe")


# ======================================================================================
# AuxData
# ======================================================================================


class TestAuxData:
    """Typed accessors on the auxiliary-data mapping."""

    def test_typed_accessors(self) -> None:
        aux = AuxData({"Iteration": "42", "Residual": "1.5e-3", "Converged": "yes"})
        assert aux.as_int("Iteration") == 42
        assert aux.as_float("Residual") == pytest.approx(1.5e-3)
        assert aux.as_bool("Converged") is True
        assert aux.as_int("missing", default=-1) == -1


# ======================================================================================
# Round-trips (write via tecio.open under the hood, read back with Dataset)
# ======================================================================================


class TestRoundTrip:
    """End-to-end write/read across every writer format."""

    def test_ordered_zone(self, fmt: str, output_path: Callable) -> None:
        """A basic ordered dataset round-trips structure and values."""
        i, j, k = 4, 3, 2
        x, y, z = create_ordered((i, j, k))
        c = scalar_field(x, y, z)
        ds = Dataset({"x": x, "y": y, "z": z, "c": c}, title="rt")

        rt = _write_read(ds, _path(output_path, fmt, "rt_ordered"))

        assert rt.num_zones == 1
        assert rt.variables == ["x", "y", "z", "c"]
        assert rt.zone[0].dimensions == (i, j, k)
        np.testing.assert_allclose(
            rt.zone[0].get_array("c").ravel(order="F"),
            c.ravel(order="F"),
            rtol=_RTOL[fmt],
        )

    def test_multiple_zones(self, fmt: str, output_path: Callable) -> None:
        """Multiple zones and their node counts survive the round-trip."""
        xa, ya, za = create_ordered((5, 1, 1))
        xb, yb, zb = create_ordered((7, 1, 1))
        ds = Dataset(title="multi")
        ds.add_ijk_zone({"x": xa.ravel(), "c": scalar_field(xa, ya).ravel()}, title="z0")
        ds.add_ijk_zone({"x": xb.ravel(), "c": scalar_field(xb, yb).ravel()}, title="z1")

        rt = _write_read(ds, _path(output_path, fmt, "rt_multi"))

        assert rt.num_zones == 2
        assert rt.zone[0].num_nodes == 5
        assert rt.zone[1].num_nodes == 7

    def test_variable_sharing_preserved(self, fmt: str, output_path: Callable) -> None:
        """A shared variable is written as a share and re-read as a reference."""
        x, y, z = create_ordered((5, 4, 1))
        ds = Dataset(title="share")
        src = ds.add_ijk_zone(
            {"x": x, "y": y, "z": z, "c": scalar_field(x, y, z)}, title="src"
        )
        dst = ds.add_ijk_zone({"c": scalar_field(x + 1, y, z)}, title="dst")
        for name in ("x", "y", "z"):
            dst.get_variable(name).share_from(src.get_variable(name))

        path = _path(output_path, fmt, "rt_share")
        rt = _write_read(ds, path)

        assert rt.num_zones == 2
        shared = rt.zone[1].get_variable("x")
        assert shared.is_shared(), "share should round-trip as a reference"
        assert shared.shared_zone == 1
        np.testing.assert_allclose(
            rt.zone[1].get_array("x").ravel(order="F"), x.ravel(order="F"), rtol=_RTOL[fmt]
        )
        # Reader-parity cross-check: the raw reader also reports the share.
        with tecio.open(str(path), "r") as r:
            assert r.zone[1].variable[0].shared_zone is not None
        # And it can be branched into an independent copy after loading.
        rt.branch_variables()
        assert not rt.zone[1].get_variable("x").is_shared()

    def test_passive_variable(self, fmt: str, output_path: Callable) -> None:
        """A passive variable round-trips as passive."""
        ds = Dataset()
        z = ds.add_ijk_zone({"x": np.arange(5.0)}, title="z")
        ds.add_variable("unused")  # passive in the only zone
        assert z.get_variable("unused").is_passive()

        rt = _write_read(ds, _path(output_path, fmt, "rt_passive"))
        assert rt.zone[0].get_variable("unused").is_passive()

    def test_zone_auxiliary_data(self, fmt: str, output_path: Callable) -> None:
        """Zone-level aux data round-trips for every format."""
        ds = Dataset(title="aux")
        ds.add_ijk_zone(
            {"x": np.arange(4.0)},
            title="z",
            aux={"Author": "pytest", "MeshType": "structured"},
        )

        rt = _write_read(ds, _path(output_path, fmt, "rt_zone_aux"))

        assert rt.zone[0].auxdata["Author"] == "pytest"
        assert rt.zone[0].auxdata["MeshType"] == "structured"

    def test_dataset_auxiliary_data_szl(self, output_path: Callable) -> None:
        """Dataset-level aux data round-trips (verified on SZL only).

        Dataset-level auxiliary data written through :meth:`Dataset.write`
        currently round-trips for SZL but not for PLT/DAT -- for those formats
        the dataset aux added just after opening the writer does not reach the
        file header.  Zone-level aux (covered above) is unaffected and works for
        every format.
        """
        ds = Dataset(title="aux")
        ds.auxdata.update({"Solver": "TestCode", "Mach": "0.72"})
        ds.add_ijk_zone({"x": np.arange(4.0)}, title="z")

        rt = _write_read(ds, output_path("rt_dataset_aux.szplt"))

        assert rt.auxdata["Solver"] == "TestCode"
        assert rt.auxdata["Mach"] == "0.72"

    def test_file_type_grid(self, fmt: str, output_path: Callable) -> None:
        """``FileType.GRID`` survives the round-trip."""
        ds = Dataset({"x": np.arange(5.0)}, file_type=FileType.GRID)
        rt = _write_read(ds, _path(output_path, fmt, "rt_grid"))
        assert rt.file_type == FileType.GRID

    def test_fe_zone(self, fmt: str, output_path: Callable) -> None:
        """An FE zone with owned connectivity round-trips."""
        x, y, z, node_map = create_FE_tet()
        ds = Dataset(title="fe")
        ds.add_fe_zone({"x": x, "y": y, "z": z}, node_map, title="tets")

        rt = _write_read(ds, _path(output_path, fmt, "rt_fe"))

        zone = rt.zone[0]
        assert zone.zone_type == ZoneType.FETETRAHEDRON
        assert zone.num_nodes == int(node_map.max())
        assert zone.num_elements == node_map.shape[0]
        np.testing.assert_array_equal(np.asarray(zone.node_map), node_map)


class TestRoundTripSZL:
    """SZL-only round-trips for the trickier features."""

    def test_fe_connectivity_sharing_preserved(self, output_path: Callable) -> None:
        """FE connectivity sharing round-trips as a reference (szl).

        ``_copy_zones`` forwards ``con_sharing``, so the sharing zone is
        re-emitted as a connectivity share and read back as a reference; the
        resolved node map is identical on both zones.
        """
        x, y, node_map = create_FE_tri()
        ds = Dataset(title="con_share")
        grid = ds.add_fe_zone({"x": x, "y": y}, node_map, title="grid")
        dst = ds.add_zone(
            Zone("sol", ZoneType.FETRIANGLE, num_nodes=len(x), num_elements=node_map.shape[0])
        )
        dst.add_variable("x", x)
        dst.add_variable("y", y)
        dst.share_connectivity_from(grid)

        rt = _write_read(ds, output_path("rt_con_share.szplt"))

        assert rt.num_zones == 2
        assert rt.zone[1].shares_connectivity()
        assert rt.zone[1].shared_connectivity == 1
        np.testing.assert_array_equal(np.asarray(rt.zone[0].node_map), node_map)
        np.testing.assert_array_equal(np.asarray(rt.zone[1].node_map), node_map)


# ======================================================================================
# Real-file loading (shared Onera reference fixture)
# ======================================================================================


class TestReadOneraFile:
    """Load the Onera wing solution and check the container mirrors the file."""

    def test_metadata(self, onera_file: Path) -> None:
        """Title, variable list, and counts match the file header."""
        ds = Dataset(onera_file)
        assert ds.num_zones == 2
        assert ds.num_vars == 18
        assert ds.variables[:3] == ["x", "y", "z"]
        assert "Pressure" in ds.variables
        assert "Visualization" in ds.title

    def test_zone_topology(self, onera_file: Path) -> None:
        """Both FE zones report the expected type and node/element counts."""
        ds = Dataset(onera_file)
        volume, surface = ds.zone
        assert volume.title == "FluidVolume"
        assert volume.zone_type == ZoneType.FEBRICK
        assert volume.num_nodes == 46417
        assert volume.num_elements == 43008
        assert volume.node_map.shape[0] == 43008
        assert surface.title == "WingSurface"
        assert surface.zone_type == ZoneType.FEQUADRILATERAL
        assert surface.num_nodes == 1453
        assert surface.num_elements == 1408

    def test_zone_level_aux(self, onera_file: Path) -> None:
        """The surface zone carries its boundary-condition aux data."""
        ds = Dataset(onera_file)
        surface = ds.zone[1]
        assert surface.auxdata["Common.BoundaryCondition"] == "Wall"
        assert surface.auxdata.as_bool("Common.IsBoundaryZone") is True

    def test_variable_data_and_ownership(self, onera_file: Path) -> None:
        """Coordinates come back nodal, sized to the zone, and owned (not shared)."""
        ds = Dataset(onera_file)
        volume = ds.zone[0]
        x = volume.get_array("x")
        assert x is not None
        assert x.size == 46417
        assert not volume.get_variable("x").is_shared()
        assert not volume.shares_connectivity()

    def test_get_array_matches_variable_values(self, onera_file: Path) -> None:
        """``get_array`` agrees with ``variable[...].values`` (reader parity)."""
        ds = Dataset(onera_file)
        zone = ds.zone[0]
        for name in ds.variables[:3]:
            np.testing.assert_array_equal(
                zone.get_array(name), zone.get_variable(name).values
            )

    def test_subset_zones(self, onera_file: Path) -> None:
        """``zones=`` keeps only the requested zones."""
        ds = Dataset(onera_file, zones=[1])
        assert ds.num_zones == 1
        assert ds.zone[0].title == "WingSurface"

    def test_subset_variables(self, onera_file: Path) -> None:
        """``variables=`` keeps only the requested variables, in order."""
        ds = Dataset(onera_file, variables=["x", "y", "z", "Pressure"])
        assert ds.variables == ["x", "y", "z", "Pressure"]
        assert ds.num_vars == 4
        assert ds.zone[0].get_array("Pressure") is not None


# ======================================================================================
# DataFrame export (optional pandas)
# ======================================================================================


class TestDataFrameExport:
    """``to_dataframe`` remains as an export-only convenience."""

    def test_to_dataframe_resolves_shared(self) -> None:
        """A shared column is read through to its source array."""
        pytest.importorskip("pandas")
        x = np.arange(3.0)
        ds = Dataset()
        src = ds.add_ijk_zone({"x": x, "p": np.arange(3.0)}, title="a")
        dst = ds.add_ijk_zone({"p": np.arange(3.0) + 9}, title="b")
        dst.get_variable("x").share_from(src.get_variable("x"))

        df = ds.to_dataframe(1)
        assert list(df.columns) == ["x", "p"]
        np.testing.assert_array_equal(df["x"].to_numpy(), x)


# ======================================================================================
# Entry point
# ======================================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
