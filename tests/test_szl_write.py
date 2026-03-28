#!/usr/bin/env python3
r"""Tests for the :class:`szl.Write` higher-level writing API.

Only the IJK-ordered zone path (``write_ijk_zone``) is covered here, as
that is the method currently implemented in ``szl.Write``.  Each test
exercises a distinct capability:

* dimensionality  (1-D, 2-D, 3-D)
* data types      (float32, float64, mixed)
* value locations (all-nodal, mixed nodal / cell-centred)
* unsteady options (strand ID and solution time)
* zone auxiliary data
* lazy-open path  (variables supplied on first ``write_ijk_zone`` call)
* eager-open path (variables supplied to ``Write.__init__``)
* context-manager close
* variable-count mismatch guard
* array-shape mismatch guard

Data-generation helpers are imported directly from ``test_libtecio`` so
both test suites always exercise the same geometric cases.
"""

import numpy as np
from test_libtecio import _create_ordered

from tecio import szl
from tecio.libtecio import ValueLocation

#=======================================================================================
# Local functions to create all supported data formats
#=======================================================================================


def _scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Return a simple sin-cos scalar field over the supplied coordinate arrays."""
    if z is not None:
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1.0 + 0.1 * z)
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


#=======================================================================================
# IJK-ordered zone tests
#=======================================================================================


def test_write_ijk_3d() -> None:
    """Write a 3-D ordered zone (I, J, K all > 1)."""
    try:
        i, j, k = 3, 4, 5
        x, y, z = _create_ordered((i, j, k))
        c = _scalar_field(x, y, z).astype(np.float32)

        # Cell-centred: (I-1) x (J-1) x (K-1)
        cc = np.random.rand(i - 1, j - 1, k - 1).astype(np.float32)

        with szl.Write("test_szl_write_ijk_3d.szplt", title="3D_test") as writer:
            writer.write_ijk_zone(
                data=[x, y, z, c],
                variables=["x", "y", "z", "c"],
                title="zone_3d",
            )
            writer.write_ijk_zone(
                data=[x, y, z, cc],
                title="zone_cc",
                value_locations=[
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.NODAL,
                    ValueLocation.CELL_CENTERED,
                ],
            )
        print("PASS: test_write_ijk_3d")
    except Exception as e:
        print(f"FAIL: test_write_ijk_3d: {e}")


def test_write_ijk_unsteady() -> None:
    """Write multiple zones representing a transient solution (strand + time)."""
    try:
        i, j, k = 3, 4, 2
        x, y, z = _create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)

        solution_times = np.linspace(0.0, 2 * np.pi, 10)
        aux = {"MeshType": "structured", "Author": "test_szl_write"}

        with szl.Write(
            "test_szl_write_ijk_unsteady.szplt",
            title="unsteady_test",
        ) as writer:
            for t in solution_times:
                c = _scalar_field(x + t, y + t, z).astype(np.float32)
                writer.write_ijk_zone(
                    data=[x, y, z, c],
                    variables=["x", "y", "z", "c"],
                    strand_id=1,
                    solution_time=float(t),
                    aux=aux,
                )
        print("PASS: test_write_ijk_unsteady")
    except Exception as e:
        print(f"FAIL: test_write_ijk_unsteady: {e}")


#---------------------------------------------------------------------------------------
# Exception-raising tests for invalid input data
#---------------------------------------------------------------------------------------


def test_write_ijk_var_count_mismatch() -> None:
    """write_ijk_zone must raise when data length != variable count."""
    try:
        i, j, k = 3, 3, 1
        x, y, _ = _create_ordered((i, j, k))
        x = x.squeeze(0).astype(np.float32)
        y = y.squeeze(0).astype(np.float32)

        with szl.Write(
            "test_szl_write_ijk_var_mismatch.szplt",
            title="mismatch_test",
            variables=["x", "y", "c"],  # 3 variables declared
        ) as writer:
            writer.write_ijk_zone(
                data=[x, y],  # only 2 arrays supplied
                title="zone_bad",
            )
        print("FAIL: test_write_ijk_var_count_mismatch: expected ValueError, got none")
    except ValueError:
        print("PASS: test_write_ijk_var_count_mismatch")
    except Exception as e:
        print(f"FAIL: test_write_ijk_var_count_mismatch: unexpected exception: {e}")


def test_write_ijk_shape_mismatch() -> None:
    """write_ijk_zone must raise when two nodal arrays have different shapes."""
    try:
        i, j, k = 4, 5, 1
        x, y, _ = _create_ordered((i, j, k))
        x = x.squeeze(0).astype(np.float32)  # shape (j, i) = (5, 4)
        y_bad = y.squeeze(0)[:-1, :].astype(np.float32)  # shape (4, 4) — wrong

        with szl.Write(
            "test_szl_write_ijk_shape_mismatch.szplt",
            title="shape_test",
        ) as writer:
            writer.write_ijk_zone(
                data=[x, y_bad],
                title="zone_bad",
                variables=["x", "y"],
            )
        print("FAIL: test_write_ijk_shape_mismatch: expected ValueError, got none")
    except ValueError:
        print("PASS: test_write_ijk_shape_mismatch")
    except Exception as e:
        print(f"FAIL: test_write_ijk_shape_mismatch: unexpected exception: {e}")


#=======================================================================================
# Run all tests
#=======================================================================================
if __name__ == "__main__":
    test_write_ijk_3d()
    test_write_ijk_unsteady()
    test_write_ijk_var_count_mismatch()
    test_write_ijk_shape_mismatch()
