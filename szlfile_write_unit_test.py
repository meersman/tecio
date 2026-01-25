#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

import szlfile_write as szlw
from szlio import DataType, FileType, ValueLocation

# quick smoke tests / examples adapted from original sample


def test():
    f = szlw.open_file(
        "test.szplt", "Title", ["byte", "short", "long", "float", "double"]
    )
    z = szlw.create_ordered_zone(
        f,
        "Zone",
        (3, 3, 1),
        var_sharing=None,
        var_data_types=[
            DataType.BYTE,
            DataType.INT16,
            DataType.INT32,
            DataType.FLOAT,
            DataType.DOUBLE,
        ],
    )
    szlw.zone_write_uint8_values(
        f, z, 1, np.asarray([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.uint8)
    )
    szlw.zone_write_int16_values(
        f, z, 2, np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int16)
    )
    szlw.zone_write_int32_values(
        f, z, 3, np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    )
    szlw.zone_write_float_values(f, z, 4, np.linspace(0, 1, 9, dtype=np.float32))
    szlw.zone_write_double_values(f, z, 5, np.linspace(1, 2, 9, dtype=np.float64))
    szlw.close_file(f)
    print("Wrote test.szplt")


def test_gridandsolution(grid_file="grid.szplt", solution_template="solution.szplt"):
    grid_file_handle = szlw.open_file(
        grid_file, "Title", ["x", "y"], file_type=FileType.GRID
    )
    value_locations = [ValueLocation.NODAL, ValueLocation.NODAL]
    z = szlw.create_ordered_zone(
        grid_file_handle,
        "Zone",
        (3, 3, 1),
        value_locations=value_locations,
        var_data_types=[DataType.DOUBLE, DataType.DOUBLE],
    )
    szlw.zone_set_solution_time(grid_file_handle, z, strand=1)
    szlw.zone_write_double_values(
        grid_file_handle,
        z,
        1,
        np.asarray([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.float64),
    )
    szlw.zone_write_double_values(
        grid_file_handle,
        z,
        2,
        np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.float64),
    )

    for t in [1, 2, 3]:
        outfile = f"{t}_{solution_template}"
        solution_file_handle = szlw.open_file(
            outfile,
            "Title",
            ["c"],
            file_type=FileType.SOLUTION,
            grid_file_handle=grid_file_handle,
        )
        value_locations = [ValueLocation.CELL_CENTERED]
        sol_z = szlw.create_ordered_zone(
            solution_file_handle,
            "Zone",
            (3, 3, 1),
            value_locations=value_locations,
            var_data_types=[DataType.DOUBLE],
        )
        szlw.zone_set_solution_time(
            solution_file_handle, sol_z, strand=1, solution_time=float(t)
        )

        # compute cell-centered count for an ordered zone (skip dims == 1)
        shape = (3, 3, 1)
        num_cells = 1
        for dim in shape:
            if dim == 1:
                continue
            num_cells *= dim - 1

        values = np.asarray([t * i for i in range(1, 10)], dtype=np.float64)[:num_cells]
        szlw.zone_write_double_values(
            solution_file_handle,
            sol_z,
            1,
            values,
        )
        szlw.close_file(solution_file_handle)

    szlw.close_file(grid_file_handle)


def test_ordered_ijk(file_name: str, ijk_dim: tuple[int, int, int]):
    var_names = ["x", "y", "z", "c"]
    file_handle = szlw.open_file(file_name, "Title", var_names)
    value_locations = [
        ValueLocation.NODAL,
        ValueLocation.NODAL,
        ValueLocation.NODAL,
        ValueLocation.CELL_CENTERED,
    ]
    var_data_types = [DataType.FLOAT] * len(var_names)
    zone = szlw.create_ordered_zone(
        file_handle,
        "Zone",
        ijk_dim,
        value_locations=value_locations,
        var_data_types=var_data_types,
    )

    x_ = np.linspace(0.0, ijk_dim[0] - 1, ijk_dim[0])
    y_ = np.linspace(0.0, ijk_dim[1] - 1, ijk_dim[1])
    z_ = np.linspace(0.0, ijk_dim[2] - 1, ijk_dim[2])
    x, y = np.meshgrid(x_, y_, indexing="xy")
    x = np.array([x] * ijk_dim[2])
    y = np.array([y] * ijk_dim[2])
    z = np.repeat(z_, ijk_dim[0] * ijk_dim[1])

    szlw.zone_write_float_values(file_handle, zone, 1, x.flatten().astype(np.float32))
    szlw.zone_write_float_values(file_handle, zone, 2, y.flatten().astype(np.float32))
    szlw.zone_write_float_values(file_handle, zone, 3, z.astype(np.float32))

    num_cells = 1
    for i in ijk_dim:
        if i == 1:
            continue
        num_cells *= i - 1
    szlw.zone_write_float_values(
        file_handle, zone, 4, np.linspace(0, 1, num_cells, dtype=np.float32)
    )
    szlw.close_file(file_handle)


if __name__ == "__main__":
    test()
    test_gridandsolution()
    test_ordered_ijk("ij_ordered.szplt", (3, 4, 1))
    test_ordered_ijk("jk_ordered.szplt", (1, 3, 4))
    test_ordered_ijk("ik_ordered.szplt", (3, 1, 5))
    test_ordered_ijk("ijk_ordered.szplt", (3, 4, 5))
