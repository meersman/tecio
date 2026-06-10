#!/usr/bin/env python3
# ruff: noqa: D100 D103

import numpy as np
from tqdm import tqdm

import tecio

# Parameters
M = 401
N = 401

G = 0.01
M1 = 1.0
M2 = 1.0

c = 1.0

dx = 2.0 / (M - 1)
x = np.linspace(-1, 1, M)
y = np.linspace(-1, 1, N)

X, Y = np.meshgrid(x, y, indexing="ij")

dt = 0.125 * dx / c
t_end = 10.0
write_interval = 10

# Fields
f = np.zeros((M, N, 3))  # wave field (3 time levels)
absorb = np.zeros((M, N))
a = np.zeros((M, N))

# Absorption mask
for i in range(1, M - 1):
    for j in range(1, N - 1):
        r = np.sqrt(x[i] ** 2 + y[j] ** 2)
        if r < 0.8:
            absorb[i, j] = 1.0
        else:
            n = 0.045
            absorb[i, j] = 0.8**n / r**n

# Time loop
t = 0.0
n_steps = int(t_end / dt)

with tecio.open("gravity_waves.szplt", "w") as szl:
    for k in tqdm(range(n_steps), ncols=100, desc="Time stepping"):
        # Wave equation update
        f[1:-1, 1:-1, 2] = (
            dt**2
            * (
                c**2
                * (f[1:-1, 0:-2, 1] - 2 * f[1:-1, 1:-1, 1] + f[1:-1, 2:, 1])
                / dx**2
                + c**2
                * (f[0:-2, 1:-1, 1] - 2 * f[1:-1, 1:-1, 1] + f[2:, 1:-1, 1])
                / dx**2
            )
            + 2 * f[1:-1, 1:-1, 1]
            - f[1:-1, 1:-1, 0]
        )

        t += dt
        a.fill(0.0)

        # Moving "black holes"
        radius = 0.025
        omega = 4.0

        xp = radius * np.sin(2 * np.pi * omega * t)
        yp = radius * np.cos(2 * np.pi * omega * t)

        # potential field
        pot = G * M1 / np.sqrt((xp - X) ** 2 + (yp - Y) ** 2) + G * M2 / np.sqrt(
            (-xp - X) ** 2 + (-yp - Y) ** 2
        )

        # Absorption
        f *= absorb[:, :, None]

        # Time shift (back store)
        f[:, :, 0:2] = f[:, :, 1:3]

        # Impose potential constraint
        mask = pot > c**2
        a[mask] = -(c**2)

        f[:, :, 0][mask] = -(c**2)
        f[:, :, 1][mask] = -(c**2)

        # Jacobi smoothing
        if k % write_interval == 0:
            f[1:-1, 1:-1, 1] = 0.25 * (
                f[1:-1, 0:-2, 1] + f[1:-1, 2:, 1] + f[0:-2, 1:-1, 1] + f[2:, 1:-1, 1]
            )

            f[1:-1, 1:-1, 0] = 0.25 * (
                f[1:-1, 0:-2, 0] + f[1:-1, 2:, 0] + f[0:-2, 1:-1, 0] + f[2:, 1:-1, 0]
            )

            # Write Tecplot zone
            szl.write_ijk_zone(
                title="Wave Field",
                variables=["x", "y", "phi"],
                data=[X, Y, f[:, :, 2]] if szl.current_zone == 0 else [f[:, :, 2]],
                var_sharing=None if szl.current_zone == 0 else [1, 1, 0],
                strand_id=1,
                solution_time=t,
            )

    print("Finalizing output file")

print("Done")
