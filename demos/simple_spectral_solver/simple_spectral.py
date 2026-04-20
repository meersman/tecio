#!/usr/bin/env python3
# ruff: noqa: D100 D103

import os
# Prevent oversubscription
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
from tqdm import tqdm

import tecio

# Setup fftw performance flags
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = os.cpu_count()

# Domain
width = 2.0
height = 1.0

# Grid
nx = 512
ny = 256

x = np.linspace(0, width, nx, endpoint=False)
y = np.linspace(0, height, ny, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="ij")

# Spectral wavenumbers
kx = 2 * np.pi / width * np.fft.fftshift(np.arange(-nx//2, nx//2))
ky = 2 * np.pi / height * np.fft.fftshift(np.arange(-ny//2, ny//2))

KX, KY = np.meshgrid(kx, ky, indexing="ij")

# Combine spectral grid to solve x/y-momentum together
kn = np.zeros((nx, ny, 2))
kn[:,:,0] = KX
kn[:,:,1] = KY

lap = (1j*KX)**2 + (1j*KY)**2  # spectral laplacian
lap[lap == 0] = 1.0  # avoid division by zero (fix later for pressure)

# Time series
t = 0.0
time = np.arange(1e-10, 30.0, 0.05)

nu = 0.0009  # viscosity

# External forcing
f_x = 40.0 * np.exp(-1000 * ((X - 1.0)**2 + (Y - 0.5)**2))

# Spectral fields
u_hat = np.zeros((nx, ny, 2), dtype=complex)
f_hat = np.zeros((nx, ny, 2), dtype=complex)
f_hat[:,:,0] = fft.fft2(f_x)
p_hat = np.zeros((nx, ny), dtype=complex)
convect = np.zeros((nx, ny, 2), dtype=complex)

with tecio.open("simple_spectral.szplt", "w") as szl:

    for k in tqdm(range(len(time)), desc="Time Stepping", ncols=100):

        while t < time[k]:
            # Semi-Implicit Method for Pressure Linked Equations (SIMPLE)

            # # Velocity in physical space for nonlinear terms
            u = np.real(fft.ifft2(u_hat, axes=(0, 1)))

            # Transform nonlinear terms
            uu_hat = fft.fft2(u[:,:,0]*u[:,:,0])
            uv_hat = fft.fft2(u[:,:,0]*u[:,:,1])
            vv_hat = fft.fft2(u[:,:,1]*u[:,:,1])

            # x-momentum equation (u*del)u) = dx(uu) + dy(uv)
            convect[:,:,0] = 1j*KX*uu_hat + 1j*KY*uv_hat

            # y-momentum equatino (u*del)v = dx(uv) + dy(vv)
            convect[:,:,1] = 1j*KX*uv_hat + 1j*KY*vv_hat

            # Adaptive timestep (CFL)
            umax = np.max(np.abs(u))
            dt = min(0.5 * min(dx, dy) / (umax + 1e-8), time[k] - t)

            # Spectral velocity integration using Explicit Euler for convection and
            # modified Implicit Euler for diffusion
            u_hat = (u_hat*(1/dt + nu*lap[:,:,None]) + f_hat*np.sign(np.sin(t))
                     - convect) / (1/dt - nu*lap[:,:,None])

            # laplacian(p) = divergence(u)
            p_hat = np.sum(1j*kn*u_hat, axis=2) / lap
            p_hat[0, 0] = 0.0  # enforce mean-zero pressure

            # Correct spectral velocity field using spectral pressure gradients
            u_hat -= 1j * kn * p_hat[:,:,None]

            # Time update
            t += dt

        # vorticity = du/dy - dv/dx = ifft(ik_y u - ik_x v)
        vorticity = np.real(fft.ifft2(1j*KY*u_hat[:,:,0] - 1j*KX*u_hat[:,:,1]))

        # Write Tecplot zone
        szl.write_ijk_zone(
            title="Vorticity Field",
            variables=["x", "y", "omega"],
            data=[X, Y, vorticity] if szl.current_zone==0 else [vorticity],
            var_sharing=None if szl.current_zone==0 else [1, 1, 0],
            strand_id=1,
            solution_time=t,
        )

    print("Finalizing output file")

print("Done")
