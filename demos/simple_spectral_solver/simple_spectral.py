#!/usr/bin/env python3
# ruff: noqa: D100, D103

import numpy as np
from scipy import fft
from tqdm import tqdm

import tecio

# Threads for dft (-1 for os.cpu_count())
THREADS=4

# Write interval (every X steps)
write_interval = 4

# Domain
width = 2.0
height = 1.0

# Grid
nx = 512
ny = 256

x = np.linspace(0, width, nx, endpoint=False)
y = np.linspace(0, height, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

dx = x[1] - x[0]
dy = y[1] - y[0]
min_dxdy = min(dx, dy) # minimum for CFL based timestep

# Spectral wavenumbers
kx = 2 * np.pi / width * np.fft.fftfreq(nx, d=1.0/nx)
ky = 2 * np.pi / height * np.fft.fftfreq(ny, d=1.0/ny)
KX, KY = np.meshgrid(kx, ky, indexing="ij")

# Precompute ikx and iky for spectral differentiation
ikx = 1j * KX
iky = 1j * KY

lap = (1j*KX)**2 + (1j*KY)**2  # spectral laplacian
lap[lap == 0] = 1.0  # avoid division by zero (fix later for pressure)

# Time series
t = 0.0
time_array = np.arange(1e-10, 30.0, 0.05)

nu = 0.0009  # viscosity

# External forcing
fx = 40.0 * np.exp(-1000 * ((X - 1.0)**2 + (Y - 0.5)**2))
fy = np.zeros_like(X)

# Preallocate spectral fields
u_hat = np.zeros((nx, ny), dtype=complex)
v_hat = np.zeros((nx, ny), dtype=complex)
fx_hat = np.zeros((nx, ny), dtype=complex)
fy_hat = np.zeros((nx, ny), dtype=complex)
fx_hat = fft.fft2(fx)
fy_hat = fft.fft2(fy)
p_hat = np.zeros((nx, ny), dtype=complex)
conv_u = np.zeros((nx, ny), dtype=complex)
conv_v = np.zeros((nx, ny), dtype=complex)

# Precompute invariant terms in the momentum and pressure equation
nu_lap = nu * lap
ikx_lap = ikx/lap
iky_lap = iky/lap
_uv_buf = np.empty((nx, ny), dtype=float)

with tecio.open("simple_spectral.szplt", "w") as szl:

    for k in tqdm(range(len(time_array)), desc="Time Stepping", ncols=100):

        while t < time_array[k]:
            # Semi-Implicit Method for Pressure Linked Equations (SIMPLE)

            # Velocity in physical space for nonlinear terms
            u = np.real(fft.ifft2(u_hat, workers=THREADS))
            v = np.real(fft.ifft2(v_hat, workers=THREADS))

            # Transform nonlinear terms
            uu_hat = fft.fft2(u*u, workers=THREADS)
            uv_hat = fft.fft2(u*v, workers=THREADS)
            vv_hat = fft.fft2(v*v, workers=THREADS)

            # x-momentum equation (u*del)u) = dx(uu) + dy(uv)
            conv_u[:,:] = ikx*uu_hat + iky*uv_hat

            # y-momentum equatino (u*del)v = dx(uv) + dy(vv)
            conv_v[:,:] = ikx*uv_hat + iky*vv_hat

            # Adaptive timestep (CFL)
            np.hypot(u, v, out=_uv_buf)  # sqrt(u^2 + v^2) — magnitude
            uvmax = _uv_buf.max()
            dt = min(0.5 * min_dxdy / (uvmax + 1e-8), time_array[k] - t)

            # Spectral velocity integration using Explicit Euler for convection and
            # modified Implicit Euler for diffusion
            u_hat = (u_hat*(1/dt + nu_lap) + fx_hat*np.sign(np.sin(t))
                     - conv_u) / (1/dt - nu_lap)
            v_hat = (v_hat*(1/dt + nu_lap) + fy_hat*np.sign(np.sin(t))
                     - conv_v) / (1/dt - nu_lap)

            # laplacian(p) = divergence(u)
            # p_hat = (ikx*u_hat[:,:,0] + iky*u_hat[:,:,1]) / lap
            p_hat[:,:] = ikx_lap*u_hat + iky_lap*v_hat
            p_hat[0, 0] = 0.0  # enforce mean-zero pressure

            # Correct spectral velocity field using spectral pressure gradients
            u_hat -= ikx * p_hat
            v_hat -= iky * p_hat

            # Time update
            t += dt


        # Calculate flow quantities for output
        if k%write_interval==0:

            uvel = np.real(fft.ifft2(u_hat, workers=THREADS))
            vvel = np.real(fft.ifft2(v_hat, workers=THREADS))
            pres = np.real(fft.ifft2(p_hat, workers=THREADS))

            # Vorticity = du/dy - dv/dx = ifft(iky u - ikx v)
            vort = np.real(fft.ifft2(iky*u_hat - ikx*v_hat, workers=THREADS))

            # Divergence = du/dx + dv/dy = ifft(ikx u + iky v)
            div = np.real(fft.ifft2(ikx*u_hat + iky*v_hat, workers=THREADS))

            # Write Tecplot zone
            szl.write_ijk_zone(
                title=f"Flow Field Step {k}",
                variables=["x", "y", "uvel", "vvel", "pres", "vort", "div"],
                data=[X, Y, uvel, vvel, pres, vort, div] if szl.current_zone==0 else [uvel, vvel, pres, vort, div],
                var_sharing=None if szl.current_zone==0 else [1, 1, 0, 0, 0, 0, 0],
                strand_id=1,
                solution_time=t,
            )

    print("Finalizing output file")

print("Done")
