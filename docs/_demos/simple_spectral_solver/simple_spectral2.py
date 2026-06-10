#!/usr/bin/env python3
# ruff: noqa: D100, D103, E501
# fmt: off

import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates
from tqdm import tqdm

import tecio

# Threads for dft (-1 for os.cpu_count())
THREADS=-1

# Write interval (every X steps)
WRITE_INTERVAL = 10

#=======================================================================================

# Grid setup
# ----------
# Domain
width = 1.0
height = 1.0

# Resolution
nx = 256
ny = 256

# Mesh
x = np.linspace(0, width, nx, endpoint=False)
y = np.linspace(0, height, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

# Time series
# -----------
t = 0.0
time_array = np.arange(1e-10, 15.0, 0.01)
cfl = 0.1  # 256^2
# cfl = 0.15  # 512^2
# cfl = 0.3  # 1024^2

# Fluid properties
# ----------------
nu = 0.00003675  # viscosity

# Initial condition
# -----------------
L = 0.025
U1 = 0.5
U2 = -0.5
Um = (U1 - U2)/2
# Piecewise definition over y in [0, 1]
m1 = (Y >= 0) & (Y < 1/4)
m2 = (Y >= 1/4) & (Y < 1/2)
m3 = (Y >= 1/2) & (Y < 3/4)
m4 = (Y >= 3/4) & (Y <= 1)
u_init = np.zeros_like(X)
u_init[m1] = U1 - Um * np.exp((Y[m1] - 1/4) / L)
u_init[m2] = U2 + Um * np.exp((1/4 - Y[m2]) / L)
u_init[m3] = U2 + Um * np.exp((Y[m3] - 3/4) / L)
u_init[m4] = U1 - Um * np.exp((3/4 - Y[m4]) / L)
v_init = 0.01*np.sin(4*np.pi*X)
u_hat = fft.fft2(u_init, workers=THREADS)
v_hat = fft.fft2(v_init, workers=THREADS)

# # Create "color-dyed" regions,  1 in the inner jet region, 0 in outer stream
phi = np.zeros((nx, ny))
phi[(Y >= 1/4) & (Y < 3/4)] = 1.0

#=======================================================================================

# Calculate grid spacing
dx = x[1] - x[0]
dy = y[1] - y[0]
min_dxdy = min(dx, dy) # minimum for CFL based timestep

# Create spectral grid
kx = 2 * np.pi / width * np.fft.fftfreq(nx, d=1.0/nx)
ky = 2 * np.pi / height * np.fft.fftfreq(ny, d=1.0/ny)
KX, KY = np.meshgrid(kx, ky, indexing="ij")

# Spectral laplacian
lap = (1j*KX)**2 + (1j*KY)**2
lap[lap == 0] = 1.0  # avoid division by zero (fix later for pressure)

# Preallocate spectral fields
p_hat = np.zeros((nx, ny), dtype=complex)
conv_u = np.zeros((nx, ny), dtype=complex)
conv_v = np.zeros((nx, ny), dtype=complex)

# Precompute invariant terms in the momentum and pressure equation
ikx = 1j * KX
iky = 1j * KY
nu_lap = nu * lap
ikx_lap = ikx / lap
iky_lap = iky / lap
_uv_buf = np.empty((nx, ny), dtype=float)

with tecio.open(f"simple_spectral2_nx_{nx}_ny_{ny}.szplt", "w") as szl:
    # Set aux data for output variables
    szl.add_auxdataset_dict({
        "Common.XVar": 1,
        "Common.YVar": 2,
        "Common.UVar": 3,
        "Common.VVar": 4,
        "Common.CVar": 9,  # Set default to open with "tracer" variable
        "Nx": nx,
        "Ny": ny,
        "CFL": cfl,
        "Viscosity": nu,
        "U1": U1,
        "U2": U2,
        "L": L,
    })

    for k in tqdm(range(len(time_array)), desc="Spectral integration loop", ncols=100):
        count = 0
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
            dt = min(cfl*min_dxdy/(uvmax + 1e-8), time_array[k] - t)

            # Spectral velocity integration using Explicit Euler for convection and
            # modified Implicit Euler for diffusion
            u_hat = (u_hat*(1/dt + nu_lap) - conv_u) / (1/dt - nu_lap)
            v_hat = (v_hat*(1/dt + nu_lap) - conv_v) / (1/dt - nu_lap)

            # laplacian(p) = divergence(u)
            p_hat[:,:] = ikx_lap*u_hat + iky_lap*v_hat
            p_hat[0, 0] = 0.0  # enforce mean-zero pressure

            # Correct spectral velocity field using spectral pressure gradients
            u_hat -= ikx * p_hat
            v_hat -= iky * p_hat

            # Time update
            t += dt

            count +=1

        # Update the tracer
        if k > 0:
            dt_out = time_array[k] - time_array[k-1]   # update interval

            # Departure points in grid index space
            ix = (X - u * dt_out) / dx % nx
            iy = (Y - v * dt_out) / dy % ny

            phi = map_coordinates(phi, [ix.ravel(), iy.ravel()],
                                  order=1, mode='wrap').reshape(nx, ny)

        # Calculate flow quantities for output
        if k % WRITE_INTERVAL == 0:

            uvel = np.real(fft.ifft2(u_hat, workers=THREADS)).astype(np.float32)
            vvel = np.real(fft.ifft2(v_hat, workers=THREADS)).astype(np.float32)
            pres = np.real(fft.ifft2(p_hat, workers=THREADS)).astype(np.float32)

            # Vorticity = du/dy - dv/dx = ifft(iky u - ikx v)
            vort = np.real(fft.ifft2(iky*u_hat - ikx*v_hat, workers=THREADS)).astype(np.float32)

            # Divergence = du/dx + dv/dy = ifft(ikx u + iky v)
            div = np.real(fft.ifft2(ikx*u_hat + iky*v_hat, workers=THREADS)).astype(np.float32)

            # Q-criterion = 1/2[(du/dx)^2 + (dv/dy)^2 + 2(du/dx)(dv/dy)]
            dudx = np.real(fft.ifft2(ikx * u_hat, workers=THREADS))
            dudy = np.real(fft.ifft2(iky * u_hat, workers=THREADS))
            dvdx = np.real(fft.ifft2(ikx * v_hat, workers=THREADS))
            dvdy = np.real(fft.ifft2(iky * v_hat, workers=THREADS))
            qcrit = -(dudx**2 + dvdy**2 + 2*dudy*dvdx) / 2

            flow = [uvel, vvel, pres, vort, div, qcrit.astype(np.float32), phi.astype(np.float32)]

            # write Tecplot zone
            szl.write_ijk_zone(
                title=f"Flow Field Step {k}",
                variables=["x", "y", "uvel", "vvel", "pres", "vort", "div", "qcrit", "tracer"],
                data=[X, Y, *flow] if szl.current_zone==0 else flow,
                var_sharing=None if szl.current_zone==0 else [1, 1]+[0]*len(flow),
                strand_id=1,
                solution_time=time_array[k],
                aux={"dt": dt, "SubIterCount": count},
            )

    print("Finalizing output file")

print("Done")
