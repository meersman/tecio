#!/usr/bin/env python3
# ruff: noqa: D100, D103, E501

import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates
from tqdm import tqdm

import tecio

# Threads for dft (-1 for os.cpu_count())
THREADS=-1

# Write interval (every X steps)
write_interval = 5

# Domain
width = 1.0
height = 1.0

# Grid
# nx = 1024
# ny = 1024
nx = 256
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
time_array = np.arange(1e-10, 15.0, 0.01)

# nu = 0.0009  # viscosity
nu = 0.00003675  # viscosity

def spectral_forcing(t):
    # External forcing
    # Opposing offset jets — creates large-scale rotation
    f_amp   = 60.0
    f_sigma = 0.002   # jet width

    # Left jet — positioned above centreline, pointing right (+x)
    # x_l, y_l = 0.15, height/2 + 0.15
    # Right jet — positioned below centreline, pointing left (-x)
    x_r, y_r = 0.25, height/2

    # gauss_l = f_amp * np.exp(-((X - x_l)**2 + (Y - y_l)**2) / f_sigma)
    # gauss_r = f_amp * np.exp(-((X - x_r)**2 + (Y - y_r)**2) / f_sigma)

    # left jet points right, right jet points left
    # fx = gauss_l - gauss_r
    if t <= 1:
        fx = f_amp * np.exp(-((X - x_r)**2 + (Y - y_r)**2) / f_sigma)
    else:
        fx = np.zeros((nx, ny))
    fy = np.zeros_like(fx)
    # fx = 40.0 * np.exp(-1000 * ((X - 1.0)**2 + (Y - 0.5)**2))
    # fy = np.zeros_like(X)
    return fft.fft2(fx, workers=THREADS), fft.fft2(fy, workers=THREADS)

# Initial condition
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
u_init[m1] = U1 - Um * np.exp( (Y[m1] - 1/4) / L)
u_init[m2] = U2 + Um * np.exp(-(Y[m2] - 1/4) / L)
u_init[m3] = U2 + Um * np.exp( (Y[m3] - 3/4) / L)   # note: -(3/4 - y) = y - 3/4
u_init[m4] = U1 - Um * np.exp(-(Y[m4] - 3/4) / L)
v_init = 0.01*np.sin(4*np.pi*X)
u_hat = fft.fft2(u_init, workers=THREADS)
v_hat = fft.fft2(v_init, workers=THREADS)

# # Create "color-dyed regions",  1 in the inner jet region, 0 in outer streams
phi = np.zeros((nx, ny))
phi[(Y >= 1/4) & (Y < 3/4)] = 1.0
# phi_hat = fft.fft2(phi, workers=THREADS)


# Preallocate spectral fields
p_hat = np.zeros((nx, ny), dtype=complex)
conv_u = np.zeros((nx, ny), dtype=complex)
conv_v = np.zeros((nx, ny), dtype=complex)

# Precompute invariant terms in the momentum and pressure equation
nu_lap = nu*lap
ikx_lap = ikx/lap
iky_lap = iky/lap
_uv_buf = np.empty((nx, ny), dtype=float)

with tecio.open("simple_spectral.szplt", "w") as szl:
    # # Set aux data for output variables
    # tecio.szl.write_dataset_aux_data(
    #     szl.handle,
    #     {
    #         "Common.XVar": 1,
    #         "Common.YVar": 2,
    #         "Common.UVar": 3,
    #         "Common.VVar": 4,
    #         "Common.VVar": 4,
    #         "Common.PressureVar": 5,
    #     },
    # )

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
            dt = min(0.1 * min_dxdy / (uvmax + 1e-8), time_array[k] - t)

            # Calculate forcing term
            # fx_hat, fy_hat = spectral_forcing(t)

            # Spectral velocity integration using Explicit Euler for convection and
            # modified Implicit Euler for diffusion
            # u_hat = (u_hat*(1/dt + nu_lap) + fx_hat - conv_u) / (1/dt - nu_lap)
            # v_hat = (v_hat*(1/dt + nu_lap) + fy_hat - conv_v) / (1/dt - nu_lap)
            u_hat = (u_hat*(1/dt + nu_lap) - conv_u) / (1/dt - nu_lap)
            v_hat = (v_hat*(1/dt + nu_lap) - conv_v) / (1/dt - nu_lap)

            # laplacian(p) = divergence(u)
            # p_hat = (ikx*u_hat[:,:,0] + iky*u_hat[:,:,1]) / lap
            p_hat[:,:] = ikx_lap*u_hat + iky_lap*v_hat
            p_hat[0, 0] = 0.0  # enforce mean-zero pressure

            # Correct spectral velocity field using spectral pressure gradients
            u_hat -= ikx * p_hat
            v_hat -= iky * p_hat

            # Time update
            t += dt

        # Update the tracer
        if k > 0:
            dt_out = time_array[k] - time_array[k-1]   # output interval

            # Departure points in grid index space
            ix = (X - u * dt_out) / dx % nx
            iy = (Y - v * dt_out) / dy % ny

            phi = map_coordinates(phi, [ix.ravel(), iy.ravel()],
                                  order=1, mode='wrap').reshape(nx, ny)

        # Calculate flow quantities for output
        if k%write_interval==0:

            uvel = np.real(fft.ifft2(u_hat, workers=THREADS))
            vvel = np.real(fft.ifft2(v_hat, workers=THREADS))
            pres = np.real(fft.ifft2(p_hat, workers=THREADS))

            # Vorticity = du/dy - dv/dx = ifft(iky u - ikx v)
            vort = np.real(fft.ifft2(iky*u_hat - ikx*v_hat, workers=THREADS))

            # Divergence = du/dx + dv/dy = ifft(ikx u + iky v)
            div = np.real(fft.ifft2(ikx*u_hat + iky*v_hat, workers=THREADS))

            # Q-criterion = 1/2[(du/dx)^2 + (dv/dy)^2 + 2(du/dx)(dv/dy)]
            dudx = np.real(fft.ifft2(ikx * u_hat, workers=THREADS))
            dudy = np.real(fft.ifft2(iky * u_hat, workers=THREADS))
            dvdx = np.real(fft.ifft2(ikx * v_hat, workers=THREADS))
            dvdy = np.real(fft.ifft2(iky * v_hat, workers=THREADS))
            qcrit = -(dudx**2 + dvdy**2 + 2*dudy*dvdx) / 2

            # write Tecplot zone
            szl.write_ijk_zone(
                title=f"Flow Field Step {k}",
                variables=["x", "y", "uvel", "vvel", "pres", "vort", "div", "qcrit", "tracer"],
                data=[X, Y, uvel, vvel, pres, vort, div, qcrit, phi] if szl.current_zone==0 else [uvel, vvel, pres, vort, div, qcrit, phi],
                var_sharing=None if szl.current_zone==0 else [1, 1, 0, 0, 0, 0, 0, 0, 0],
                strand_id=1,
                solution_time=t,
                aux={"dt":dt},
            )

    print("Finalizing output file")

print("Done")
