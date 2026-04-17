# Create an animation of a Lorenz attractor

This demo generates a Lorenz attractor using a numerical RK4 integrator, then writes both full and time-dependent trajectory data using the TecIO Python API for visualization and animation in Tecplot.

---

## 1. Lorenz system + RK4 integration

We first define the Lorenz system:

```python
import numpy as np

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

def lorenz(state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])
```

Then integrate using classical RK4:

```python
dt = 0.01
n_steps = 5000

traj = np.zeros((n_steps, 3))
state = np.array([1.0, 1.0, 1.0])

for i in range(n_steps):
    traj[i] = state

    k1 = lorenz(state)
    k2 = lorenz(state + 0.5 * dt * k1)
    k3 = lorenz(state + 0.5 * dt * k2)
    k4 = lorenz(state + dt * k3)

    state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
```

This produces a full trajectory stored in `traj`.

---

## 2. Prepare data arrays

```python id="prep"
x = traj[:, 0]
y = traj[:, 1]
z = traj[:, 2]
```

---

## 3. Writing time-dependent Tecplot data with TecIO

We open a SZL file and write multiple zones:

* a full trajectory zone
* an animated “growing curve”
* a moving particle representation

```python
import tecio

with tecio.open("lorenz.szplt", "w") as szl:

    # Full trajectory (static reference zone)
    szl.write_ijk_zone(
        title="Lorenz Attractor",
        variables=["x", "y", "z", "t", "tau"],
        data=[x, y, z],
        passive_vars=[False, False, False, True, True],
        strand_id=0,
    )
```

---

### 3.1 Growing trajectory (time-dependent animation)

Each zone contains the trajectory up to time step *i*, with a solution time assigned:

```python
    for i in range(0, n_steps, 10):
        szl.write_ijk_zone(
            title="Trajectory",
            variables=["x", "y", "z", "t", "tau"],
            data=[
                x[:i+1],
                y[:i+1],
                z[:i+1],
                np.arange(i+1) * dt,
                (np.arange(i+1) - i - 1) * dt,
            ],
            strand_id=1,
            solution_time=i * dt,
        )
```

This creates a time-stranded dataset that Tecplot can animate over.

---

### 3.2 Moving particle representation

A second strand represents a single particle moving along the attractor:

```python
    for i in range(0, n_steps, 10):
        szl.write_ijk_zone(
            title="Particle",
            variables=["x", "y", "z", "t", "tau"],
            data=[
                x[i:i+1],
                y[i:i+1],
                z[i:i+1],
            ],
            passive_vars=[False, False, False, True, True],
            strand_id=2,
            solution_time=i * dt,
        )
```

This allows Tecplot to animate a point following the trajectory.

---

## 4. Create animation with Tecplot

To create the included animation run the included macro file in Tecplot batch mode from the command line

```bash
$ tec360 -b lorenz.mcr
```

Final result:
![lorenz-demo](lorenz.gif)