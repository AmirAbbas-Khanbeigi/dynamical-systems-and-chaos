import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters of the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Lorenz system differential equations
def lorenz(t, state):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Initial conditions and time span
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 50)  # Start and end times
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Time points to evaluate

# Solving the Lorenz system
solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

# Extracting the solution for plotting
x, y, z = solution.y

# Plotting the Lorenz attractor
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, lw=0.5)

# Labels and title
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
