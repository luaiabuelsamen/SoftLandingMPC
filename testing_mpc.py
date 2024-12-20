# CLOSE LOOP FORMULATION
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from dynamics import RocketDynamics
from MPC import solve_cftoc


state_dim = 12
input_dim = 4
mass = 919.200  # kg
Ixx, Iyy, Izz = 330.427, 332.721, 334.932  # kg·m²
rocket = RocketDynamics(mass,
                        Ixx,
                        Iyy,
                        Izz)
target_state = np.zeros(state_dim)
target_input = np.array([0, 0, 0, rocket.mass * rocket.g])
initial_state = np.array([0, 0,       # φ, φ_dot
                          0.8863, 0,  # θ, θ_dot
                          -0.49, 0,   # ψ, ψ_dot
                          500, -100,  # x, x_dot
                          500, 30,    # y, y_dot
                          1500, 400]) # z, z_dot
rocket.state = initial_state.copy()

# Initial Conditions
initial_state = np.array([0, 0,       # φ, φ_dot
                          0.8863, 0,  # θ, θ_dot
                          -0.49, 0,   # ψ, ψ_dot
                          500, -100,  # x, x_dot
                          500, 30,    # y, y_dot
                          1500, 400]) # z, z_dot
# MPC Parameters
N = 10   # Horizon length
T_end = 10.0   # Total simulation time in seconds
Ts = rocket.Ts
num_steps = int(T_end / Ts)

# Targets and Bounds
state_dim = len(initial_state)
input_dim = 4

target_state = np.zeros(state_dim)
target_input = np.array([0, 0, 0, rocket.mass * rocket.g])  # Hover thrust

state_bounds = [
    (-np.pi, np.pi),   (-10, 10),     # φ, φ_dot
    (-np.pi, np.pi),   (-10, 10),     # θ, θ_dot
    (-np.pi, np.pi),   (-10, 10),     # ψ, ψ_dot
    (-100, 10000),     (-10, 10),     # x, x_dot
    (-100, 10000),     (-10, 10),     # y, y_dot
    (0, 10000),        (-10, 10)      # z, z_dot
]

input_bounds = [
    (-50, 50),   # U1
    (-50, 50),   # U2
    (-50, 50),   # U3
    (0, 20000)   # U4
]

# Storage for simulation
states_history = np.zeros((num_steps+1, state_dim))
inputs_history = np.zeros((num_steps, input_dim))

states_history[0, :] = initial_state

for step in range(num_steps):
    current_state = states_history[step, :].copy()

    # Solve MPC
    x_optimal, u_optimal = solve_cftoc(
        rocket,
        current_state,
        target_state=target_state,
        target_input=target_input,
        state_bounds=state_bounds,
        input_bounds=input_bounds,
        N=N,
        N_u=N,
        N_cy=N
    )

    u_applied = u_optimal[0, :]
    inputs_history[step, :] = u_applied
    next_state = rocket.Ad @ current_state + rocket.Bd @ u_applied
    rocket.state = next_state.copy()
    states_history[step+1, :] = next_state

t_vector = np.linspace(0, T_end, num_steps+1)

x, y, z = states_history[:,6], states_history[:,8], states_history[:,10]
xdot, ydot, zdot = states_history[:,7], states_history[:,9], states_history[:,11]
roll, pitch, yaw = states_history[:,0], states_history[:,2], states_history[:,4]

# Plot 1: x, y, z vs t
plt.figure(figsize=(10, 6))
plt.plot(t_vector, x, label='x')
plt.plot(t_vector, y, label='y')
plt.plot(t_vector, z, label='z')
plt.title(f'Position (x, y, z) vs Time (N={N})')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)

# Plot 2: xdot, ydot, zdot vs t
plt.figure(figsize=(10, 6))
plt.plot(t_vector, xdot, label=r'$\dot{x}$')
plt.plot(t_vector, ydot, label=r'$\dot{y}$')
plt.plot(t_vector, zdot, label=r'$\dot{z}$')
plt.title(f'Velocity vs Time (N={N})')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

# Plot 3: roll, pitch, yaw vs t
plt.figure(figsize=(10, 6))
plt.plot(t_vector, roll, label='Roll $\phi$')
plt.plot(t_vector, pitch, label='Pitch $\theta$')
plt.plot(t_vector, yaw, label='Yaw $\psi$')
plt.title(f'Angles (Roll, Pitch, Yaw) vs Time (N={N})')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)

# Plot 4: Control inputs vs t
t_control = np.linspace(0, T_end - Ts, num_steps)
plt.figure(figsize=(10, 6))
for i in range(input_dim):
    plt.plot(t_control, inputs_history[:,i], label=f'$U_{i+1}$')
plt.title(f'Control Inputs vs Time (N={N})')
plt.xlabel('Time (s)')
plt.ylabel('Control Input')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#3D animation FOR CLOSED LOOOP MPC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc

rc('animation', html='jshtml')

# Extract state history from the closed-loop simulation
x = states_history[:, 6]
y = states_history[:, 8]
z = states_history[:, 10]

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot elements
trajectory_line, = ax.plot([], [], [], color='cyan', linestyle='--', label='Trajectory')
current_point, = ax.plot([], [], [], 'ro', markersize=10, label='Current Position')
target_point = ax.scatter(0, 0, 0, color='gold', s=100, marker='*', label='Target Point')

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('Rocket Trajectory (Closed-Loop MPC)')

# Set axis limits using min and max to ensure entire trajectory is visible
ax.set_xlim(x.min() - 50, x.max() + 50)
ax.set_ylim(y.min() - 50, y.max() + 50)
ax.set_zlim(max(0, z.min() - 50), z.max() + 50)

# Adjust the aspect ratio
# Normalize ranges for x, y, z
x_range = x.max() - x.min() + 100
y_range = y.max() - y.min() + 100
z_range = z.max() - z.min() + 100
max_range = max(x_range, y_range, z_range)

ax.set_box_aspect((x_range / max_range, y_range / max_range, z_range / max_range))

ax.legend()

def init():
    trajectory_line.set_data([], [])
    trajectory_line.set_3d_properties([])
    current_point.set_data([], [])
    current_point.set_3d_properties([])
    return trajectory_line, current_point

def animate(i):
    # Update trajectory line to show path up to time i
    trajectory_line.set_data(x[:i+1], y[:i+1])
    trajectory_line.set_3d_properties(z[:i+1])

    # Current position
    current_point.set_data([x[i]], [y[i]])
    current_point.set_3d_properties([z[i]])

    return trajectory_line, current_point

anim = FuncAnimation(fig, animate, init_func=init,
                     frames=len(x), interval=100, blit=True, repeat=True)

# Display animation in a Jupyter notebook
anim
gif_path = "/content/rocket_animation N = {10}.gif" # change name based on Horizion (N)
anim.save(gif_path, writer='pillow', fps=10)

print(f"Animation saved as {gif_path}")