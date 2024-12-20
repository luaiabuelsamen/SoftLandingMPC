import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, HTMLWriter
from matplotlib import rc

from dynamics import RocketDynamics
from MPC import solve_cftoc

#OPEN LOOP FORMULATION
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

state_bounds = [(-np.pi, np.pi), (-10, 10),  # Roll, Roll Rate
                (-np.pi, np.pi), (-10, 10),  # Pitch, Pitch Rate
                (-np.pi, np.pi), (-10, 10),  # Yaw, Yaw Rate
                (-100, 10000), (-10, 10),      # x, x_dot
                (-100, 10000), (-10, 10),      # y, y_dot
                (0, 10000), (-10, 10)]         # z, z_dot

input_bounds = [(-50, 50), (-50, 50), (-50, 50), (0, 2000)]  # U1, U2, U3, U4

simulation_steps = 100
N = 10
states = [initial_state]
inputs = []


states, controls = solve_cftoc(rocket,
                      states[-1],
                      target_state=target_state,
                      target_input=target_input,
                      state_bounds=state_bounds,
                      input_bounds=input_bounds,
                      N=N,
                      N_u=N,
                      N_cy=N)


#OPEN LOOP PLOTS
x, y, z = states[:,6], states[:,8], states[:,10]
xdot, ydot, zdot = states[:,7], states[:,9], states[:,11]
roll, pitch, yaw = states[:,0], states[:,2], states[:,4]

# Plot 1: x, y, z vs t
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 10, x.size), x, label='x')
plt.plot(np.linspace(0, 10, y.size), y, label='y')
plt.plot(np.linspace(0, 10, z.size), z, label='z')
plt.title('Position (x, y, z) open loop vs Time (N={30})')
plt.xlabel('Time (t)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)

# Plot 2: xdot, ydot, zdot vs t
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 20, xdot.size), xdot, label=r'$\dot{x}$')
plt.plot(np.linspace(0, 20, ydot.size), ydot, label=r'$\dot{y}$')
plt.plot(np.linspace(0, 20, zdot.size), zdot, label=r'$\dot{z}$')
plt.title('Velocity ($\dot{x}$, $\dot{y}$, $\dot{z}$) vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)

# Plot 3: roll, pitch, yaw vs t
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 20, roll.size), roll, label='Roll $\phi$')
plt.plot(np.linspace(0, 20, pitch.size), pitch, label='Pitch $\theta$')
plt.plot(np.linspace(0, 20, yaw.size), yaw, label='Yaw $\psi$')
plt.title('Angles (Roll, Pitch, Yaw) vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Angle (radians)')
plt.legend()
plt.grid(True)

# Plot 4: Control inputs vs t
plt.figure(figsize=(10, 6))
for i in range(input_dim):
    plt.plot(np.linspace(0, 20, controls[i].size), controls[i], label=f'$U_{i+1}$')
plt.title('Control Inputs vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Control Inputs')
plt.legend()
plt.grid(True)

# Display all plots
plt.tight_layout()
plt.show()