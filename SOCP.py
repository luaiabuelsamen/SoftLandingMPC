from dynamics import multiple_shooting, simulate
from data_types import RocketParameters, TrajectoryData, DiscretizationData, Arrow3D
import time
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from utils import  quaternion_to_rotation_matrix, tic, toc, slerp
from matplotlib import animation



class SCAlgorithm:
    """Successive Convexification Algorithm for optimal trajectory planning."""
    
    def __init__(self, model: RocketParameters, K: int = 20, 
                free_final_time: bool = True, interpolate_input: bool = True):
        """Initialize the algorithm.
        
        Args:
            model: Rocket parameters
            K: Number of discrete time steps
            free_final_time: Whether final time is a free variable
            interpolate_input: Whether to use input interpolation
        """
        self.model = model
        self.K = K
        self.free_final_time = free_final_time
        self.interpolate_input = interpolate_input
        
        # Algorithm parameters
        self.weight_time = 1.0
        self.weight_trust_region_time = 1.0
        self.weight_trust_region_trajectory = 1.0
        self.weight_virtual_control = 1000.0
        
        self.delta_tol = 1e-3
        self.nu_tol = 1e-5
        self.max_iterations = 15
        
        # Initialize trajectory and discretization data
        self.td = TrajectoryData(K, interpolate_input)
        self.dd = DiscretizationData(K, interpolate_input, free_final_time)
        
        # Store all trajectories for visualization
        self.all_trajectories = []
    
    def initialize(self):
        """Initialize the trajectory."""
        print("Initializing trajectory...")
        
        # Initialize trajectory with straight-line interpolation
        self._initialize_trajectory()
        
        print("Trajectory initialized.")
    
    def _initialize_trajectory(self):
        """Initialize trajectory with straight-line interpolation between boundary conditions."""
        for k in range(self.K):
            alpha1 = (self.K - k) / self.K
            alpha2 = k / self.K
            
            # Mass, position and linear velocity - linear interpolation
            self.td.X[k, 0] = alpha1 * self.model.x_init[0] + alpha2 * self.model.x_final[0]
            self.td.X[k, 1:7] = alpha1 * self.model.x_init[1:7] + alpha2 * self.model.x_final[1:7]
            
            # Quaternion - spherical linear interpolation (SLERP)
            q0 = self.model.x_init[7:11]
            q1 = self.model.x_final[7:11]
            self.td.X[k, 7:11] = slerp(q0, q1, alpha2)
            
            # Angular velocity - linear interpolation
            self.td.X[k, 11:14] = alpha1 * self.model.x_init[11:14] + alpha2 * self.model.x_final[11:14]
        
        # Initialize control inputs
        thrust_initial = np.array([0., 0., (self.model.T_max + self.model.T_min) / 2])
        for k in range(self.td.n_U):
            self.td.U[k] = np.append(thrust_initial, 0.0)  # [thrust_x, thrust_y, thrust_z, torque]
        
        # Initialize trajectory time
        self.td.t = self.model.final_time
    
    def solve(self, warm_start=False):
        """Solve the optimal control problem using successive convexification.
        
        Args:
            warm_start: Whether to use the current trajectory as initial guess
        """
        print(f"Solving model with successive convexification...")
        
        if not warm_start:
            self.initialize()
        
        self.all_trajectories = [self._copy_trajectory(self.td)]
        
        start_time = time.time()
        
        iteration = 0
        converged = False
        
        while iteration < self.max_iterations and not converged:
            iteration += 1
            
            print(f"Iteration {iteration}/{self.max_iterations}")
            
            converged = self._iterate()
            
            # Store trajectory for visualization
            self.all_trajectories.append(self._copy_trajectory(self.td))
        
        solve_time = time.time() - start_time
        
        if converged:
            print(f"Converged after {iteration} iterations. Time: {solve_time:.2f}s")
        else:
            print(f"Max iterations reached without convergence. Time: {solve_time:.2f}s")
    
    def _iterate(self):
        """Perform one iteration of successive convexification.
        
        Returns:
            bool: Whether the algorithm has converged
        """
        # Discretize the system
        t_disc = time.time()
        multiple_shooting(self.model, self.td, self.dd)
        print(f"Discretization time: {(time.time() - t_disc)*1000:.2f}ms")
        
        # solve the problem
        print("\n")
        print("Solving problem.\n")
        timer = tic()
        success = self._solve_socp()
        print(f"SOCP solve time: {toc(timer):.2f}ms")
        
        if not success:
            print("Solver failed to find a solution. Terminating.")
            return False
        
        # Calculate defects
        defects = self._calculate_defects()
        defect_max = max(defects)
        print(f"Maximum defect: {defect_max:.4e}")
        
        # Calculate trust region violation
        trust_region_delta = self._calculate_trust_region_delta()
        print(f"Trust region delta: {trust_region_delta:.4e}")
        
        # Get virtual control norm
        norm1_nu = self.socp_result["norm1_nu"]
        print(f"Virtual control norm: {norm1_nu:.4e}")
        
        # Update trust region weight if virtual control is small
        if norm1_nu < self.nu_tol:
            self.weight_trust_region_trajectory *= 2.0
            print(f"Increasing trust region weight to {self.weight_trust_region_trajectory}")
        
        # Check for convergence
        converged = (trust_region_delta < self.delta_tol and norm1_nu < self.nu_tol)
        
        print(f"Trajectory time: {self.td.t:.4f}s")
        
        return converged
    
    def _solve_socp(self):
        """Solve the convex subproblem using SOCP.
        
        The SOCP (Second-Order Cone Programming) problem is formulated with:
        - Variables for states, controls, virtual control and trust regions
        - Linearized dynamics constraints
        - Boundary conditions and path constraints
        - Trust region constraints
        - L1 norm minimization for virtual control
        """
        # Create optimization variables
        X = cp.Variable((self.K, 14))
        U = cp.Variable((self.td.n_U, 4))
        nu = cp.Variable((self.K-1, 14))  # Virtual control
        nu_bound = cp.Variable((self.K-1, 14), nonneg=True)  # Virtual control bounds
        norm1_nu = cp.Variable(nonneg=True)  # L1 norm of virtual control
        delta = cp.Variable(self.K, nonneg=True)  # Trust region for state and input
        
        # Create sigma variable for free final time
        if self.free_final_time:
            sigma = cp.Variable(1)
            delta_sigma = cp.Variable(1, nonneg=True)
        
        # Define constraints
        constraints = []
        
        # Dynamics constraints using multiple shooting
        for k in range(self.K-1):
            # Linearized dynamics: x_{k+1} = A_k * x_k + B_k * u_k + C_k * u_{k+1} + s_k * sigma + z_k + nu_k
            rhs = self.dd.A[k] @ X[k] + self.dd.B[k] @ U[k] + self.dd.z[k] + nu[k]
            
            if self.dd.interpolated_input:
                rhs += self.dd.C[k] @ U[k+1]
                
            if self.free_final_time:
                rhs += self.dd.s[k] * sigma
                
            constraints.append(X[k+1] == rhs)
        
        # Initial and final state constraints
        constraints.append(X[0] == self.model.x_init)
        constraints.append(X[-1, 1:7] == self.model.x_final[1:7])  # Position and velocity
        constraints.append(X[-1, 11:14] == self.model.x_final[11:14])  # Angular velocity
        
        # Final control constraints
        constraints.append(U[-1, :3] == 0)  # Zero thrust at final time
        constraints.append(U[-1, 3] == 0)   # Zero torque at final time
        
        # State constraints
        
        # Mass must be greater than dry mass
        constraints.append(X[:, 0] >= self.model.m_dry)
        
        # Glide slope constraint
        for k in range(self.K):
            # lateral_position = sqrt(x^2 + y^2)
            # height = z
            # lateral_position <= tan(gamma_gs) * height
            constraints.append(cp.norm(X[k, 1:3], 2) <= self.model.tan_gamma_gs * X[k, 3])
        
        # Maximum tilt angle constraint (using quaternion)
        # Approximation: ||(qx, qy)|| <= sqrt((1-cos(theta_max))/2)
        tilt_const = np.sqrt((1.0 - np.cos(self.model.theta_max)) / 2.0)
        for k in range(self.K):
            constraints.append(cp.norm(X[k, 8:10], 2) <= tilt_const)
        
        # Maximum angular velocity
        for k in range(self.K):
            constraints.append(cp.norm(X[k, 11:14], 2) <= self.model.w_B_max)
        
        # Control constraints
        
        # Thrust magnitude constraints
        for k in range(self.td.n_U):
            constraints.append(cp.norm(U[k, :3], 2) <= self.model.T_max)
            
            # Minimum thrust constraint (linearized)
            if k < self.K-1:  # Skip final control
                thrust_dir = self.td.U[k, :3] / np.linalg.norm(self.td.U[k, :3])
                constraints.append(thrust_dir @ U[k, :3] >= self.model.T_min)
        
        # Gimbal angle constraint
        # ||thrust_xy|| <= tan(gimbal_max) * thrust_z
        gimbal_const = np.tan(self.model.gimbal_max)
        for k in range(self.td.n_U-1):  # Skip final control
            constraints.append(cp.norm(U[k, :2], 2) <= gimbal_const * U[k, 2])
        
        # Virtual control constraints
        constraints.append(cp.sum(nu_bound) <= norm1_nu)
        for k in range(self.K-1):
            constraints.append(nu[k] <= nu_bound[k])
            constraints.append(-nu_bound[k] <= nu[k])
        
        # Trust region constraints for trajectory
        for k in range(self.K):
            # State trust region
            state_deviation = X[k] - self.td.X[k]
            
            # Add control deviation if applicable
            if k < self.td.n_U:
                control_deviation = U[k] - self.td.U[k]
                
                # We need to ensure the dimensions match for concatenation
                # CVXPY requires all dimensions except axis 0 to match
                # For vectors, this means they should both be column vectors or row vectors
                # Let's make sure they're handled as 1D vectors for the norm calculation
                constraints.append(
                    cp.norm(cp.hstack([cp.norm(state_deviation, 2), cp.norm(control_deviation, 2)]), 2) <= delta[k]
                )
            else:
                constraints.append(cp.norm(state_deviation, 2) <= delta[k])
                
        # Free final time constraints
        if self.free_final_time:
            # Linearized trust region for sigma
            # (sigma - sigma0)^2 <= delta_sigma
            # Reformulated as:
            # norm([0.5 - 0.5*delta_sigma, (sigma0 - sigma)]) <= 0.5 + 0.5*delta_sigma
            constraints.append(sigma >= 1.0)  # Minimum trajectory time
            constraints.append(
                cp.norm(cp.vstack([0.5 - 0.5 * delta_sigma, self.td.t - sigma]), 2) 
                <= 0.5 + 0.5 * delta_sigma
            )
        
        # Objective function
        objective = self.weight_virtual_control * norm1_nu + self.weight_trust_region_trajectory * cp.sum(delta)
        
        if self.free_final_time:
            objective += self.weight_time * sigma + self.weight_trust_region_time * delta_sigma
        
        # Create and solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=True)
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Problem status is {prob.status}")
            
            # Store results directly as their values, not as CVXPY variables
            self.socp_result = {
                "X": X.value,
                "U": U.value,
                "nu": nu.value,
                "nu_bound": nu_bound.value,
                "norm1_nu": norm1_nu.value.item(),  # More robust than float()
                "delta": delta.value
            }
            
            if self.free_final_time:
                self.socp_result["sigma"] = sigma.value.item()
                self.socp_result["delta_sigma"] = delta_sigma.value.item()
            
            # Update trajectory data
            self.td.X = X.value
            self.td.U = U.value
            
            if self.free_final_time:
                self.td.t = sigma.value.item()
                
            return True

        except Exception as e:
            print(f"Solver failed: {e}")
            return False

    
    def _calculate_defects(self):
        """Calculate integration defects between discretization points."""
        defects = []
        
        for k in range(self.K-1):
            # Get current state and controls
            x_k = self.td.X[k]
            u_k = self.td.U[k]
            u_kp1 = self.td.U[k+1] if self.dd.interpolated_input else u_k
            
            # Simulate forward one step
            dt = self.td.t / (self.K - 1)
            x_kp1_sim = simulate(self.model, dt, u_k, u_kp1, x_k)
            
            # Calculate defect
            defect = np.linalg.norm(x_kp1_sim - self.td.X[k+1])
            defects.append(defect)
            
        return defects
    
    def _calculate_trust_region_delta(self):
        """Calculate the trust region violation."""
        return np.sum(self.socp_result["delta"])
    
    def _copy_trajectory(self, td):
        """Create a deep copy of trajectory data."""
        td_copy = TrajectoryData(self.K, self.interpolate_input)
        td_copy.X = td.X.copy()
        td_copy.U = td.U.copy()
        td_copy.t = td.t
        return td_copy
    
    def get_solution(self):
        """Get the solution trajectory."""
        return self.td
    
    def visualize_trajectory(self, figure_num=None):
        """Visualize the trajectory in 3D."""
        fig = plt.figure(figure_num, figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot position trajectory
        ax.plot(self.td.X[:, 1], self.td.X[:, 2], self.td.X[:, 3], 'k.-', label='Position')
        
        # Plot thrust vectors
        scale = 0.1  # Scale for visualization
        for k in range(0, self.td.n_U, max(1, self.td.n_U // 10)):  # Plot fewer arrows for clarity
            pos = self.td.X[k, 1:4]
            q = self.td.X[k, 7:11]
            R = quaternion_to_rotation_matrix(q)
            thrust = self.td.U[k, :3]
            
            if np.linalg.norm(thrust) > 0:
                thrust_normalized = thrust / np.linalg.norm(thrust) * scale
                thrust_I = -R @ thrust_normalized  # Negative sign because thrust acts in opposite direction
                ax.quiver(pos[0], pos[1], pos[2], 
                         thrust_I[0], thrust_I[1], thrust_I[2], 
                         color='r', arrow_length_ratio=0.2)
        
        # Plot orientation
        for k in range(0, self.K, max(1, self.K // 10)):
            pos = self.td.X[k, 1:4]
            q = self.td.X[k, 7:11]
            R = quaternion_to_rotation_matrix(q)
            
            # Body x-axis (forward)
            axis_x = R @ np.array([scale, 0, 0])
            ax.quiver(pos[0], pos[1], pos[2], axis_x[0], axis_x[1], axis_x[2], color='b')
            
            # Body z-axis (up)
            axis_z = R @ np.array([0, 0, scale])
            ax.quiver(pos[0], pos[1], pos[2], axis_z[0], axis_z[1], axis_z[2], color='g')
        
        # Plot start and end points
        ax.scatter(self.td.X[0, 1], self.td.X[0, 2], self.td.X[0, 3], 
                  c='g', marker='o', s=100, label='Start')
        ax.scatter(self.td.X[-1, 1], self.td.X[-1, 2], self.td.X[-1, 3], 
                  c='r', marker='o', s=100, label='End')
        
        # Set labels and limits
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # Equal aspect ratio
        max_range = np.max([
            np.ptp(self.td.X[:, 1]),
            np.ptp(self.td.X[:, 2]),
            np.ptp(self.td.X[:, 3])
        ])
        mid_x = np.mean([np.min(self.td.X[:, 1]), np.max(self.td.X[:, 1])])
        mid_y = np.mean([np.min(self.td.X[:, 2]), np.max(self.td.X[:, 2])])
        mid_z = np.mean([np.min(self.td.X[:, 3]), np.max(self.td.X[:, 3])])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(0, mid_z + max_range)
        
        ax.legend()
        plt.title(f'Rocket Trajectory (t = {self.td.t:.2f}s)')
        fig.savefig('trajectory', dpi=300, bbox_inches='tight')
        return fig, ax
    
    def visualize_iteration(self, iteration=None):
        """Visualize a specific iteration of the solver."""
        if iteration is None:
            iteration = len(self.all_trajectories) - 1
        
        iteration = min(max(0, iteration), len(self.all_trajectories) - 1)
        
        td = self.all_trajectories[iteration]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot position trajectory
        ax.plot(td.X[:, 1], td.X[:, 2], td.X[:, 3], 'k.-', label='Position')
        
        # Plot start and end points
        ax.scatter(td.X[0, 1], td.X[0, 2], td.X[0, 3], 
                  c='g', marker='o', s=100, label='Start')
        ax.scatter(td.X[-1, 1], td.X[-1, 2], td.X[-1, 3], 
                  c='r', marker='o', s=100, label='End')
        
        # Set labels and limits
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # Equal aspect ratio
        max_range = np.max([
            np.ptp(td.X[:, 1]),
            np.ptp(td.X[:, 2]),
            np.ptp(td.X[:, 3])
        ])
        mid_x = np.mean([np.min(td.X[:, 1]), np.max(td.X[:, 1])])
        mid_y = np.mean([np.min(td.X[:, 2]), np.max(td.X[:, 2])])
        mid_z = np.mean([np.min(td.X[:, 3]), np.max(td.X[:, 3])])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(0, mid_z + max_range)
        
        ax.legend()
        plt.title(f'Rocket Trajectory - Iteration {iteration} (t = {td.t:.2f}s)')
        fig.savefig('iterations', dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_iterations(self):
        """Plot all iterations of the solution."""
        if not self.all_trajectories:
            print("No trajectories to plot.")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.all_trajectories)))
        
        for i, td in enumerate(self.all_trajectories):
            ax.plot(td.X[:, 1], td.X[:, 2], td.X[:, 3], 
                   color=colors[i], alpha=0.5, linewidth=1 + i/5,
                   label=f'Iteration {i}')
        
        # Plot start and end points of final trajectory
        td = self.all_trajectories[-1]
        ax.scatter(td.X[0, 1], td.X[0, 2], td.X[0, 3], 
                  c='g', marker='o', s=100, label='Start')
        ax.scatter(td.X[-1, 1], td.X[-1, 2], td.X[-1, 3], 
                  c='r', marker='o', s=100, label='End')
        
        # Set labels
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # Equal aspect ratio
        max_range = np.max([
            np.ptp(td.X[:, 1]),
            np.ptp(td.X[:, 2]),
            np.ptp(td.X[:, 3])
        ])
        mid_x = np.mean([np.min(td.X[:, 1]), np.max(td.X[:, 1])])
        mid_y = np.mean([np.min(td.X[:, 2]), np.max(td.X[:, 2])])
        mid_z = np.mean([np.min(td.X[:, 3]), np.max(td.X[:, 3])])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(0, mid_z + max_range)
        
        plt.title('Rocket Trajectory - All Iterations')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        fig.savefig('Iterations', dpi=300, bbox_inches='tight')
        return fig, ax
    
    def plot_controls(self):
        """Plot the control inputs over time."""
        if not self.td:
            print("No trajectory to plot.")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # Time points
        dt = self.td.t / (self.K - 1)
        time_points = np.linspace(0, self.td.t, self.td.n_U)
        
        # Thrust magnitude
        thrust_mag = np.linalg.norm(self.td.U[:, :3], axis=1)
        axes[0].plot(time_points, thrust_mag)
        axes[0].set_ylabel('Thrust [N]')
        axes[0].grid(True)
        
        # Thrust direction (x, y components)
        axes[1].plot(time_points, self.td.U[:, 0], label='Thrust X')
        axes[1].plot(time_points, self.td.U[:, 1], label='Thrust Y')
        axes[1].set_ylabel('Lateral Thrust [N]')
        axes[1].grid(True)
        axes[1].legend()
        
        # Thrust direction (z component)
        axes[2].plot(time_points, self.td.U[:, 2])
        axes[2].set_ylabel('Vertical Thrust [N]')
        axes[2].grid(True)
        
        # Torque
        axes[3].plot(time_points, self.td.U[:, 3])
        axes[3].set_ylabel('Torque [Nm]')
        axes[3].set_xlabel('Time [s]')
        axes[3].grid(True)
        
        plt.tight_layout()
        fig.savefig('Controls', dpi=300, bbox_inches='tight')
        return fig, axes
    
    def plot_states(self):
        """Plot the state variables over time."""
        if not self.td:
            print("No trajectory to plot.")
            return
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        
        # Time points
        dt = self.td.t / (self.K - 1)
        time_points = np.linspace(0, self.td.t, self.K)
        
        # Mass
        axes[0].plot(time_points, self.td.X[:, 0])
        axes[0].set_ylabel('Mass [kg]')
        axes[0].grid(True)
        
        # Position
        axes[1].plot(time_points, self.td.X[:, 1], label='X')
        axes[1].plot(time_points, self.td.X[:, 2], label='Y')
        axes[1].plot(time_points, self.td.X[:, 3], label='Z')
        axes[1].set_ylabel('Position [m]')
        axes[1].grid(True)
        axes[1].legend()
        
        # Velocity
        axes[2].plot(time_points, self.td.X[:, 4], label='Vx')
        axes[2].plot(time_points, self.td.X[:, 5], label='Vy')
        axes[2].plot(time_points, self.td.X[:, 6], label='Vz')
        axes[2].set_ylabel('Velocity [m/s]')
        axes[2].grid(True)
        axes[2].legend()
        
        # Quaternion
        axes[3].plot(time_points, self.td.X[:, 7], label='qw')
        axes[3].plot(time_points, self.td.X[:, 8], label='qx')
        axes[3].plot(time_points, self.td.X[:, 9], label='qy')
        axes[3].plot(time_points, self.td.X[:, 10], label='qz')
        axes[3].set_ylabel('Quaternion')
        axes[3].grid(True)
        axes[3].legend()
        
        # Angular velocity
        axes[4].plot(time_points, self.td.X[:, 11], label='ωx')
        axes[4].plot(time_points, self.td.X[:, 12], label='ωy')
        axes[4].plot(time_points, self.td.X[:, 13], label='ωz')
        axes[4].set_ylabel('Angular Velocity [rad/s]')
        axes[4].set_xlabel('Time [s]')
        axes[4].grid(True)
        axes[4].legend()
        
        plt.tight_layout()
        fig.savefig('States', dpi=300, bbox_inches='tight')
        return fig, axes

    def create_rocket_animation(self, filename='rocket_trajectory.gif', fps=15, dpi=150):
        """Create a GIF of the rocket trajectory with a proper 3D rocket model.
        
        Args:
            filename: Output GIF filename
            fps: Frames per second
            dpi: Dots per inch for the animation
        """
        from matplotlib.animation import FuncAnimation
        import numpy as np
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Set up the figure and 3D axis
        fig = plt.figure(figsize=(12, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')
        
        # Calculate the trajectory time step
        dt = self.td.t / (self.K - 1)
        
        # Calculate frame count for smooth animation
        n_frames = int(5 * fps)  # 5 second animation
        
        # Get the trajectory data
        X = self.td.X
        U = self.td.U
        
        # Set up axis limits and labels
        max_range = np.max([
            np.ptp(X[:, 1]),
            np.ptp(X[:, 2]),
            np.ptp(X[:, 3])
        ])
        mid_x = np.mean([np.min(X[:, 1]), np.max(X[:, 1])])
        mid_y = np.mean([np.min(X[:, 2]), np.max(X[:, 2])])
        mid_z = np.mean([np.min(X[:, 3]), np.max(X[:, 3])])
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(0, mid_z + max_range)  # Start from 0
        
        # Set styles for space-like environment
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        
        ax.set_xlabel('X [m]', color='white')
        ax.set_ylabel('Y [m]', color='white')
        ax.set_zlabel('Z [m]', color='white')
        
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        # Plot the full trajectory as a reference
        trajectory_line = ax.plot(X[:, 1], X[:, 2], X[:, 3], 'w-', alpha=0.3, linewidth=1)[0]
        
        # Start and end points
        ax.scatter(X[0, 1], X[0, 2], X[0, 3], c='lime', marker='o', s=100)
        ax.scatter(X[-1, 1], X[-1, 2], X[-1, 3], c='red', marker='o', s=100)
        
        # Scale for rocket
        rocket_height = max_range * 0.04
        rocket_radius = rocket_height * 0.2
        
        # Create rocket body model
        def create_rocket_model(rocket_height, rocket_radius):
            """Create the rocket model vertices and faces."""
            h = rocket_height
            r = rocket_radius
            
            # Number of points around the circumference
            n = 16
            
            # Create cylinder for main body
            theta = np.linspace(0, 2*np.pi, n)
            x = np.cos(theta) * r
            y = np.sin(theta) * r
            
            # Bottom ring
            bottom_ring = np.column_stack([x, y, np.zeros_like(x)])
            
            # Top ring (80% of the height)
            top_ring = np.column_stack([x, y, np.ones_like(x) * 0.8 * h])
            
            # Create nosecone
            nose_tip = np.array([0, 0, h])
            
            # Create fins
            fin_height = 0.4 * h
            fin_width = 0.15 * h
            fin_offset = 0.1 * h  # Distance from bottom
            
            fin_vertices = []
            n_fins = 4
            for i in range(n_fins):
                angle = 2 * np.pi * i / n_fins
                # Base of fin at side of rocket
                x1, y1 = r * np.cos(angle), r * np.sin(angle)
                
                # Create two vertices at the bottom of the fin
                fin_vertices.append([x1, y1, fin_offset])
                fin_vertices.append([x1, y1, fin_offset + fin_height])
                
                # Create outer vertex
                x2, y2 = (r + fin_width) * np.cos(angle), (r + fin_width) * np.sin(angle)
                fin_vertices.append([x2, y2, fin_offset])
            
            fin_vertices = np.array(fin_vertices)
            
            # Define faces (triangles or quads)
            faces = []
            
            # Body cylinder faces
            for i in range(n):
                j = (i + 1) % n
                # Quad face connecting bottom and top rings
                faces.append([bottom_ring[i], bottom_ring[j], top_ring[j], top_ring[i]])
            
            # Bottom circle face
            faces.append(bottom_ring)
            
            # Nosecone faces
            for i in range(n):
                j = (i + 1) % n
                faces.append([top_ring[i], top_ring[j], nose_tip])
            
            # Fin faces
            for i in range(n_fins):
                idx = i * 3
                # Triangle face for each fin
                faces.append([fin_vertices[idx], fin_vertices[idx+1], fin_vertices[idx+2]])
            
            return faces
        
        # Create rocket faces
        rocket_faces = create_rocket_model(rocket_height, rocket_radius)
        
        # Initialize empty plots for the animation elements
        trail_line, = ax.plot([], [], [], 'r-', linewidth=2)
        title = ax.set_title('', color='white', fontsize=12)
        
        # Initialize variables for objects that need to be updated
        rocket_body = None
        thrust_arrow = None
        
        # Trail length
        trail_length = min(20, self.K)
        
        # Initialize the animation
        def init():
            trail_line.set_data([], [])
            trail_line.set_3d_properties([])
            title.set_text('')
            return trail_line, title
        
        # Transform faces based on position and orientation
        def transform_rocket(faces, pos, R):
            transformed_faces = []
            for face in faces:
                # For a list of points (face), transform each point
                if isinstance(face, np.ndarray) and face.ndim == 2:
                    # Handle polygon face (e.g., bottom circle)
                    transformed = np.array([R @ point + pos for point in face])
                    transformed_faces.append(transformed)
                else:
                    # Handle individual points for triangles/quads
                    transformed = [R @ np.array(point) + pos for point in face]
                    transformed_faces.append(transformed)
            return transformed_faces
        
        # Update function for the animation
        def update(frame):
            nonlocal rocket_body, thrust_arrow
            
            # Remove previous objects if they exist
            if rocket_body:
                rocket_body.remove()
            if thrust_arrow:
                thrust_arrow.remove()
            
            # Calculate the current time index
            t_idx = min(int(frame / n_frames * self.K), self.K - 1)
            
            # Current position
            pos = X[t_idx, 1:4]
            
            # Current orientation (quaternion)
            q = X[t_idx, 7:11]
            R = quaternion_to_rotation_matrix(q)
            
            # In body frame, up is along x-axis, we need to realign to z-axis in the interface
            # Define rotation to align rocket model with body frame
            R_align = np.array([
                [0, 0, 1],  # Rocket z-axis maps to body x-axis
                [0, 1, 0],  # Rocket y-axis maps to body y-axis
                [-1, 0, 0]  # Rocket x-axis maps to negative body z-axis
            ])
            
            # Combined rotation
            R_combined = R @ R_align
            
            # Update trail
            start_idx = max(0, t_idx - trail_length + 1)
            trail_x = X[start_idx:t_idx+1, 1]
            trail_y = X[start_idx:t_idx+1, 2]
            trail_z = X[start_idx:t_idx+1, 3]
            trail_line.set_data(trail_x, trail_y)
            trail_line.set_3d_properties(trail_z)
            
            # Transform rocket faces based on current position and orientation
            transformed_faces = transform_rocket(rocket_faces, pos, R_combined)
            
            # Create 3D rocket body
            rocket_body = Poly3DCollection(transformed_faces, alpha=0.9, linewidths=0.5, edgecolors='white')
            rocket_body.set_facecolor('#3878B2')  # Blue color for rocket
            ax.add_collection3d(rocket_body)
            
            # Get thrust at current time
            if t_idx < self.td.n_U:
                thrust = U[t_idx, :3]
            else:
                thrust = U[-1, :3]
            
            # Create thrust vector if magnitude is positive
            thrust_mag = np.linalg.norm(thrust)
            if thrust_mag > 0:
                # Scale thrust vector for visualization
                thrust_scale = rocket_height * 1.5 * thrust_mag / self.model.T_max
                
                # Thrust direction in body frame (negative z-axis)
                thrust_dir_body = np.array([0, 0, -1])
                thrust_dir_I = R @ thrust_dir_body
                
                # Thrust origin is at the bottom of the rocket
                thrust_origin = pos + R_combined @ np.array([0, 0, -rocket_height/2])
                
                # Add thrust arrow
                thrust_arrow = ax.quiver(
                    thrust_origin[0], thrust_origin[1], thrust_origin[2],
                    thrust_dir_I[0], thrust_dir_I[1], thrust_dir_I[2],
                    color='orange', alpha=0.8, arrow_length_ratio=0.15,
                    length=thrust_scale, normalize=True, linewidth=2
                )
                
                # Add thrust exhaust glow (a scatter point at thrust origin)
                ax.scatter(
                    thrust_origin[0], thrust_origin[1], thrust_origin[2],
                    c='yellow', s=50, alpha=0.8
                )
            
            # Update title
            current_time = t_idx * dt
            title.set_text(f'Rocket Trajectory (t = {current_time:.2f}s / {self.td.t:.2f}s)')
            
            return trail_line, title, rocket_body
        
        # Create the animation
        animation = FuncAnimation(fig, update, frames=n_frames, init_func=init, 
                                interval=1000/fps, blit=False)
        
        # Save the animation as a GIF using pillow
        animation.save(filename, writer='pillow', fps=fps, dpi=dpi)
        
        plt.close(fig)
        print(f"GIF saved to {filename}")
        
        return filename