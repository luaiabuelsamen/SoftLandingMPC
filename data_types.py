from dataclasses import dataclass
from utils import slerp
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

@dataclass
class RocketParameters:
    """Parameters for the 3D rocket landing model."""
    # Physical parameters
    g_I: np.ndarray  # gravity vector in inertial frame [m/s^2]
    J_B: np.ndarray  # inertia matrix in body frame [kg*m^2]
    r_T_B: np.ndarray  # position of the thrust point in the body frame [m]
    m_wet: float  # initial (wet) mass [kg]
    m_dry: float  # final (dry) mass [kg]
    T_min: float  # minimum thrust [N]
    T_max: float  # maximum thrust [N]
    
    # Control constraints
    gimbal_max: float  # maximum gimbal angle [rad]
    theta_max: float  # maximum tilt angle [rad]
    gamma_gs: float  # glideslope angle [rad]
    w_B_max: float  # maximum angular velocity [rad/s]
    
    # Initial and final states
    r_I_init: np.ndarray  # initial position [m]
    v_I_init: np.ndarray  # initial velocity [m/s]
    q_B_I_init: np.ndarray  # initial orientation (quaternion) [w, x, y, z]
    w_B_init: np.ndarray  # initial angular velocity [rad/s]
    
    r_I_final: np.ndarray  # final position [m]
    v_I_final: np.ndarray  # final velocity [m/s]
    q_B_I_final: np.ndarray  # final orientation (quaternion) [w, x, y, z]
    w_B_final: np.ndarray  # final angular velocity [rad/s]
    
    # Simulation parameters
    final_time: float  # final time [s]
    alpha_m: float = 0.0  # mass depletion rate parameter [1/s]
    
    # Scaling parameters (for non-dimensionalization)
    m_scale: float = 1.0
    r_scale: float = 1.0
    
    @property
    def tan_gamma_gs(self) -> float:
        """Tangent of the glideslope angle."""
        return np.tan(self.gamma_gs)
    
    @property
    def x_init(self) -> np.ndarray:
        """Initial state vector [m; r; v; q; w]."""
        return np.concatenate([
            [self.m_wet], 
            self.r_I_init, 
            self.v_I_init, 
            self.q_B_I_init, 
            self.w_B_init
        ])
    
    @property
    def x_final(self) -> np.ndarray:
        """Final state vector [m; r; v; q; w]."""
        return np.concatenate([
            [self.m_dry], 
            self.r_I_final, 
            self.v_I_final, 
            self.q_B_I_final, 
            self.w_B_final
        ])
    
    def nondimensionalize(self):
        """Convert parameters to non-dimensional form."""
        self.m_scale = self.m_wet
        self.r_scale = np.linalg.norm(self.r_I_init)
        
        # Scale physical parameters
        self.alpha_m *= self.r_scale
        self.r_T_B /= self.r_scale
        self.g_I /= self.r_scale
        self.J_B /= (self.m_scale * self.r_scale * self.r_scale)
        
        # Scale initial and final states
        self.m_wet /= self.m_scale
        self.m_dry /= self.m_scale
        
        self.r_I_init /= self.r_scale
        self.v_I_init /= self.r_scale
        
        self.r_I_final /= self.r_scale
        self.v_I_final /= self.r_scale
        
        # Scale thrust constraints
        self.T_min /= (self.m_scale * self.r_scale)
        self.T_max /= (self.m_scale * self.r_scale)
    
    def redimensionalize(self):
        """Convert parameters back to dimensional form."""
        # Reverse the scaling from nondimensionalize
        self.alpha_m /= self.r_scale
        self.r_T_B *= self.r_scale
        self.g_I *= self.r_scale
        self.J_B *= (self.m_scale * self.r_scale * self.r_scale)
        
        self.m_wet *= self.m_scale
        self.m_dry *= self.m_scale
        
        self.r_I_init *= self.r_scale
        self.v_I_init *= self.r_scale
        
        self.r_I_final *= self.r_scale
        self.v_I_final *= self.r_scale
        
        self.T_min *= (self.m_scale * self.r_scale)
        self.T_max *= (self.m_scale * self.r_scale)

class TrajectoryData:
    """Container for trajectory data."""
    
    def __init__(self, K: int, interpolate_input: bool = True):
        """Initialize trajectory data.
        
        Args:
            K: Number of discrete time steps
            interpolate_input: Whether to use input interpolation
        """
        self.X = np.zeros((K, 14))  # States: [m, r, v, q, w]
        n_u = K if interpolate_input else K-1
        self.U = np.zeros((n_u, 4))  # Controls: [thrust vector (3), torque]
        self.t = 0.0  # Total time
    
    @property
    def n_X(self) -> int:
        """Number of state vectors."""
        return self.X.shape[0]
    
    @property
    def n_U(self) -> int:
        """Number of control vectors."""
        return self.U.shape[0]
    
    @property
    def interpolated_input(self) -> bool:
        """Whether input interpolation is used."""
        return self.n_U == self.n_X
    
    def input_at_time(self, t: float) -> np.ndarray:
        """Get interpolated input at time t."""
        t = np.clip(t, 0.0, self.t)
        
        if t == self.t:
            return self.U[-1]
        
        dt = self.t / (self.n_X - 1)
        interp_value = (t % dt) / dt
        i = int(t / dt)
        
        u0 = self.U[i]
        u1 = self.U[i+1] if self.interpolated_input else self.U[i]
        
        return u0 + interp_value * (u1 - u0)
    
    def state_at_time(self, t: float) -> np.ndarray:
        """Get interpolated state at time t."""
        t = np.clip(t, 0.0, self.t)
        
        if t == self.t:
            return self.X[-1]
        
        dt = self.t / (self.n_X - 1)
        interp_value = (t % dt) / dt
        i = int(t / dt)
        
        x0 = self.X[i]
        x1 = self.X[i+1]
        
        # For quaternion, use spherical linear interpolation (SLERP)
        q0 = x0[7:11]
        q1 = x1[7:11]
        q_interp = slerp(q0, q1, interp_value)
        
        # Linear interpolation for other states
        x_interp = x0.copy()
        x_interp[:7] = x0[:7] + interp_value * (x1[:7] - x0[:7])
        x_interp[7:11] = q_interp
        x_interp[11:] = x0[11:] + interp_value * (x1[11:] - x0[11:])
        
        return x_interp

class DiscretizationData:
    """Container for discretization data."""
    
    def __init__(self, K: int, interpolate_input: bool = True, free_final_time: bool = False):
        """Initialize discretization data.
        
        Args:
            K: Number of discrete time steps
            interpolate_input: Whether to use input interpolation
            free_final_time: Whether final time is a free variable
        """
        self.A = [None] * (K-1)  # State transition matrices
        self.B = [None] * (K-1)  # Control matrices
        
        if interpolate_input:
            self.C = [None] * (K-1)  # Next control matrices
        else:
            self.C = []
            
        if free_final_time:
            self.s = [None] * (K-1)  # Scaling vectors
        else:
            self.s = []
            
        self.z = [None] * (K-1)  # Constant terms
    
    @property
    def interpolated_input(self) -> bool:
        """Whether input interpolation is used."""
        return len(self.C) > 0
    
    @property
    def variable_time(self) -> bool:
        """Whether final time is a free variable."""
        return len(self.s) > 0
    
    @property
    def n_X(self) -> int:
        """Number of state transition matrices."""
        return len(self.A)
    
    @property
    def n_U(self) -> int:
        """Number of control matrices."""
        return len(self.B)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)