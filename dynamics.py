import numpy as np
from data_types import RocketParameters, TrajectoryData, DiscretizationData
from utils import omega_matrix, quaternion_to_rotation_matrix, numerical_jacobian
from scipy.integrate import solve_ivp

def rocket_dynamics(t: float, x: np.ndarray, u: np.ndarray, params: RocketParameters) -> np.ndarray:
    m = x[0]
    r = x[1:4]
    v = x[4:7]
    q = x[7:11]
    w = x[11:14]
    
    thrust = u[:3]
    torque = np.array([0, 0, u[3]])
    
    R_I_B = quaternion_to_rotation_matrix(q)

    m_dot = -params.alpha_m * np.linalg.norm(thrust)
    r_dot = v
    v_dot = (1.0/m) * (R_I_B @ thrust) + params.g_I
    q_dot = 0.5 * omega_matrix(w) @ q
    
    J_B_inv = np.diag(1.0 / params.J_B)
    w_dot = J_B_inv @ (np.cross(params.r_T_B, thrust) + torque - np.cross(w, params.J_B * w))
    
    return np.concatenate([[m_dot], r_dot, v_dot, q_dot, w_dot])


def simulate(model_params: RocketParameters, dt: float, u0: np.ndarray, 
             u1: np.ndarray, x: np.ndarray) -> np.ndarray:

    def ode_func(t, x):
        u = u0 + (t / dt) * (u1 - u0)
        return rocket_dynamics(t, x, u, model_params)
    
    sol = solve_ivp(
        ode_func,
        (0, dt),
        x,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    return sol.y[:, -1]


def multiple_shooting(model_params: RocketParameters, td: TrajectoryData, 
                     dd: DiscretizationData) -> None:
    """
    Compute multiple shooting discretization.
    """
    for k in range(td.n_X - 1):
        x_k = td.X[k]
        u_k = td.U[k]
        u_kp1 = td.U[k+1] if dd.interpolated_input else u_k
        
        V = np.zeros((14, 1 + 14 + 4 + dd.interpolated_input * 4 + dd.variable_time + 1))
        V[:, 0] = x_k
        V[:, 1:15] = np.eye(14)
        
        dt = td.t / (td.n_X - 1)
        
        if dd.variable_time:
            dt_normalized = 1.0 / (td.n_X - 1)
        else:
            dt_normalized = dt

        def ode_func(t, V_flat):
            V_matrix = V_flat.reshape(V.shape)
            x = V_matrix[:, 0]

            if dd.interpolated_input:
                u = u_k + (t / dt_normalized) * (u_kp1 - u_k)
            else:
                u = u_k
                
            f = rocket_dynamics(t, x, u, model_params)

            A, B = numerical_jacobian(rocket_dynamics, t, x, u, model_params)
            
            if dd.variable_time:
                A *= td.t
                B *= td.t
            
            Phi_A = V_matrix[:, 1:15]
            Phi_A_inv = np.linalg.inv(Phi_A)
            
            dVdt = np.zeros_like(V_matrix)

            if dd.variable_time:
                dVdt[:, 0] = td.t * f
            else:
                dVdt[:, 0] = f

            dVdt[:, 1:15] = A @ Phi_A
            
            col_idx = 15

            if dd.interpolated_input:
                alpha = (dt_normalized - t) / dt_normalized
                dVdt[:, col_idx:col_idx+4] = Phi_A_inv @ B * alpha
                col_idx += 4
                
                beta = t / dt_normalized
                dVdt[:, col_idx:col_idx+4] = Phi_A_inv @ B * beta
                col_idx += 4
            else:
                dVdt[:, col_idx:col_idx+4] = Phi_A_inv @ B
                col_idx += 4

            if dd.variable_time:
                dVdt[:, col_idx] = Phi_A_inv @ f
                col_idx += 1
                dVdt[:, col_idx] = Phi_A_inv @ (-A @ x - B @ u)
            else:
                dVdt[:, col_idx] = Phi_A_inv @ (f - A @ x - B @ u)
                
            return dVdt.flatten()
        
        V_flat_init = V.flatten()

        sol = solve_ivp(
            ode_func,
            (0, dt_normalized),
            V_flat_init,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        V_result = sol.y[:, -1].reshape(V.shape)
        
        col_idx = 1

        dd.A[k] = V_result[:, col_idx:col_idx+14].copy()
        col_idx += 14
        
        dd.B[k] = dd.A[k] @ V_result[:, col_idx:col_idx+4]
        col_idx += 4

        if dd.interpolated_input:
            dd.C[k] = dd.A[k] @ V_result[:, col_idx:col_idx+4]
            col_idx += 4

        if dd.variable_time:
            dd.s[k] = dd.A[k] @ V_result[:, col_idx:col_idx+1]
            col_idx += 1

        dd.z[k] = dd.A[k] @ V_result[:, col_idx]
