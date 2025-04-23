import numpy as np
import time

def tic():
    return time.time() * 1000.0

def toc(start_time):
    return time.time() * 1000.0 - start_time

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
    dot = np.sum(q0 * q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = np.sin(t * theta_0) / sin_theta_0
    
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy + qw*qz), 2*(qx*qz - qw*qy)],
        [2*(qx*qy - qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz + qw*qx)],
        [2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])


def omega_matrix(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    
    return np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])


def euler_to_quaternion(angles: np.ndarray) -> np.ndarray:
    phi, theta, psi = angles
    
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qw, qx, qy, qz])

def numerical_jacobian(f, t, x, u, params, eps=1e-7):
    """
    Compute the Jacobian matrices numerically using finite differences.
    This optimized version computes both state and control Jacobians with
    fewer function evaluations by using central differences for better accuracy.
    
    """
    n_x = len(x)
    n_u = len(u)
    
    A = np.zeros((n_x, n_x))
    B = np.zeros((n_x, n_u))

    for i in range(n_x):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = f(t, x_plus, u, params)
        f_minus = f(t, x_minus, u, params)
        A[:, i] = (f_plus - f_minus) / (2 * eps)
    
    for i in range(n_u):
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i] += eps
        u_minus[i] -= eps
        f_plus = f(t, x, u_plus, params)
        f_minus = f(t, x, u_minus, params)
        B[:, i] = (f_plus - f_minus) / (2 * eps)
    
    return A, B

