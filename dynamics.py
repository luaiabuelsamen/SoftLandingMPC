#rocket dynmaics
import numpy as np

class RocketDynamics:
    def __init__(self, mass,
                 Ixx,
                 Iyy,
                 Izz,
                 initial_state = None,
                 g=9.81,
                 Ts = 0.1
                 ):
        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.g = g

        #using the state defined above
        self.state = initial_state if initial_state else np.zeros(12)
        self.Ts = Ts
        self.Ad, self.Bd = self.discretize_dynamics()

    def non_linear_dynamics(self, U1, U2, U3, U4, theta, psi):
        x_ddot = (U4 / self.mass) * np.cos(theta) * np.cos(psi)
        y_ddot = (U4 / self.mass) * np.cos(theta) * np.sin(psi)
        z_ddot = -(U4 / self.mass) * np.sin(theta) - self.g

        phi_ddot = U1 / self.Ixx
        theta_ddot = U2 / self.Iyy
        psi_ddot = U3 / self.Izz

        return x_ddot, y_ddot, z_ddot, phi_ddot, theta_ddot, psi_ddot

    def linear_dynamics(self):
        # Define A (12x12) and B (12x4)
        A = np.zeros((12, 12))
        B = np.zeros((12, 4))

        # Angular dynamics
        A[0, 1] = 1  # d(phi)/dt = phi_dot
        A[2, 3] = 1  # d(theta)/dt = theta_dot
        A[4, 5] = 1  # d(psi)/dt = psi_dot

        B[1, 0] = 1 / self.Ixx  # phi_ddot depends on U1
        B[3, 1] = 1 / self.Iyy  # theta_ddot depends on U2
        B[5, 2] = 1 / self.Izz  # psi_ddot depends on U3

        # Linear position dynamics
        A[6, 7] = 1  # d(x)/dt = x_dot
        A[8, 9] = 1  # d(y)/dt = y_dot
        A[10, 11] = 1  # d(z)/dt = z_dot
        B[11, 3] = 1 / self.mass  # z_ddot depends on U4 (positive direction thrust)
        A[11, 11] = -self.g  # Add gravity as a constant downward acceleration

        return A, B
    def step_forward(self, U):
        self.state = self.Ad @ self.state + self.Bd @ np.array(U)
        return self.state
    def discretize_dynamics(self): # Using ZOH after linearization
        Ad = np.array([
            [1., 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0.1, 0., 0., 0., 0., 0., 0., 0.04905, 0.001635],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.981, 0.04905],
            [0., 0., 0., 0., 1., 0.1, 0., 0., -0.04905, -0.001635, 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., -0.981, -0.04905, 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0.1, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.1, 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.1],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
        ])

        Bd = np.array([
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.43951284e-06],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.08790257e-04],
            [0.00000000e+00, 0.00000000e+00, 1.22039698e-07, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 4.88158790e-06, 0.00000000e+00],
            [0.00000000e+00, -1.22850677e-07, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, -4.91402707e-06, 0.00000000e+00, 0.00000000e+00],
            [1.51319353e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [3.02638707e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 1.50276057e-05, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 3.00552114e-04, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 1.49284034e-05, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 2.98568068e-04, 0.00000000e+00]
        ])

        return Ad, Bd



