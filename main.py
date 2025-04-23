import numpy as np
from dynamics import RocketParameters
import matplotlib.pyplot as plt
from utils import euler_to_quaternion
from SOCP import SCAlgorithm

if __name__ == "__main__":
    rocket_params = RocketParameters(
    g_I=np.array([0.0, 0.0, -9.81]),
    J_B=np.array([5000000.0, 5000000.0, 70000.0]),
    r_T_B=np.array([0.0, 0.0, -15.0]),
    m_wet=24000.0,
    m_dry=22000.0,
    T_min=200000.0,
    T_max=420000.0,
    
    gimbal_max=np.deg2rad(15.0),
    theta_max=np.deg2rad(90.0),
    gamma_gs=np.deg2rad(30.0),
    w_B_max=np.deg2rad(60.0),
    
    # Initial state: offset position, descending with tilt
    r_I_init=np.array([200.0, 200.0, 800.0]),
    v_I_init=np.array([-40.0, -40.0, -80.0]),
    q_B_I_init=euler_to_quaternion(np.deg2rad([-20.0, 20.0, 0.0])),
    w_B_init=np.array([0.0, 0.0, 0.0]),
    
    # Final state: landed position, zero velocity, upright
    r_I_final=np.array([0.0, 0.0, 0.0]),
    v_I_final=np.array([0.0, 0.0, 0.0]),
    q_B_I_final=euler_to_quaternion(np.array([0.0, 0.0, 0.0])),
    w_B_final=np.array([0.0, 0.0, 0.0]),
    
    final_time=12.0,
    alpha_m=1.0 / (275.0 * 9.81)  # I_sp = 275s
)


    # Create solver
    solver = SCAlgorithm(
        model=rocket_params,
        K=20,
        free_final_time=True,
        interpolate_input=True
    )
    
    # Solve the problem
    solver.solve()
    video_file = solver.create_rocket_animation()
    # # Plot results
    solver.visualize_trajectory()
    solver.plot_iterations()
    solver.plot_controls()
    solver.plot_states()
    
    # plt.show()