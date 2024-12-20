from pyomo.environ import *
import numpy as np

state_dim = 12
input_dim = 4


def solve_cftoc(rocket, current_state, target_state, target_input,
              state_bounds, input_bounds, N=20, N_u=1, N_cy=5):
    model = ConcreteModel()
    model.T = RangeSet(0, N-1)
    model.T_control = RangeSet(0, N-2)

    nx = len(current_state)
    nu = len(target_input)

    model.x = Var(range(nx), model.T, domain=Reals)
    model.u = Var(range(nu), model.T, domain=Reals)
    model.eps1 = Var(domain=NonNegativeReals)  # Slack variable 1
    model.eps2 = Var(domain=NonNegativeReals)  # Slack variable 2
    for i in range(nx):
        model.x[i,0].fix(current_state[i])

    # Weights from the problem formulation
    W = np.zeros(20)
    W[0] = W[2] = W[4] = 0      # phi, theta, ps i
    W[1] = 0                # phi_dot
    W[3] = W[5] = 0          # theta_dot, psi_dot
    W[6] = 100   # Higher weight on x position
    W[8] = 100   # Higher weight on y position
    W[10] = 100  # Higher weight on z position
    W[7] = 20    # x_dot
    W[9] = 20    # y_dot
    W[11] = 20   # z_dot
    W[12:16] = 0.01  # Lower weights on control inputs
    W[16:20] = 0.001  # Lower weights on ΔU



    # DEFINE COST
    def objective_rule(m):
        state_cost = sum(
            W[i] * (m.x[i,t] - target_state[i])**2
            for i in range(nx) for t in m.T
        )

        input_cost = sum(
            W[i+12] * (m.u[i,t] - target_input[i])**2
            for i in range(nu) for t in m.T
        )

        delta_u_cost = sum(
            W[i+16] * (m.u[i,t] - m.u[i,t-1])**2
            for i in range(nu) for t in m.T if t > 0
        )

        slack_cost = 1000 * m.eps1 + 1000 * m.eps2
        terminal_cost = sum(
            10000 * W[i] * (model.x[i, N-1] - target_state[i])**2 for i in range(nx)
        )


        return state_cost + input_cost + delta_u_cost + slack_cost +terminal_cost#+ final_state_cost

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # DEFINE DYNAMICS CONSTRAINT
    def dynamics_rule(m, i, t):
        if t < N-1:
            state_next = sum(rocket.Ad[i,j] * m.x[j,t] for j in range(nx)) + \
                        sum(rocket.Bd[i,j] * m.u[j,t] for j in range(nu))
            return m.x[i,t+1] == state_next
        return Constraint.Skip


    model.dynamics = Constraint(range(nx), model.T, rule=dynamics_rule)

    # DEFINE CONSTRAINTS
    for i, (lb, ub) in enumerate(state_bounds):
        if i == 10:
            # Skip altitude here and handle it separately
            continue
        for t in model.T:
            model.add_component(
                f'state_lb_{i}_{t}',
                Constraint(expr=model.x[i,t] >= lb - model.eps1)
            )
            model.add_component(
                f'state_ub_{i}_{t}',
                Constraint(expr=model.x[i,t] <= ub + model.eps1)
            )

    # Now add a separate hard constraint for z ≥ 0 (and any upper bound if needed)
    for t in model.T:
        model.add_component(
            f'z_lb_{t}',
            Constraint(expr=model.x[10,t] >= 0)  # hard constraint, no slack
        )

    solver = SolverFactory('ipopt')

    results = solver.solve(model)

    u_optimal = np.array([[value(model.u[i,t]) for i in range(nu)]
                         for t in range(N)])

    x_optimal = np.array([[value(model.x[i,t]) for i in range(nx)]
                    for t in range(N)])

    return x_optimal, u_optimal
