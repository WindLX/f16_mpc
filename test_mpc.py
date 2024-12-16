import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from f16_mpc.mpc import MPC
from f16_mpc.linear import LinearModel


def openloop_simulation(model: LinearModel, num_steps: int):
    states = np.zeros((num_steps, 2))
    controls = np.zeros((num_steps, 1))
    states[0] = model.x0

    controls[10:] = 1

    for t in range(1, num_steps):
        states[t] = model(states[t - 1], controls[t - 1])

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(states[:, 0], label="State 1")
    plt.plot(states[:, 1], label="State 2")
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    plt.title("Open-loop Response of the Linear Model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    init_state = np.array([0, 0])
    init_control = np.array([0])
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[1], [0.5]])
    model = LinearModel(init_state, init_control, A, B)

    openloop_simulation(model, 50)

    Q = np.array([[1, 0], [0, 0]])
    R = np.eye(1)
    C = np.eye(2)
    u_min = np.array([-1])
    u_max = np.array([1])
    # y_min = np.array([-1, -1])
    # y_max = np.array([10, 1])
    prediction_horizon = 10
    solver_options = {
        "step_size": 0.1,
        "max_iters": 1000,
        "tol": 1e-6,
        "log_interval": 100,
    }
    mpc = MPC(
        model,
        Q,
        R,
        C,
        u_min=u_min,
        u_max=u_max,
        # y_min,
        # y_max,
        prediction_horizon=prediction_horizon,
        solver_options=solver_options,
    )

    x0 = np.array([0, 0])
    x_ref = np.array([10, 0])

    num_steps = 50
    states = np.zeros((num_steps, 2))
    controls = np.zeros((num_steps, 1))
    states[0] = x0

    for t in range(1, num_steps):
        u0 = np.zeros(prediction_horizon)
        u, _ = mpc.solver.solve(
            mpc.objective_func(states[t - 1], x_ref),
            mpc.gradient_func(states[t - 1], x_ref),
            mpc.constraints(states[t - 1]),
            u0,
        )
        controls[t - 1] = u[0]
        states[t] = model(states[t - 1], controls[t - 1])

    sns.set_theme(style="darkgrid")
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(states[:, 0], label="State 1")
    axs[0].axhline(y=x_ref[0], color="r", linestyle="--", label="Reference State 1")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("State 1 Value")
    axs[0].set_title("Closed-loop Response of State 1 with MPC")
    axs[0].legend()

    axs[1].plot(states[:, 1], label="State 2")
    axs[1].axhline(y=x_ref[1], color="r", linestyle="--", label="Reference State 2")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("State 2 Value")
    axs[1].set_title("Closed-loop Response of State 2 with MPC")
    axs[1].legend()

    axs[2].plot(controls, label="Control Action")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Control Value")
    axs[2].set_title("Control Actions over Time")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
