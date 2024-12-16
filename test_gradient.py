import argparse
import logging

import pyf16
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from f16_mpc.linear import F16LinearModel, DiscreteF16LinearModel
from f16_mpc.mpc import MPC
from f16_mpc.solver import (
    ProjectionGradientDescentSolver,
    MomentumGradientDescentSolver,
    AdamSolver,
)
from f16_mpc.render import F16TcpRender


state_names = [
    "npos",
    "epos",
    "altitude",
    "phi",
    "theta",
    "psi",
    "velocity",
    "alpha",
    "beta",
    "p",
    "q",
    "r",
]

control_names = [
    "thrust",
    "elevator",
    "aileron",
    "rudder",
]


def plot(
    states_data: list[np.ndarray],
    control_data: list[np.ndarray],
):

    state_data = pd.DataFrame(states_data, columns=state_names)
    state_data["time_step"] = range(state_data.shape[0])

    control_data = pd.DataFrame(control_data, columns=control_names)
    control_data["time_step"] = range(control_data.shape[0])

    # Convert angles from radians to degrees
    angle_columns = ["phi", "theta", "psi", "alpha", "beta", "p", "q", "r"]
    state_data[angle_columns] = state_data[angle_columns].apply(np.rad2deg)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for i, state in enumerate(state_names):
        sns.lineplot(data=state_data, x="time_step", y=state, ax=axes[i])
        axes[i].set_title(f"{state} Evolution Over Time")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel(state)

    for i, control in enumerate(control_names):
        sns.lineplot(
            data=control_data, x="time_step", y=control, ax=axes[len(state_names) + i]
        )
        axes[len(state_names) + i].set_title(f"{control} Evolution Over Time")
        axes[len(state_names) + i].set_xlabel("Time Step")
        axes[len(state_names) + i].set_ylabel(control)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    aero_model = pyf16.AerodynamicModel("./models/f16_model")
    aero_model.install("./models/f16_model/data")
    control_limits = aero_model.load_ctrl_limits()

    trim_target = pyf16.TrimTarget(15000, 500, None, None)
    trim_init = None
    trim_result = pyf16.trim(aero_model, trim_target, control_limits, trim_init)
    # trim_control
    # [2109.4128697158994, -2.244149780568322, -0.09357788625211498, 0.09446875516094778]

    init_state = pyf16.State(
        0,
        0,
        15000,
        np.deg2rad(40),
        np.deg2rad(30),
        0,
        500,
        np.deg2rad(20),
        0,
        np.deg2rad(40),
        np.deg2rad(40),
        np.deg2rad(40),
    )

    # init_state = trim_result.state
    init_control = trim_result.control

    states = [np.array(init_state.to_list())]
    controls = [np.array(init_control.to_list())]

    f16 = pyf16.PlaneBlock(
        pyf16.SolverType.RK4,
        0.01,
        aero_model,
        pyf16.CoreInit(init_state, init_control),
        [0, 0, 0],
        control_limits,
    )

    linear_model = F16LinearModel(
        f16,
        np.array(init_state.to_list()),
        np.array(init_control.to_list()),
    )
    discrete_linear_model = DiscreteF16LinearModel(linear_model)

    # npos epos altitude phi theta psi velocity alpha beta p q r
    Q = np.diag([0, 0, 0, 1, 1, 0, 0, 0, 0, 0.1, 0.1, 0.1])
    # throttle elevator aileron rudder
    R = np.diag([0, 0, 0, 0])
    P = np.diag([0, 0, 0, 1, 1, 0, 0, 0, 0, 0.1, 0.1, 0.1])
    C = np.eye(12)
    u_min = np.array([2100, -10, -10, -20])
    u_max = np.array([2200, 10, 10, 20])
    y_min = None
    y_max = None

    solver_option = {
        "learning_rate": 1e-2,
        "max_iters": 1000,
        "tol": 1e-3,
        "log_interval": 1000,
    }

    histories = {"pgd": [], "mgd": []}
    gradient_histories = {"pgd": [], "mgd": []}

    solver = ProjectionGradientDescentSolver(**solver_option)

    mpc = MPC(
        discrete_linear_model,
        Q,
        R,
        P,
        C,
        solver,
        u_min,
        u_max,
        y_min,
        y_max,
        prediction_horizon=30,
    )

    for i in range(400):
        if i % 50 == 0 and i != 0:
            linear_model = F16LinearModel(
                f16,
                states[-1],
                controls[-1],
            )
            discrete_linear_model = DiscreteF16LinearModel(linear_model)
            mpc.update_model(discrete_linear_model)

        u0 = np.stack([controls[-1]] * mpc.prediction_horizon)
        control, history, gradient_history = mpc.solve(states[-1], u0)
        controls.append(control)

        core_output = f16.update(
            pyf16.Control.from_list(controls[-1].tolist()), 0.01 * i
        )
        states.append(np.array(core_output.state.to_list()))

        histories["pgd"].append(history[-1])
        gradient_histories["pgd"].append(gradient_history[-1])

    f16.delete_model()

    f16 = pyf16.PlaneBlock(
        pyf16.SolverType.RK4,
        0.01,
        aero_model,
        pyf16.CoreInit(init_state, init_control),
        [0, 0, 0],
        control_limits,
    )
    linear_model = F16LinearModel(
        f16,
        np.array(init_state.to_list()),
        np.array(init_control.to_list()),
    )
    discrete_linear_model = DiscreteF16LinearModel(linear_model)
    solver = MomentumGradientDescentSolver(**solver_option)

    mpc = MPC(
        discrete_linear_model,
        Q,
        R,
        P,
        C,
        solver,
        u_min,
        u_max,
        y_min,
        y_max,
        prediction_horizon=30,
    )

    for i in range(1000):
        if i % 50 == 0 and i != 0:
            linear_model = F16LinearModel(
                f16,
                states[-1],
                controls[-1],
            )
            discrete_linear_model = DiscreteF16LinearModel(linear_model)
            mpc.update_model(discrete_linear_model)

        u0 = np.stack([controls[-1]] * mpc.prediction_horizon)
        control, history, gradient_history = mpc.solve(states[-1], u0)
        controls.append(control)

        core_output = f16.update(
            pyf16.Control.from_list(controls[-1].tolist()), 0.01 * i
        )
        states.append(np.array(core_output.state.to_list()))

        histories["mgd"].append(history[-1])
        gradient_histories["mgd"].append(gradient_history[-1])

    f16.delete_model()
    aero_model.uninstall()

    # Plot optimization objective values and gradient values for both solvers
    def plot_optimization_histories(histories, gradient_histories, solver_names):
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        for solver_name in solver_names:
            objective_values = histories[solver_name.lower()]
            sns.lineplot(
                x=range(len(objective_values)),
                y=objective_values,
                ax=axes[0],
                label=solver_name,
            )

        axes[0].set_title("Optimization Objective Values")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Objective Value")
        axes[0].legend()

        for solver_name in solver_names:
            gradient_norms = gradient_histories[solver_name.lower()]
            sns.lineplot(
                x=range(len(gradient_norms)),
                y=gradient_norms,
                ax=axes[1],
                label=solver_name,
            )

        axes[1].set_title("Gradient Norms")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Gradient Norm")
        axes[1].legend()

        fig.tight_layout()
        plt.show()

    # Collect histories and gradient histories for both solvers
    all_histories = histories
    all_gradient_histories = gradient_histories
    solver_names = ["pgd", "mgd"]

    plot_optimization_histories(all_histories, all_gradient_histories, solver_names)
