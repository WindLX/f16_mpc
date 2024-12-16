import numpy as np
import pyf16
import logging
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
    "throttle",
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
    # state_data[angle_columns] = state_data[angle_columns].apply(np.rad2deg)

    state_data_melted = state_data.melt(
        id_vars="time_step", var_name="state", value_name="value"
    )
    control_data_melted = control_data.melt(
        id_vars="time_step", var_name="control", value_name="value"
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for i, state in enumerate(state_names):
        sns.lineplot(data=state_data, x="time_step", y=state, ax=axes[i])
        axes[i].set_title(f"{state} Evolution Over Time")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("State Value")

    for i, control in enumerate(control_names):
        sns.lineplot(
            data=control_data, x="time_step", y=control, ax=axes[len(state_names) + i]
        )
        axes[len(state_names) + i].set_title(f"{control} Evolution Over Time")
        axes[len(state_names) + i].set_xlabel("Time Step")
        axes[len(state_names) + i].set_ylabel("Control Value")

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
        np.deg2rad(10),
        np.deg2rad(10),
        np.deg2rad(10),
        500,
        np.deg2rad(3),
        0,
        np.deg2rad(5),
        np.deg2rad(-5),
        np.deg2rad(5),
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
    u_min = np.array([2100, -5, -10, -20])
    u_max = np.array([2200, 5, 10, 20])
    y_min = None
    y_max = None

    solver_option = {
        "learning_rate": 1e-2,
        "max_iters": 1000,
        "tol": 1e-3,
        "log_interval": 1000,
    }
    # solver = ProjectionGradientDescentSolver(**solver_option)
    solver = MomentumGradientDescentSolver(**solver_option)
    # solver = AdamSolver(**solver_option)
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
        control = mpc.solve(states[-1], u0)
        controls.append(control)

        core_output = f16.update(
            pyf16.Control.from_list(controls[-1].tolist()), 0.01 * i
        )
        states.append(np.array(core_output.state.to_list()))

    f16.delete_model()
    aero_model.uninstall()

    plot(states, controls)
