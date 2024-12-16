import numpy as np
import pyf16
import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from f16_mpc.linear import F16LinearModel, DiscreteF16LinearModel

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    aero_model = pyf16.AerodynamicModel("./models/f16_model")
    aero_model.install("./models/f16_model/data")
    control_limits = aero_model.load_ctrl_limits()

    trim_target = pyf16.TrimTarget(15000, 500, None, None)
    trim_init = None
    trim_result = pyf16.trim(aero_model, trim_target, control_limits, trim_init)

    print(trim_result.state.to_list())
    print(trim_result.control.to_list())
    print(trim_result.state_extend.to_list())

    f16 = pyf16.PlaneBlock(
        pyf16.SolverType.RK4,
        0.01,
        aero_model,
        trim_result.to_core_init(),
        [0, 0, 0],
        control_limits,
    )

    linear_model = F16LinearModel(
        f16,
        np.array(trim_result.state.to_list()),
        np.array(trim_result.control.to_list()),
        epsilon_A=1e-5,
        epsilon_B=1e-5,
    )
    discrete_linear_model = DiscreteF16LinearModel(linear_model)

    states = [np.array(trim_result.state.to_list())]
    linear_states = [np.array(trim_result.state.to_list())]
    discrete_linear_states = [np.array(trim_result.state.to_list())]
    control = trim_result.control
    control = pyf16.Control(2109, -2, 0, 5)
    control_ndarray = np.array(control.to_list())
    for i in range(1000):
        core_output = f16.update(control, 0.01 * i)
        states.append(np.array(core_output.state.to_list()))

        if i % 50 == 0 and i != 0:
            linear_model = F16LinearModel(
                f16,
                np.array(states[-2]),
                control_ndarray,
                epsilon_A=1e-5,
                epsilon_B=1e-5,
            )
            discrete_linear_model = DiscreteF16LinearModel(linear_model)

        linear_state = linear_model(linear_states[-1], control_ndarray, 0.01)
        linear_states.append(linear_state)

        discrete_linear_state = discrete_linear_model(
            discrete_linear_states[-1], control_ndarray
        )
        discrete_linear_states.append(discrete_linear_state)

    f16.delete_model()
    aero_model.uninstall()

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

    df_states = pd.DataFrame(states, columns=state_names)
    df_linear_states = pd.DataFrame(linear_states, columns=state_names)
    df_discrete_linear_states = pd.DataFrame(
        discrete_linear_states, columns=state_names
    )

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, state in enumerate(state_names):
        sns.lineplot(data=df_states[state], label="Original Model", ax=axes[i])
        sns.lineplot(data=df_linear_states[state], label="Linear Model", ax=axes[i])
        sns.lineplot(
            data=df_discrete_linear_states[state],
            label="Discrete Linear Model",
            ax=axes[i],
        )
        axes[i].set_title(f"State: {state}")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel(state)
        axes[i].legend()

    plt.tight_layout()
    plt.show()
