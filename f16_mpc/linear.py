import numpy as np
import pyf16


class LinearModel:
    def __init__(
        self,
        init_state: np.ndarray,
        init_control: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        solver: pyf16.SimpleSolver,
    ) -> None:
        self.init_state, self.init_control = init_state, init_control
        self.A, self.B = A, B
        self.solver = solver

    @property
    def state_matrix(self) -> np.ndarray:
        return self.A

    @property
    def control_matrix(self) -> np.ndarray:
        return self.B

    @property
    def x0(self) -> np.ndarray:
        return self.init_state

    @property
    def u0(self) -> np.ndarray:
        return self.init_control

    @property
    def state_shape(self) -> int:
        return self.A.shape[0]

    @property
    def control_shape(self) -> int:
        return self.B.shape[1]

    def update(self, state: np.ndarray, control: np.ndarray, t: float) -> np.ndarray:
        def dynamics(_t, state, control):
            dxdt = self.A @ state + self.B @ control
            # 转换为list
            dxdt = dxdt.tolist()
            return dxdt

        state = state.tolist()
        control = control.tolist()
        state = self.solver.solve(dynamics, t, state, control)
        state = np.array(state)
        return state

    def __call__(self, state: np.ndarray, control: np.ndarray, t: float) -> np.ndarray:
        return self.update(state, control, t)


class F16LinearModel(LinearModel):
    def __init__(
        self,
        f16: pyf16.PlaneBlock,
        init_state: np.ndarray,
        init_control: np.ndarray,
        epsilon_A: float | np.ndarray = 1e-5,
        epsilon_B: float | np.ndarray = 1e-5,
        solver: pyf16.SimpleSolver = pyf16.SimpleSolver(pyf16.SolverType.RK4, 0.01),
    ) -> None:
        self.f16 = f16
        self.init_state_dot = self._compute_init_state_dot(init_state, init_control)
        A, B = self._compute_jacobians(init_state, init_control, epsilon_A, epsilon_B)
        super().__init__(init_state, init_control, A, B, solver)

    def _state_equation(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        state = pyf16.State.from_list(state.tolist())
        control = pyf16.Control.from_list(control.tolist())
        core_init = pyf16.CoreInit(state, control)
        self.f16.reset(core_init)
        core_output = self.f16.update(control, 0.01)
        state_dot = self.f16.state_dot
        return np.array(state_dot.to_list())

    def _compute_jacobian(
        self, func: callable, x: np.ndarray, epsilon: float | np.ndarray
    ) -> np.ndarray:
        fx = func(x)
        m = len(fx)
        n = len(x)
        jacobian_matrix = np.zeros((m, n))
        if type(epsilon) == float:
            epsilon = np.array([epsilon] * n)
        for i in range(n):
            x_eps = np.array(x, dtype=float)
            x_eps[i] += epsilon[i]
            fx_eps = func(x_eps)
            jacobian_matrix[:, i] = (fx_eps - fx) / epsilon[i]
        return jacobian_matrix

    def _compute_jacobians(
        self,
        init_state: np.ndarray,
        init_control: np.ndarray,
        epsilon_A: float | np.ndarray,
        epsilon_B: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        jacobian_state = self._compute_jacobian(
            lambda s: self._state_equation(s, init_control),
            init_state,
            epsilon_A,
        )
        jacobian_control = self._compute_jacobian(
            lambda c: self._state_equation(init_state, c),
            init_control,
            epsilon_B,
        )
        return jacobian_state, jacobian_control

    def _compute_init_state_dot(
        self, init_state: np.ndarray, init_control: np.ndarray
    ) -> np.ndarray:
        init_state_dot = self._state_equation(init_state, init_control)
        init_state_dot = np.array(init_state_dot.tolist())
        return init_state_dot

    def update(self, state: np.ndarray, control: np.ndarray, t: float) -> np.ndarray:
        def dynamics(_t, state, control):
            state = np.array(state)
            control = np.array(control)
            dxdt = (
                self.A @ (state - self.init_state)
                + self.B @ (control - self.init_control)
                + self.init_state_dot
            )
            dxdt = dxdt.tolist()
            return dxdt

        state = state.tolist()
        control = control.tolist()
        state = self.solver.solve(dynamics, t, state, control)
        state = np.array(state)
        return state


class DiscreteF16LinearModel:
    def __init__(self, model: F16LinearModel):
        self.init_state, self.init_control = model.init_state, model.init_control
        self.A = model.A * model.solver.delta_t + np.eye(model.state_shape)
        self.B = model.B * model.solver.delta_t
        self._bias = (
            model.init_state_dot * model.solver.delta_t
            - model.A @ model.init_state * model.solver.delta_t
            - model.B @ model.init_control * model.solver.delta_t
        )

    def update(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return self.A @ state + self.B @ control + self._bias

    def __call__(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return self.update(state, control)

    @property
    def state_matrix(self) -> np.ndarray:
        return self.A

    @property
    def control_matrix(self) -> np.ndarray:
        return self.B

    @property
    def x0(self) -> np.ndarray:
        return self.init_state

    @property
    def u0(self) -> np.ndarray:
        return self.init_control

    @property
    def state_shape(self) -> int:
        return self.A.shape[0]

    @property
    def control_shape(self) -> int:
        return self.B.shape[1]

    @property
    def E(self) -> np.ndarray:
        return self._bias
