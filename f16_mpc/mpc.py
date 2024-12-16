from typing import Optional
from functools import partial

import numpy as np

from f16_mpc.linear import DiscreteF16LinearModel
from f16_mpc.solver import GradientDescentSolverBase


class MPC:
    def __init__(
        self,
        model: DiscreteF16LinearModel,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        C: np.ndarray,
        solver: GradientDescentSolverBase,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        y_min: Optional[np.ndarray] = None,
        y_max: Optional[np.ndarray] = None,
        prediction_horizon: int = 10,
    ) -> None:
        self.model = model
        self.Q = Q
        self.R = R
        self.P = P
        self.C = C
        self.u_min = u_min
        self.u_max = u_max
        self.y_min = y_min
        self.y_max = y_max
        self._prediction_horizon = prediction_horizon

        self.constraints = partial(self._build_constraints)
        self.objective_func = partial(self._build_objective_func)

        self.solver = solver

    def update_model(self, model: DiscreteF16LinearModel) -> None:
        self.model = model
        self.constraints = partial(self._build_constraints)
        self.objective_func = partial(self._build_objective_func)

    def _build_constraints(self, x0: np.ndarray) -> callable:
        # T_bar = [[CA], [CA^2], ..., [CA^N]]
        T_bar = np.block(
            [
                [self.C @ np.linalg.matrix_power(self.model.A, i)]
                for i in range(1, self.prediction_horizon + 1)
            ]
        )

        # E_bar = [[CE], [CAE], ..., [CA^(N-1)E]]
        # E.shape = (12,)
        # E_bar.shape = (12 * N ,)
        E_bar = np.block(
            [
                [self.C @ np.linalg.matrix_power(self.model.A, i - 1) @ self.model.E]
                for i in range(1, self.prediction_horizon + 1)
            ]
        )
        E_bar = E_bar.flatten()

        # S_bar = [[CB, 0, ..., 0], [CAB, CB, 0, ..., 0], ..., [CA^(N-1)B, CA^(N-2)B, ..., CB]]
        S_bar_blocks = []
        for i in range(self.prediction_horizon):
            row_blocks = []
            for j in range(self.prediction_horizon):
                if i >= j:
                    row_blocks.append(
                        self.C
                        @ np.linalg.matrix_power(self.model.A, i - j)
                        @ self.model.B
                    )
                else:
                    row_blocks.append(np.zeros_like(self.model.B))
            S_bar_blocks.append(row_blocks)
        S_bar = np.block(S_bar_blocks)

        def func(u: np.ndarray) -> np.ndarray:
            if self.y_max is not None and self.y_min is not None:
                y = S_bar @ u + E_bar + T_bar @ x0
                y = y.reshape(self.prediction_horizon, -1)
                y = y.clip(self.y_min, self.y_max)
                y = y.flatten()
                u = np.linalg.pinv(S_bar) @ (y - E_bar - T_bar @ x0)
            u = u.reshape(self.prediction_horizon, -1)
            if self.u_max is not None and self.u_min is not None:
                u = u.clip(self.u_min, self.u_max)
            u = u.flatten()
            return u

        return func

    @staticmethod
    def _block_diag(*diag_blocks) -> np.array:
        blocks = []
        for i in range(len(diag_blocks)):
            row_blocks = []
            for j in range(len(diag_blocks)):
                if i == j:
                    row_blocks.append(diag_blocks[i])
                else:
                    row_blocks.append(np.zeros_like(diag_blocks[i]))
            blocks.append(row_blocks)
        return np.block(blocks)

    def _build_objective_func(self, x0: np.ndarray) -> callable:
        # x_{t+1} = f(x_0, u_0) + A(x_0, u_0)(x_t - x_0) + B(x_0, u_0)(u_t - u_0)

        Q_bar_blocks = [self.Q] * (self.prediction_horizon - 1) + [self.P]
        Q_bar = self._block_diag(*Q_bar_blocks)

        R_bar_blocks = [self.R] * self.prediction_horizon
        R_bar = self._block_diag(*R_bar_blocks)

        # T_bar = [[A], [A^2], ..., [A^N]]
        T_bar = np.block(
            [
                [np.linalg.matrix_power(self.model.A, i)]
                for i in range(1, self.prediction_horizon + 1)
            ]
        )

        # E_bar = [[E], [AE], ..., [A^(N-1)E]]
        # E.shape = (12,)
        # E_bar.shape = (12 * N ,)
        E_bar = np.block(
            [
                [np.linalg.matrix_power(self.model.A, i - 1) @ self.model.E]
                for i in range(1, self.prediction_horizon + 1)
            ]
        )
        E_bar = E_bar.flatten()

        # S_bar = [[B, 0, ..., 0], [AB, B, 0, ..., 0], ..., [A^(N-1)B, A^(N-2)B, ..., B]]
        S_bar_blocks = []
        for i in range(self.prediction_horizon):
            row_blocks = []
            for j in range(self.prediction_horizon):
                if i >= j:
                    row_blocks.append(
                        np.linalg.matrix_power(self.model.A, i - j) @ self.model.B
                    )
                else:
                    row_blocks.append(np.zeros_like(self.model.B))
            S_bar_blocks.append(row_blocks)
        S_bar = np.block(S_bar_blocks)

        H = S_bar.T @ Q_bar @ S_bar + R_bar
        F = (E_bar.T + x0.T @ T_bar.T) @ Q_bar @ S_bar
        Y = (E_bar.T + x0.T @ T_bar.T) @ Q_bar @ (
            E_bar + T_bar @ x0
        ) + x0.T @ self.Q @ x0

        # Check if H is positive definite
        if not np.all(np.linalg.eigvals(H) >= -1e-20):
            raise ValueError(
                "The matrix H is not positive definite, the problem might not be convex."
            )

        def func(u: np.ndarray) -> float:
            cost = u.T @ H @ u + F @ u + u.T @ F.T + Y
            grad_u = 2 * H @ u + 2 * F
            return cost, grad_u

        return func

    def solve(
        self, x0: np.ndarray, u0: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solves the MPC optimization problem.

        Parameters:
        x0 (np.ndarray): The initial state vector.
        x_ref (np.ndarray): The reference state vector.
        u0 (Optional[np.ndarray]): The initial guess for the control inputs, shape (prediction_horizon, control_shape).

        Returns:
        np.ndarray: The optimal control input for the current time step.
        """

        if u0 is None:
            u0 = np.zeros((self.prediction_horizon, self.model.control_shape))
        u0 = u0.flatten()

        u, history, gradient_history = self.solver.solve(
            self.objective_func(x0),
            self.constraints(x0),
            u0,
        )

        u = u.reshape(self.prediction_horizon, -1)
        return u[0], history, gradient_history

    @property
    def prediction_horizon(self) -> int:
        return self._prediction_horizon
