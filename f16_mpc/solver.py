from typing import Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class GradientDescentSolverBase:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        max_iters: int = 1000,
        tol: float = 1e-5,
        log_interval: int = 1000,
    ) -> None:
        """
        初始化梯度下降求解器基类。

        Parameters:
        - learning_rate: 学习率。
        - max_iters: 最大迭代次数。
        - tol: 收敛阈值，梯度范数小于该值时停止。
        - log_interval: 每多少次迭代输出日志信息。
        """
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.log_interval = log_interval

    def solve(
        self,
        func: callable,
        constraints: Optional[callable],
        x0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("solve method not implemented")


class ProjectionGradientDescentSolver(GradientDescentSolverBase):
    def solve(
        self,
        func: callable,
        constraints: Optional[callable],
        x0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = x0
        y, grad = func(x)
        history = [y]

        for i in range(self.max_iters):
            if np.linalg.norm(grad) < self.tol:
                if self.log_interval > 0:
                    print(
                        f"Converged after {i} iterations, Objective Function Value: {history[-1]}"
                    )
                break

            if self.log_interval > 0 and i % self.log_interval == 0 and i != 0:
                print(f"Iteration {i}, Objective Function Value: {history[-1]}")

            x = x - self.learning_rate * grad

            if constraints:
                x = constraints(x)

            y, grad = func(x)
            history.append(y)

        history = np.array(history)

        if self.log_interval > 0:
            print(
                f"Final Objective Function Value: {history[-1]}, Gradient Norm: {np.linalg.norm(grad)}"
            )

        return x, history


class MomentumGradientDescentSolver(GradientDescentSolverBase):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        max_iters: int = 1000,
        tol: float = 1e-5,
        log_interval: int = 1000,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(learning_rate, max_iters, tol, log_interval)
        self.momentum = momentum

    def solve(
        self,
        func: callable,
        constraints: Optional[callable],
        x0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = x0
        y, grad = func(x)
        history = [y]

        v = np.zeros_like(x)

        for i in range(self.max_iters):
            if np.linalg.norm(grad) < self.tol:
                if self.log_interval > 0:
                    print(
                        f"Converged after {i} iterations, Objective Function Value: {history[-1]}"
                    )
                break

            if self.log_interval > 0 and i % self.log_interval == 0 and i != 0:
                print(f"Iteration {i}, Objective Function Value: {history[-1]}")

            v = self.momentum * v + self.learning_rate * grad
            x = x - v

            if constraints:
                x = constraints(x)

            y, grad = func(x)
            history.append(y)

        history = np.array(history)

        if self.log_interval > 0:
            print(
                f"Final Objective Function Value: {history[-1]}, Gradient Norm: {np.linalg.norm(grad)}"
            )

        return x, history


class AdamSolver(GradientDescentSolverBase):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        max_iters: int = 1000,
        tol: float = 1e-5,
        log_interval: int = 1000,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate, max_iters, tol, log_interval)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def solve(
        self,
        func: callable,
        constraints: Optional[callable],
        x0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = x0
        y, grad = func(x)
        history = [y]

        m = np.zeros_like(x)
        v = np.zeros_like(x)
        t = 0

        for i in range(self.max_iters):
            t += 1

            if np.linalg.norm(grad) < self.tol:
                if self.log_interval > 0:
                    print(
                        f"Converged after {i} iterations, Objective Function Value: {history[-1]}"
                    )
                break

            if self.log_interval > 0 and i % self.log_interval == 0 and i != 0:
                print(f"Iteration {i}, Objective Function Value: {history[-1]}")

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2

            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            x = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            if constraints:
                x = constraints(x)

            y, grad = func(x)
            history.append(y)

        history = np.array(history)

        if self.log_interval > 0:
            print(
                f"Final Objective Function Value: {history[-1]}, Gradient Norm: {np.linalg.norm(grad)}"
            )

        return x, history


if __name__ == "__main__":
    # 二次型
    def func(x: np.ndarray) -> float:
        Q = np.array([[2, 0.5], [0.5, 1]])
        b = np.array([1, 2])
        grad = Q @ x + b
        return 0.5 * x.T @ Q @ x + b.T @ x, grad

    def constraints(x: np.ndarray) -> np.ndarray:
        return np.clip(x, -10, 10)

    solver = ProjectionGradientDescentSolver()

    x0 = np.array([3, 4])
    x, history = solver.solve(func, constraints=constraints, x0=x0)

    print(x)
    print(history)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Convergence of Projection Gradient Descent")
    plt.show()
