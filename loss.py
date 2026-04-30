import numpy as np


_EXACT_SPECTRAL_NORM_DIM = 2_000


def _spectral_norm_squared_upper_bound(A: np.ndarray) -> float:
    """Return ||A||_2^2 exactly when cheap, otherwise use ||A||_F^2."""
    min_dim = min(A.shape)
    if min_dim <= _EXACT_SPECTRAL_NORM_DIM:
        gram = A.T @ A if A.shape[0] >= A.shape[1] else A @ A.T
        return max(np.linalg.eigvalsh(gram).max().item(), 0.0)
    return np.linalg.norm(A, 'fro').item() ** 2


class QuadraticLoss:
    fwd_mat_vec = 1
    grad_mat_vec = 2

    @classmethod
    def lipschitz_estimate(cls, A, mu) -> float:
        # For the quadratic loss, the Lipschitz constant of the gradient is given by the largest eigenvalue of A^T A / m + mu * I
        # Equivalently, L = ||A||_2^2 / m + mu.
        # When computing ||A||_2 exactly is too expensive, ||A||_F^2 / m + mu is a safe upper bound.
        spectral_sq = _spectral_norm_squared_upper_bound(A)
        return spectral_sq / A.shape[0] + mu

    def __init__(self, A: np.ndarray, b: np.ndarray, mu: float = 0.):
        self.A = A
        self.b = b
        self.mu = mu

    def __call__(self, x: np.ndarray) -> float:
        gap = self.A @ x - self.b
        loss = 0.5 * np.dot(gap, gap) / self.A.shape[0]
        if self.mu > 0:
            loss += 0.5 * self.mu * np.dot(x, x)
        return loss.item()
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        grad = self.A.T @ (self.A @ x - self.b) / self.A.shape[0]
        if self.mu > 0:
            grad += self.mu * x
        return grad


class LogisticLoss:
    fwd_mat_vec = 1
    grad_mat_vec = 2
    
    @classmethod
    def lipschitz_estimate(cls, A, mu) -> float:
        # For the logistic loss, the Lipschitz constant of the gradient is given by the largest eigenvalue of A^T A / 4m + mu * I
        # Equivalently, L <= ||A||_2^2 / (4m) + mu.
        # When computing ||A||_2 exactly is too expensive, ||A||_F^2 / (4m) + mu is a safe upper bound.
        # The factor of 4 comes from the fact that l''(t)=σ(t)(1−σ(t)) ≤ 1/4, where σ(t) := sigmoid(t).
        spectral_sq = _spectral_norm_squared_upper_bound(A)
        return spectral_sq / (4 * A.shape[0]) + mu

    def __init__(self, A: np.ndarray, b: np.ndarray, mu: float = 0.):
        self.A = A
        self.b = b
        self.mu = mu

    def __call__(self, x: np.ndarray) -> float:
        gap = self.A @ x - self.b
        loss = np.mean(np.log1p(np.exp(gap)))
        if self.mu > 0:
            loss += 0.5 * self.mu * np.dot(x, x)
        return loss.item()

    def grad(self, x: np.ndarray) -> np.ndarray:
        gap = self.A @ x - self.b
        exp_gap = np.exp(gap)
        grad = self.A.T @ (exp_gap / (1 + exp_gap)) / self.A.shape[0]
        if self.mu > 0:
            grad += self.mu * x
        return grad


class L1Loss:
    fwd_mat_vec = 1
    grad_mat_vec = 2

    def __init__(self, A: np.ndarray, b: np.ndarray, mu: float = 0.):
        self.A = A
        self.b = b
        self.mu = mu
        self.grad_mat_vec = 2

    def __call__(self, x: np.ndarray) -> float:
        gap = self.A @ x - self.b
        loss = np.mean(np.abs(gap))
        if self.mu > 0:
            loss += 0.5 * self.mu * np.dot(x, x)
        return loss.item()

    def grad(self, x: np.ndarray) -> np.ndarray:
        gap = self.A @ x - self.b
        grad = self.A.T @ np.sign(gap) / self.A.shape[0]
        if self.mu > 0:
            grad += self.mu * x
        return grad
