import math
import time
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from loss import QuadraticLoss, LogisticLoss, L1Loss


LOSSES = {
    'quadratic': QuadraticLoss,
    'logistic': LogisticLoss,
    'l1': L1Loss
}


def gradient_method(
    A: np.ndarray,
    b: np.ndarray,
    loss: Literal["quadratic", "logistic"],
    mu: float,
    x_0: np.ndarray,
    n_iters: int,
    L0: float,
    adaptive: bool = False,
    grad_norm_threshold: float = 1e-9,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, list], Literal["success", "max_iters_reached"]]:
    """Gradient method.

    Returns a tuple (x_sol, history, status).

    Args:
        A: data matrix
        b: data vector
        loss: loss function to minimize ("quadratic" or "logistic")
        mu: strong convexity parameter of the loss function
        x_0: initial point for the method
        n_iters: number of iterations to run the method for
        L0: initial guess for the smoothness parameter of the loss function
        adaptive: whether using an adaptive search for estimating the Lipschitz constant
            at each iteration of the method, or using the constant value L 0 for all
            iterations (constant step-size)
        show_progress: whether to display a progress bar for this run
    Returns:
        x_sol: the best observed iterate by function value
        history: a dictionary containing lists of values of interest at each iteration:
            - history['func'] is the list of function values among all main iterates {x_k}, k≥0
            - history['grad'] is the list of gradient norms among all main iterates {x_k}, k≥0
            - history['time'] is the list of time snapshots taken at the start of every iteration;
            - history['mat_vec'] is the list of total numbers of
                matrix-vector products Ah and A^top h used up to each iteration
                (cumulative statistics)
        status: status information about the run of the method (e.g., whether it converged, etc.).
    """
    mat_vec = 0
    x = x_0.copy()  # Ensure x_0 is not modified
    L = L0
    loss_fn = LOSSES[loss](A, b, mu)
    start = time.perf_counter()

    func = loss_fn(x)
    mat_vec += loss_fn.fwd_mat_vec
    grad = loss_fn.grad(x)
    mat_vec += loss_fn.grad_mat_vec
    grad_norm = np.linalg.norm(grad).item()
    history = {
        'func': [func],
        'grad': [grad_norm],
        'time': [0.],
        'mat_vec': [mat_vec],
    }
    best_x = x.copy()
    best_func = func
    status: Literal["success", "max_iters_reached"] = (
        "success" if grad_norm < grad_norm_threshold else "max_iters_reached"
    )
    with tqdm(total=n_iters, leave=False, disable=not show_progress, position=1) as pbar:
        for _ in range(n_iters):
            if status == "success":
                break
            if adaptive:
                grad_sq_norm = np.dot(grad, grad).item()
                while True:
                    step_size = 1 / L
                    x_new = x - step_size * grad
                    func_new = loss_fn(x_new)
                    mat_vec += loss_fn.fwd_mat_vec
                    if func - func_new >= (1 / (2 * L)) * grad_sq_norm:
                        x = x_new
                        func = func_new
                        L *= 0.5
                        break
                    L *= 2
            else:
                step_size = 1 / L
                x = x - step_size * grad
                func = loss_fn(x)
                mat_vec += loss_fn.fwd_mat_vec

            grad = loss_fn.grad(x)
            mat_vec += loss_fn.grad_mat_vec
            end = time.perf_counter()
            grad_norm = np.linalg.norm(grad).item()
            pbar.set_postfix({"func": func, "grad_norm": grad_norm, "L": L})

            # Update history
            history['func'].append(func)
            history['grad'].append(grad_norm)
            history['time'].append(end - start)
            history['mat_vec'].append(mat_vec)
            if func < best_func:
                best_func = func
                best_x = x.copy()
            pbar.update(1)
            if grad_norm < grad_norm_threshold:
                status = "success"
                pbar.total = pbar.n
                pbar.refresh()
                break

    return best_x, history, status


def fast_gradient_method(
    A: np.ndarray,
    b: np.ndarray,
    loss: Literal["quadratic", "logistic"],
    mu: float,
    x_0: np.ndarray,
    n_iters: int,
    L0: float,
    adaptive: bool = False,
    grad_norm_threshold: float = 1e-9,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, list], Literal["success", "max_iters_reached"]]:
    """Fast gradient method.
    
    Args:
        (same as `gradient_method`)
    Returns:
        (same as `gradient_method`)
    """
    mat_vec = 0
    x = x_0.copy()  # Ensure x_0 is not modified
    v = x_0.copy()
    Ak = 0.
    L = L0
    loss_fn = LOSSES[loss](A, b, mu)
    restart_period = None
    if mu > 0:
        restart_period = math.ceil(math.sqrt(8 * max(L, mu) / mu))
    steps_since_restart = 0
    start = time.perf_counter()

    func = loss_fn(x)
    mat_vec += loss_fn.fwd_mat_vec
    grad_x = loss_fn.grad(x)
    mat_vec += loss_fn.grad_mat_vec
    grad_norm = np.linalg.norm(grad_x).item()
    history = {
        'func': [func],
        'grad': [grad_norm],
        'time': [0.],
        'mat_vec': [mat_vec],
    }
    best_x = x.copy()
    best_func = func
    status: Literal["success", "max_iters_reached"] = (
        "success" if grad_norm < grad_norm_threshold else "max_iters_reached"
    )
    with tqdm(total=n_iters, leave=False, disable=not show_progress, position=1) as pbar:
        for _ in range(n_iters):
            if status == "success":
                break
            if restart_period is not None and steps_since_restart >= restart_period:
                Ak = 0.0
                v = x.copy()
                steps_since_restart = 0

            if adaptive:
                L_trial = L
                while True:
                    ak = (1 + math.sqrt(1 + 4 * Ak * L_trial)) / (2 * L_trial)
                    Ak_new = Ak + ak
                    gamma = ak / Ak_new
                    y = (1 - gamma) * x + gamma * v
                    grad_y = loss_fn.grad(y)
                    mat_vec += loss_fn.grad_mat_vec
                    v_new = v - ak * grad_y
                    x_new = gamma * v_new + (1 - gamma) * x
                    func_y = loss_fn(y)
                    mat_vec += loss_fn.fwd_mat_vec
                    func_new = loss_fn(x_new)
                    mat_vec += loss_fn.fwd_mat_vec
                    diff = x_new - y
                    upper_model = (
                        func_y
                        + np.dot(grad_y, diff)
                        + 0.5 * L_trial * np.dot(diff, diff)
                    )
                    if func_new <= upper_model:
                        v = v_new
                        x = x_new
                        Ak = Ak_new
                        func = func_new
                        L = L_trial * 0.5
                        if restart_period is not None:
                            restart_period = math.ceil(math.sqrt(8 * max(L, mu) / mu))
                        break
                    L_trial *= 2
            else:
                ak = (1 + math.sqrt(1 + 4 * Ak * L)) / (2 * L)
                Ak += ak
                gamma = ak / Ak
                y = (1 - gamma) * x + gamma * v
                grad_y = loss_fn.grad(y)
                mat_vec += loss_fn.grad_mat_vec
                v = v - ak * grad_y
                x = gamma * v + (1 - gamma) * x
                func = loss_fn(x)
                mat_vec += loss_fn.fwd_mat_vec

            steps_since_restart += 1
            grad_x = loss_fn.grad(x)
            mat_vec += loss_fn.grad_mat_vec
            end = time.perf_counter()
            grad_norm = np.linalg.norm(grad_x).item()
            pbar.set_postfix({"func": func, "grad_norm": grad_norm, "L": L})

            # Update history
            history['func'].append(func)
            history['grad'].append(grad_norm)
            history['time'].append(end - start)
            history['mat_vec'].append(mat_vec)
            if func < best_func:
                best_func = func
                best_x = x.copy()
            pbar.update(1)

            if grad_norm < grad_norm_threshold:
                status = "success"
                pbar.total = pbar.n
                pbar.refresh()
                break

    return best_x, history, status


def subgrad_norm_bound(A: np.ndarray, b: np.ndarray, loss: str, R: float) -> float:
    """Helper method to compute sub-grad norm bound to rescale constant step-size in un-normalized subgradient method."""
    m = A.shape[0]
    A_norm = np.linalg.norm(A, 2)

    if loss in {"logistic", "l1"}:
        return A_norm / np.sqrt(m)

    if loss == "quadratic":
        return A_norm * (A_norm * R + np.linalg.norm(b)) / m
    
    raise NotImplementedError(f"Subgradient norm bound not implemented for loss type {loss}.")


def subgradient_method(
    A: np.ndarray,
    b: np.ndarray,
    loss: Literal["quadratic", "logistic", "l1"],
    R: float,
    x_0: np.ndarray,
    n_iters: int,
    gamma: float,
    normalized: bool = False,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict[str, list], Literal["success", "max_iters_reached"]]:
    """Subgradient method.
    
    Args:
        A: data matrix
        b: data vector
        loss: loss function to minimize ("quadratic", "logistic", or "l1")
        R: radius of the Euclidean ball to project onto at each iteration
        x_0: initial point for the method
        n_iters: number of iterations to run the method for 
        gamma: constant step-size parameter
        normalized: whether to use normalized subgradients
        show_progress: whether to display a progress bar for this run
    Returns:
        x_sol: the best observed iterate by function value
        history: a dictionary containing lists of values of interest at each iteration:
            - history['func'] is the list of function values among all main iterates {x_k}, k≥0
            - history['grad'] is the list of subgradient norms among all main iterates {x_k}, k≥0
            - history['time'] is the list of time snapshots taken at the start of every iteration;
            - history['mat_vec'] is the list of total numbers of
                matrix-vector products Ah and A^top h used up to each iteration
                (cumulative statistics)
        status: status information about the run of the method (e.g., whether it converged, etc.).
    """
    mat_vec = 0
    x = x_0.copy()  # Ensure x_0 is not modified
    norm_x = np.linalg.norm(x).item()
    if norm_x > R:
        x = (R / norm_x) * x

    loss_fn = LOSSES[loss](A, b)
    start = time.perf_counter()

    func = loss_fn(x)
    mat_vec += loss_fn.fwd_mat_vec
    grad = loss_fn.grad(x)
    mat_vec += loss_fn.grad_mat_vec
    grad_norm = np.linalg.norm(grad).item()
    history = {
        'func': [func],
        'grad': [grad_norm],
        'time': [0.],
        'mat_vec': [mat_vec],
    }
    best_x = x.copy()
    best_func = func
    status: Literal["success", "max_iters_reached"] = "max_iters_reached"
    with tqdm(total=n_iters, leave=False, disable=not show_progress, position=1) as pbar:
        for _ in range(n_iters):
            if normalized and grad_norm > 0:
                direction = grad / grad_norm
            else:
                direction = grad
            x = x - gamma * direction
            # Projection onto Euclidean ball of radius R
            norm_x = np.linalg.norm(x).item()
            if norm_x > R:
                x = (R / norm_x) * x
            func = loss_fn(x)
            mat_vec += loss_fn.fwd_mat_vec
            
            grad = loss_fn.grad(x)
            mat_vec += loss_fn.grad_mat_vec
            end = time.perf_counter()
            grad_norm = np.linalg.norm(grad).item()
            pbar.set_postfix({"func": func, "grad_norm": grad_norm})

            # Update history
            history['func'].append(func)
            history['grad'].append(grad_norm)
            history['time'].append(end - start)
            history['mat_vec'].append(mat_vec)
            if func < best_func:
                best_func = func
                best_x = x.copy()
            pbar.update(1)

    return best_x, history, status
