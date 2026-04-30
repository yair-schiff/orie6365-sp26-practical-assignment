import itertools
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from data import generate_data
from loss import QuadraticLoss, LogisticLoss
import grad_methods


LossName = Literal["quadratic", "logistic"]

LOSSES = {
    'quadratic': QuadraticLoss,
    'logistic': LogisticLoss,
}
LINE_STYLES = {
    grad_methods.gradient_method.__name__: "solid",
    grad_methods.fast_gradient_method.__name__: "dashed"
}


def plot_runs(
    ns: list[int],
    ms: list[int],
    sigmas: list[int | float],
    mus: list[float],
    losses: list[LossName],
    adaptive: bool,
    n_iters: int,
    total_runs: int,
    show_grad_method_pbar: bool = True,
) -> None:
    """Runs experiments and plots results.
    
    Args:
        ns: list of n (number of columns of A) values to run experiments for
        ms: list of m (number of rows of A) values to run experiments for
        sigmas: list of sigma (ill-conditioning parameter) values to run experiments for
        mus: list of mu (strong convexity parameter) values to run experiments for
        losses: list of loss function names to run experiments for (must be keys of LOSSES)
        adaptive: whether to use adaptive search for estimating the Lipschitz constant at each iteration of the optimization methods
        n_iters: number of iterations to run each optimization method for
        total_runs: total number of runs to be executed (for progress bar)
        show_grad_method_pbar: whether to show a progress bar for each optimization method (nested within the overall progress bar)
    """
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    with tqdm(total=total_runs, desc="Running experiments", leave=False) as pbar:
        for n in ns:
            for m in ms:
                if n > m:
                    continue
                plt.figure(figsize=(18, 12), tight_layout=False)  # disable tight_layout so we can control top margin
                plt.suptitle(f"Experiment Settings $n$={n}, $m$={m}\nAdaptive = {adaptive}", fontsize=16)
                for sigma in sigmas:
                    for mu in mus:
                        next_color = next(colors)
                        for col_idx, loss in enumerate(losses):
                            x0 = np.random.randn(n)
                            A, b = generate_data(m=m, n=n, sigma=sigma, seed=6365)
                            true_min = grad_methods.true_optimal_value(A=A, b=b, mu=mu, x0=x0, loss=loss)
                            run_name = f"$\\mu=${mu}, $\\sigma=${sigma}"
                            L0 = LOSSES[loss].lipschitz_estimate(A, mu)
                            # print(f"Running {run_name} with L0={L0:.2e} (Lipschitz estimate for {loss} loss)")
                            for grad_method in [grad_methods.gradient_method, grad_methods.fast_gradient_method]:
                                pbar.update(1)
                                pbar.set_postfix_str(f"{grad_method.__name__} on {m}x{n} matrix with sigma={sigma} and mu={mu} ({loss} loss)")
                                _, history, _ = grad_method(
                                    A=A, b=b, loss=loss, mu=mu, x_0=x0, n_iters=n_iters,
                                    L0=L0, adaptive=adaptive, show_progress=show_grad_method_pbar
                                )
                                func_res = np.array(history["func"]) - true_min

                                plt_kwargs = {"label": run_name, "color": next_color, "linestyle": LINE_STYLES[grad_method.__name__]}
                                # Plots vs. iteration
                                plt.subplot(3, 2*len(losses), 1 + 2*col_idx)
                                plt.plot(range(func_res.shape[0]), func_res, **plt_kwargs)
                                plt.xlabel("Iteration")
                                plt.ylabel("Functional Residual")
                                plt.title(f"{loss.title()} Loss", fontsize=14)
                                plt.subplot(3, 2*len(losses), 2 + 2*col_idx)
                                plt.plot(range(func_res.shape[0]), history["grad"], **plt_kwargs)
                                plt.xlabel("Iteration")
                                plt.yscale("log")
                                plt.ylabel("Gradient Norm")
                                plt.title(f"{loss.title()} Loss", fontsize=14)

                                # Plots vs. time
                                plt.subplot(3, 2*len(losses), len(losses)*2 + 2*col_idx + 1)
                                plt.plot(history["time"], func_res, **plt_kwargs)
                                plt.xlabel("Time (s)")
                                plt.ylabel("Functional Residual")
                                plt.subplot(3, 2*len(losses), len(losses)*2 + 2*col_idx + 2)
                                plt.plot(history["time"], history["grad"], **plt_kwargs)
                                plt.xlabel("Time (s)")
                                plt.yscale("log")
                                plt.ylabel("Gradient Norm")

                                # Plots vs. matrix-vector products
                                plt.subplot(3, 2*len(losses), len(losses)*4 + 2*col_idx + 1)
                                plt.plot(history["mat_vec"], func_res, **plt_kwargs)
                                plt.xlabel("Matrix-Vector Products")
                                plt.ylabel("Functional Residual")
                                plt.subplot(3, 2*len(losses), len(losses)*4 + 2*col_idx + 2)
                                plt.plot(history["mat_vec"], history["grad"], **plt_kwargs)
                                plt.xlabel("Matrix-Vector Products")
                                plt.yscale("log")
                                plt.ylabel("Gradient Norm")

                                # remove per-axis legends created earlier
                                fig = plt.gcf()
                                for ax in fig.axes:
                                    leg = ax.get_legend()
                                    if leg is not None:
                                        leg.remove()

                                # collect a single color for each run_name (first occurrence)
                                res_colors = {}
                                for ax in fig.axes:
                                    for line in ax.get_lines():
                                        lbl = line.get_label()
                                        if isinstance(lbl, str) and "mu" in lbl and lbl not in res_colors:
                                            res_colors[lbl] = line.get_color()

                                # make room at the top so legends aren't cut off
                                fig.subplots_adjust(top=0.78)

                                # legend 1: line style = optimization method
                                method_handles = [
                                    Line2D([0], [0], color="black", linestyle=LINE_STYLES[name], label=name.replace("_", " ").replace("method", "").title())
                                    for name in LINE_STYLES
                                ]
                                legend_methods = fig.legend(
                                    handles=method_handles,
                                    # title="Method (line style)",
                                    loc="upper center",
                                    bbox_to_anchor=(0.16, 1.0
                                                    ),
                                    bbox_transform=fig.transFigure,
                                    ncol=len(method_handles),
                                    fontsize=12,
                                )
                                fig.add_artist(legend_methods)

                                # legend 2: color = experiment setting
                                color_handles = [
                                    Line2D([0], [0], color=color, linewidth=2, linestyle="solid", label=lbl)
                                    for lbl, color in res_colors.items()
                                ]
                                fig.legend(
                                    handles=color_handles,
                                    # title=r"$\mu$ (color)",
                                    loc="upper center",
                                    bbox_to_anchor=(0.8, 1.0),
                                    bbox_transform=fig.transFigure,
                                    ncol=len(mus),
                                    fontsize=12,
                                )
                plt.show()
                plt.close()


if __name__ == "__main__":
    ns = [10]
    ms = [100]
    sigmas = [1e2, 1e4]
    mus = [0., 1e-3]
    losses: list[LossName] = ["quadratic", "logistic"]
    total_runs = len(ns) * len(ms) * len(sigmas) * len(mus) * len(losses) * 2  # 2 optimization methods
    adaptive = False
    n_iters = 1000
    plot_runs(ns=ns, ms=ms, sigmas=sigmas, mus=mus, losses=losses, total_runs=total_runs, adaptive=adaptive, n_iters=n_iters)
