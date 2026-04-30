import math

import numpy as np
import re
from pathlib import Path
from typing import Optional


def generate_data(
        n: int,
        m: int,
        sigma: float,
        seed: int = 6365,
        verbose: bool = False,
        cache_dir: Optional[str | Path] = None,
        ignore_cache: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a random instance of data D = {a1, ..., am ∈ R^n, b1, ..., bm ∈ R}.
        Data is formatted as A ∈ R^{m x n} and b ∈ R^m (numpy arrays).
        sigma > 0 is a parameter controlling ill-conditioning.
        When A has full column rank (for example, when m >= n), this construction makes
        cond(A^T A) = λ_max(B) / λ_min(B) approximately equal to sigma, where B := A^T A.
    
    Args:
        n: number of columns of A
        m: number of rows of A
        sigma: parameter of ill-conditioning of the data
        seed: random seed for reproducibility
        verbose: whether to print additional information about the generated data
    
    Returns:
        A (np.array): generated matrix A
        b (np.array): generated vector b
    """
    # determine cache directory
    if cache_dir is None:
        # find repo root by locating .git or stop at filesystem root
        current = Path(__file__).resolve().parent
        repo_root = current
        while repo_root != repo_root.parent and not (repo_root / '.git').exists():
            repo_root = repo_root.parent
        if not (repo_root / '.git').exists():
            repo_root = current
        cache_dir = repo_root / '.data_cache'
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # safe key for filename
    sigma_safe = re.sub(r'[^0-9a-zA-Z_]', '_', str(sigma))
    key = f"n{n}_m{m}_sigma{sigma_safe}_seed{seed}"
    cache_path = Path(cache_dir) / f"{key}.npz"

    if cache_path.exists() and not ignore_cache:
        if verbose:
            print(f"Loading cached data from {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            A = data['A']
            b = data['b']
        return A, b

    if verbose:
        print(f"Setting random (numpy) seed to {seed}.")
    np.random.seed(seed)

    rank = min(n, m)
    singular_values = np.linspace(1, 1/math.sqrt(sigma), n)[:rank]
    A_random = np.random.randn(m, rank)
    U, _ = np.linalg.qr(A_random, mode='reduced')
    B_random = np.random.randn(n, n)
    V, _ = np.linalg.qr(B_random)
    A = (U * singular_values) @ V[:, :rank].T

    b = np.random.randn(m)

    # save to cache (including metadata)
    try:
        np.savez_compressed(cache_path, A=A, b=b, n=n, m=m, sigma=sigma, seed=seed)
        if verbose:
            print(f"Saved generated data to cache: {cache_path}")
    except Exception:
        if verbose:
            print(f"Warning: failed to write cache to {cache_path}")

    return A, b


if __name__ == "__main__":
    n = 3
    m = 10
    sigma = 1e2
    A, b = generate_data(n, m, sigma)
    print("A:", A.shape, A)
    print("b:", b.shape, b)

    # Suggested validation test: for full-column-rank A, cond(A^T A) should match sigma.
    B = A.T @ A
    eigvals = np.linalg.eigvalsh(B)
    cond_B = eigvals.max() / eigvals.min()
    print("cond(A^T A):", cond_B)
    assert np.isclose(cond_B, sigma, rtol=1e-6, atol=1e-6), (
        f"Expected cond(A^T A) ≈ {sigma}, got {cond_B}"
    )
