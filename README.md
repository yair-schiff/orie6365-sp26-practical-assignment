# ORIE 6365 Practical Assignment

## Environment setup

Create the conda environment from the provided [`environment.yml`](./environment.yml):

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate orie6365
```

## How to run
Open the notebook [`experiments_report.ipynb`](./experiments_report.ipynb) and execute all cells.
This will run:
- gradient and fast-gradient experiments, without and with adaptive step sizes;
- subgradient experiments for quadratic, logistic, and l1 losses;
- data-generation condition-number sanity checks;
- plots used in the final report.

## Code overview

- The data generation function is in [`data.py`](./data.py).
- Losses (`quadratic`, `logistic`, `l1`) are implemented in [`loss.py`](./loss.py)
- Gradient methods (vanilla, fast, subgradient) are implemented in [`grad_methods.py`](./grad_methods.py).
- Experiment sweeps and plotting code are in [`experiments_report.ipynb`](./experiments_report.ipynb).
