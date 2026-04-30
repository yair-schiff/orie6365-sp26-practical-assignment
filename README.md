# ORIE 6365 Practial Assignment

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
Open the notebook [`experiments.ipynb`](./experiments.ipynb) and exectue all cells.
This will run:
- gradient and fast-gradient experiments, without and with adaptive step sizes (Problems 4.1 and 4.2, respectively),
- sub-gradient experiments (Problem 4.3),
- Newton's method experiments (Problem 5.1). 

## Code overview

- The data generation function is in [`data.py`](./data.py).
- Losses (`quadratic`, `logistic`, `l1`) are implemented in [`loss.py`](./loss.py)
- Gradient methods (vanilla, fast, sub-gradient) are impelemented in [`grad_methods.py`](./grad_methods.py)
- The plotting routine that sweeps over experiment configs, runs the methods, and plots results is implemented in [`plotting.py`](./plotting.py)


## TODOs:
- Implement sub-gradient method and add experiment run to notebook
- Implement Newton's method and add experiment run to notebook
