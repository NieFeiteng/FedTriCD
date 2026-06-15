# Highly Efficient Vertical Federated Clustering with Triangle-Inequality-based Pruning

This repository contains the MATLAB code for the paper **"Highly Efficient Vertical Federated Clustering with Triangle-Inequality-based Pruning"**.

The codebase includes implementations and experiment utilities for:

- `CDKM`
- `LLoyd`
- `TriCD`
- `FedTriCD`

## Usage

Set `CLUSTER_DATA_PATH` to the directory containing dataset `.csv` or sparse `.txt` files. If it is not set, `main_experiment` uses the local `data` directory.

Run the experiments from MATLAB:

```matlab
main_experiment
```

Results are written to `results/`.
