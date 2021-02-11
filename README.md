# Numerical experiments from the paper *Pricing high-dimensional Bermudan options with hierarchical tensor formats*

## Usage
Execute the script to replicate the corresponding numerical experiment.

|       | Primal | Dual |
|-------|--------|------|
|Table 1|`primal/putBasket/s0_100/script.py`|`dual/putBasket/main.py`|
|Table 2|`primal/putBasket/s0_110/script.py`|`dual/putBasket_100K/main.py`|
|Table 3||`dual/callMaximum/main.py`|
|Table 4|||
|Table 5|||

## Dependencies

- numpy
- scipy
- xerus
- matplotlib
- rich