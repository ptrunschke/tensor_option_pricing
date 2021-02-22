# Numerical experiments from the paper *Pricing high-dimensional Bermudan options with hierarchical tensor formats*

## Usage
Execute the script to replicate the corresponding numerical experiment.

|       | Primal | Dual |
|-------|--------|------|
|Table 1|`primal/putBasket/s0_100/script.py` & `primal/putBasket/s0_110/script.py`|`dual/putBasket/main.py`|
|Table 2|`primal/putBasket/s0_100/script.py` & `primal/putBasket/s0_110/script.py`|`dual/putBasket_100K/main.py`|
|Table 3|`primal/callMaximum/unsorted/script.py`|`dual/callMaximum/main.py`|
|Table 4|`primal/callMaximum/unsorted/script.py`||
|Table 5|`primal/callMaximum/sorted/script.py`||

## Dependencies

- numpy
- scipy
- [xerus](https://libxerus.org/) (branch: `SALSA`)
- matplotlib
- rich

You can run the accompanying bash script to install all dependencies in a new `conda` environment.
```
bash install.sh <env_name>
```
To install the dependencies in an existing `conda` environment you can execute the following command in the activated environment.
```
conda install -c conda-froge -c ptrunschke numpy scipy xerus_conda matplotlib rich
```
