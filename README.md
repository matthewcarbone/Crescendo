# crescendo
[![codecov](https://codecov.io/gh/x94carbone/crescendo/branch/master/graph/badge.svg?token=0M8IGBBWXQ)](https://codecov.io/gh/x94carbone/crescendo)
![Python package](https://github.com/x94carbone/crescendo/workflows/Python%20package/badge.svg?branch=master)

A greatly abstracted artificial intelligence and machine learning suite built for ease of use and broad-spectrum application.

## Installation
We recommend installing via `conda`:
```bash
conda env create -f environment.yml  # creates `crescendo` virtual environment
```

After making changes to the required packages, export the changes via
```bash
conda env export | grep -v "prefix" > environment.yml
```

### Manual developer installation
These represent the packages (and the order) in which they were installed in the tests and should almost definitely work on a fresh `conda` environment using `python 3.7.7`.
```bash
conda create -n crescendo python=3.7.7
conda activate crescendo

conda install pytorch torchvision -c pytorch
# or CUDA binaries
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

conda install -c dglteam dgl
# or CUDA binaries
# see here: https://docs.dgl.ai/en/0.4.x/install/index.html#install-from-conda

conda install -c conda-forge jupyterlab
conda install -c anaconda networkx
conda install -c conda-forge matplotlib
conda install -c anaconda pandas
conda install -c anaconda pytest
conda install -c conda-forge glob2
conda install -c rdkit rdkit
```

## Logging protocol
This package uses the `loggin` module to provide the user with information about what is happening during the program execution. The default logging stream pipes to the terminal and is usually abbreviated as `dlog` in the code. The following guidelines should be obeyed when logging.
* Inform the user of critical terminating issues via the `critical` level. These should be used right before raising an error.
* Inform the user of a potentially critical issue via the `error` level. These are intended to be similar to critical except it is not the case that the program is immediately terminated. Likely, there will be another error thrown later on.
* Inform the user of possible issues with their use of the program with the `warning` level.
* Log everything that the user should know with `dlog.info("...")`.

## Testing
To run the unit tests, we generally use
```bash
coverage run --source=crescendo test.py
```
and to view the report,
```bash
coverage report -m
```
