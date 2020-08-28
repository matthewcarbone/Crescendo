# crescendo
[![codecov](https://codecov.io/gh/x94carbone/crescendo/branch/master/graph/badge.svg?token=0M8IGBBWXQ)](https://codecov.io/gh/x94carbone/crescendo)
![Python package](https://github.com/x94carbone/crescendo/workflows/Python%20package/badge.svg?branch=master)

An artificial intelligence and machine learning pipeline currently under development, used for facilitating broad-spectrum machine learning applications with a low barrier of entry and focused on building reusable code and a robust workflow. The current focus is the [QM9 Dataset](http://quantum-machine.org/datasets/), with potential future applications to come.

## Contributing

The current version of `crescendo` is intended as an pseudo-open source repository within the BNL community and trusted collaborators. We stress the following important points regarding contributions:
* Users are encouraged to use this pipeline as a medium for finding collaborators and collaborating with other scientists of scientific projects.
* Users are encouraged to contribute the code from their own projects to the repository, however...
* Users are *by no means* required to share privately modified code, even if it cloned directly from this repository. However, we would encourage everyone to share their code for the benefit of the community.
* While we have no way to track it, if users end up using a significant portion of the code base that was written by other contributors, we would encourage the user to acknowledge their contributions(s) in any published work.

### Pull requests
All contributor contributions must be added through pull request, as direct pushes to `master` are generally dangerous except when modifying simple non-essential files e.g. the `README.md`'s. We also note the following regarding design philosophy:
* Code should be well-tested before merging to master.
* Tests should [cover](https://stackoverflow.com/questions/195008/what-is-code-coverage-and-how-do-you-measure-it#:~:text=Code%20coverage%20is%20a%20measurement,tests%20against%20the%20instrumented%20product.) most/all of the new pushed code. 


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
These represent the packages (and the order) in which they were installed in the tests and should almost definitely work on a fresh `conda` environment using `python 3.7.7`. Note as well that if using the IC GPU's, they are only compatible with CUDA version `10.1`, and thus the `torch` and `dgl` CUDA-enabled packages must be installed accordingly with the correct version, or `torch` will not detect available GPUs at runtime.


```bash
conda create -n py377-torch16 python=3.7.7
conda activate py377-torch16

conda install pytorch torchvision -c pytorch
# or CUDA binaries
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

conda install -c dglteam dgl
# or CUDA binaries
# see here: https://docs.dgl.ai/en/0.4.x/install/index.html#install-from-conda

# conda install -c dglteam dgllife
# For the MPNN
# Currently, we need to install from source due to a deprecated package in dgl
# unsure if this is going to work in the CI pipeline but we'll try it!
git clone https://github.com/awslabs/dgl-lifesci.git
cd dgl-lifesci/python
python setup.py install

conda install -c conda-forge jupyterlab
conda install -c anaconda networkx
conda install -c conda-forge matplotlib
conda install -c anaconda pandas
conda install -c anaconda pytest
conda install -c conda-forge glob2
conda install -c rdkit rdkit
conda install -c conda-forge pymatgen

# Optional, linting
conda install -c anaconda flake8
```

## Logging protocol
This package uses the `logging` module to provide the user with information about what is happening during the program execution. The default logging stream pipes to the terminal and is usually abbreviated as `dlog` in the code. The following guidelines should be obeyed when logging.
* Inform the user of critical terminating issues via the `critical` level. These should be used right before raising an error.
* Inform the user of a potentially critical issue via the `error` level. These are intended to be similar to critical except it is not the case that the program is immediately terminated. Likely, there will be another error thrown later on.
* Inform the user of possible issues with their use of the program with the `warning` level.
* Log everything that the user should know with `dlog.info("...")`.

## Testing
To run the unit tests, we generally use
```bash
coverage run --source=crescendo -m pytest
```
and to view the report,
```bash
coverage report -m
```
These checks are also used in the unit tests for the code run in the CI pipeline.
