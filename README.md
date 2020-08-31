# crescendo
[![codecov](https://codecov.io/gh/x94carbone/crescendo/branch/master/graph/badge.svg?token=0M8IGBBWXQ)](https://codecov.io/gh/x94carbone/crescendo)
![Python package](https://github.com/x94carbone/crescendo/workflows/Python%20package/badge.svg?branch=master)

An artificial intelligence and machine learning pipeline currently under development, used for facilitating broad-spectrum machine learning applications with a low barrier of entry and focused on building reusable code and a robust workflow. The current focus is the [QM9 Dataset](http://quantum-machine.org/datasets/), with potential future applications to come.

## Usage
The current setup relies either on using Jupyter Notebooks for handling the pipeline, or the much more practical, and cluster friendly, CLI.

The repository depends heavily on some key environment variables. Currently, those are
```bash
QM9_ENV_VAR = "QM9_DATA_PATH"
QM8_EP_ENV_VAR = "QM8_EP_DATA_PATH"
QM9_DS_ENV_VAR = "QM9_DS_CACHE"
QM9_OXYGEN_FEFF_ENV_VAR = "QM9_O_FEFF_PATH"
```
* `QM9_ENV_VAR` points to the directory containing the raw QM9 data downloaded from [here](http://quantum-machine.org/datasets/)
* `QM8_EP_DATA_PATH` points to the text file containing the raw QM8 electronic property data as computed in the paper [TODO]
* `QM9_OXYGEN_FEFF_ENV_VAR` points to the pickle file containing oxygen XANES FEFF spectra computed on the QM9 molecules containing at least one oxygen atom
* `QM9_DS_ENV_VAR` is an extremely important one, as this points to the "cache" location where all pickled datasets, and machine learning data will be stored

It is recommended to set these environment variables by setting them in your `.bashrc` or `.bash_profile` configs via `export ENV_VAR="path/to/whatever.txt"`. Note that there are ways around using the environment variables, but it will make the CLI interface much more cumbersome. Thus, henceforth, we assume the environment variables are set.

Here are a few things you can do with the CLI:

### Get help
Using the CLI, you can get an overview of functionality by using `--help`, for example, this
```bash
python3 run.py --help
```
will give you the high-level options for the entire repository. Currently, there is just `qm9`, but as we build this there will hopefully be more datasets. You can also select the `qm9` flag and get help for that
```bash
python3 run.py qm9 --help
```
which will list a plethora of options; we will go through many of those use cases  for QM9 now.

### Construct the raw dataset
The first step of the pipeline is always to construct the "raw" dataset. We define this loosely as a lightweight representation of the data that has gone through the core pre-processing steps. In the case of `qm9`, this includes reading the core QM9 data, and possibly pairing it with other data computed _on_ the QM9 dataset. By default, the QM8 and QM9 Oxygen FEFF spectra will try to be paired; here's an example of loading QM9 with only the QM8 EP data:
```bash
python3 run.py --debug 10000 qm9 --dataset-raw ds10k_test --no-oxygen-xanes
```
Here, we set `debug 10000` so as to only load the first 10k (arbitrarily) to save time. If you wanted to do this on the full dataset, you would not specify anything for debug, e.g. `python3 run.py qm9 --dataset-raw ds10k --no-oxygen-xanes`. Next, we specify that we want to use the `qm9` data in the pipeline, and then specify what we want to do, in this case, `--dataset-raw` which takes the dataset name as the argument, in this case, we are calling it `ds10k`. Finally, we specify not to load in the oxygen XANES data, since presumably most people using this will not have access to it.

Typical output will look something like this (with timestamps suppressed)
```
Using backend: pytorch
INFO [qm9.py:241] Loading QM9 from /Users/mc/Data/qm9
INFO [qm9.py:250] Loading from 133885 geometry files
INFO [qm9.py:268] Total number of raw QM9 data points: 10000
INFO [timing.py:22] load done 5.14 s
INFO [qm9.py:285] Reading QM8 electronic properties from /Users/mc/Data/qm8/gdb8_22k_elec_spec.txt
INFO [qm9.py:310] Total number of data points read from qm8: 1627
INFO [timing.py:22] load_qm8_electronic_properties done 0.12 s
INFO [qm9.py:338] Loaded 113938 molecules of XANES successfully from /Users/mc/Data/qm9_feff_spectra/feff_oxygen.pkl
INFO [qm9.py:342] Length of the intersection is 8505
INFO [timing.py:22] load_oxygen_xanes done 1.21 s
INFO [qm9.py:66] Saved /Users/mc/Data/qm9_crescendo_cache/ds10k/raw.pkl
```

A few things to note:
* The code knows where to look for the QM9 data, as the `QM9_DATA_PATH` environment variable is already specified. The user can use the `--path` argument to override this default choice.
* The dataset name is critically important in the pipeline. In the next step, to build on the existing dataset, you will need to specify the same dataset name.

### Construct the graph dataset
Next, the code will ingest the previously created raw dataset and convert that into another object, which we call the Graph Dataset. This is a "machine learning-ready" dataset, which in the next step, will be directly called on to produce the `torch` Data Loaders, which are the workhorse of the training protocol. An example use case would be,
```bash
python3 run.py qm9 --dataset-graph ds10k --target-type qm8properties --targets-to-use 0 1 4 --scale-targets --split 0.03 0.03 0.94
```
There's a bit to unpack here, so let's go through it piece by piece. First, we specify the same dataset name but a different procedure with `--dataset-graph ds10k`. In this case, we want to construct the graph dataset from the raw dataset. The code knows where to look, and it successfully finds the `ds10k/raw.pkl` dataset we already created.

Next, the desired target properties are specified with `--target-type qm8properties --targets-to-use 0 1 4`. Note that those properties must have been loaded in the previous step; they need to actually be in the database. This part simply selects those properties to be _used_ as the targets for machine learning. In this case, we choose the QM8 electronic properties, and specifically the first, second and fifth columns (whatever those correspond to!). We can also scale the targets (recommended) with `--scale-targets`.

Finally, we choose the test/validation/training split for the data with proportions specified by `--split 0.03 0.03 0.94`. This information is saved internally to the class object and cannot be changed without passing a `force=True` argument to the method. This is done intentionally, since once the T/V/T splits are set, _they should never be changed for that dataset again_. This is to ensure bias is not introduced during random split selection. This split will be the same for the remainder of the training process. One can of course get around this by simply making a new dataset, forcing, or deleting this one and remaking it, but it is intended to serve as an annoying reminder that we should not do this.

Typical output will look something like this (with timestamps suppressed)

```
Using backend: pytorch
INFO    [qm9.py:80] Loaded from /Users/mc/Data/qm9_crescendo_cache/ds10k/raw.pkl
WARNING [qm9.py:413] Dataset seed set to None
INFO    [timing.py:22] to_mol done 1.63 s
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
Using backend: pytorch
INFO    [timing.py:22] to_graph done 30.83 s
INFO    [qm9.py:635] Using target type qm8properties
INFO    [qm9.py:636] Using target indexes [0, 1, 4]
INFO    [qm9.py:637] Scaling targets: True
INFO    [qm9.py:661] Total number of ML-ready datapoints 1627
INFO    [ml_utils.py:54] Mean/sd of target data is 2.28e-01 +/- 4.21e-02
INFO    [qm9.py:682] Target metadata is [array([0.21944734, 0.24758382, 0.21653111]), array([0.04368971, 0.03468454, 0.04793685])]
INFO    [ml_utils.py:54] Mean/sd of target data is -2.16e-15 +/- 1.00e+00
INFO    [timing.py:22] init_ml_data done 15.93 s
INFO    [base.py:127] T/V/T props: 0.03/0.03/0.94
INFO    [base.py:134] T/V/T lengths: 48/48/1531
INFO    [qm9.py:66] Saved /Users/mc/Data/qm9_crescendo_cache/ds10k/mld.pkl
```

A few things to note:
* The initial warning was thrown because we did not specify a seed for the splitting process. Thus, the process will not be reproducible. This is usually fine since we only make the dataset once, but it is worthy of a warning. The user can get around this by specifying the `--seed` command line argument.
* The many outputs of `Using backend: pytorch` is because we use multiprocessing to construct the graphs faster, since this can take some time. The computer this was run on has 12 virtual cores, and hence, 12 instances of `Using backend: pytorch`.
* Note the benefit of scaling the targets.
* Note that depending on the choices for the proportions, the splitting process may round +/- 1 data point, which is completely irrelevant for our purposes but is noted here anyway.
* The output metadata is the mean and standard deviation for each of the 3 selected target columns.

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
