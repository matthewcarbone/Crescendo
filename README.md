# crescendo
A greatly abstracted artificial intelligence and machine learning suite built for ease of use and broad-spectrum application.

## Installation
We recommend installing via `conda`.

### Manual installation
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
```
