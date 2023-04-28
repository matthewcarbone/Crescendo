#!/bin/bash -l

export HYDRA_FULL_ERROR=1

cd examples/00_xas_to_functional_groups/00a_hyperparameter_tune || exit
bash run.sh
cd ../../..
