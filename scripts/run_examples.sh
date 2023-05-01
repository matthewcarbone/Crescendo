#!/bin/bash -l

set -e

export HYDRA_FULL_ERROR=1

cd examples || exit

cd 00_xas_to_functional_groups/00a_hyperparameter_tune || exit
bash run.sh
cd ../..

cd 00_xas_to_functional_groups/00b_ensemble_training || exit
bash run.sh
cd ../..
