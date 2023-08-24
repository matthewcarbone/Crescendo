#!/bin/bash -l

set -e

export HYDRA_FULL_ERROR=1

cd examples || exit

current_directory=$(pwd)

directories=(
    "00_xas_to_functional_groups/00a_hyperparameter_tune"
    "00_xas_to_functional_groups/00b_ensemble_training"
    "01_graph_to_xas"
)

for d in "${directories[@]}"; do
    cd "$d"
    echo "in directory $d"
    bash run.sh
    cd "$current_directory"
done
