#!/bin/bash -l

export HYDRA_FULL_ERROR=1

cd ../examples/00_xas_to_functional_groups || exit
bash run.sh
