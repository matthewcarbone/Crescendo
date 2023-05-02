#!/bin/bash -l

set -e

export HYDRA_FULL_ERROR=1

cr model=mlp data=california_housing
cr model=mlp model.architecture=4 data=california_housing
cr model=mlp model.architecture=4 data=california_housing
cr model=mlp data=california_housing debug=default
cr model=mlp data=california_housing debug=limit
cr model=mlp data=california_housing debug=overfit
cr model=mlp_random_architecture data=california_housing seed=123
cr model=mlp data=california_housing hparams_search=optuna
cr model=mlp data=california_housing hparams_search=optuna \
    'model.architecture=choice(1, 2, 3, 4, 5)'
