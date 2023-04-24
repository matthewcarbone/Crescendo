#!/bin/bash -l

export HYDRA_FULL_ERROR=1

cr model=mlp data=california_housing
cr model=mlp data=california_housing debug=default
cr model=mlp data=california_housing debug=fdr
cr model=mlp data=california_housing debug=limit
cr model=mlp data=california_housing debug=overfit
cr model=mlp_random_architecture data=california_housing seed=123
cr model=mlp data=california_housing hparams_search=optuna
