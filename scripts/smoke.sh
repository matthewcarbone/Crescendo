#!/bin/bash -l

export HYDRA_FULL_ERROR=1

cr model=mlp data=california_housing debug=default
cr model=mlp data=california_housing debug=fdr
cr model=mlp data=california_housing debug=limit
cr model=mlp data=california_housing debug=overfit
