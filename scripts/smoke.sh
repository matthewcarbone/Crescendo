#!/bin/bash

cr model=mlp data=california_housing debug=default
cr model=mlp data=california_housing debug=fdr
cr model=mlp data=california_housing debug=limit
cr model=mlp data=california_housing debug=overfit
