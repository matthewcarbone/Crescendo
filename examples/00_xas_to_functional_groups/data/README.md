# Data

This directory contains the core training data used to train the models in this project. Here, we store a subset of the carbon data, which itself is a subset of the overall database containing molecules with at least one carbon atom.

In the `C-XANES` database, we have:

-   `X_*.npy` for `* == {"train", "val", "test"}`. These are training, validation and testing inputs for the ML model, stored as compressed NumPy binaries.
-   `Y_*.npy` for `* == {"train", "val", "test"}`. These are training, validation and testing targets for the ML model, stored as compressed NumPy binaries.
-   `functional_groups.txt`. The functional group names for the targets in the directory.
-   `splits.json`. Generated from `crescendo.preprocess.array:ensemble_split` (with scikit-learn's `KFold` backend), these are overlapping training splits used to create model diversity when training ensembles. For testing here, we use 5 splits.
