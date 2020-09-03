# Utils
A general utilities module.

## Argument parsing
All logic for parsing arguments using `argparse` will be contained in `arg_parser.py`.

## Logging
The core logging module is `logger.py`. Various logging streams can be defined here and accessed throughout the codebase.

## Machine learning utilities
All utilities corresponding to general machine learning, including the very useful `Meter` class, are contained in `ml_utils.py`.

## Managers
Utilities specific to the _training_ pipeline are stored in `managers.py`. This contains a `Manager` class for each project that is contained in `crescendo`, which "manages" everything in the pipeline from start to finish.

## `rdkit` Mol utilities
Extra analysis utilities for providing summaries of `rdkit.Chem.Mol` objects are contained in `mol_utils.py`.

## Pure Python utilities
Pure python utilities are contained in `py_utils.py`.

## Random graph generation
See `graphs.py`.

## Timing
Utilities having to do with monitoring the execution time of functions are contained in `timing.py`.
