<div align="center">

# Crescendo

[![python](https://img.shields.io/badge/-Python_3.9+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/) <br>
[![image](https://github.com/matthewcarbone/crescendo/actions/workflows/smoke.yml/badge.svg)](https://github.com/matthewcarbone/crescendo/actions/workflows/smoke.yml)
[![image](https://github.com/matthewcarbone/crescendo/actions/workflows/examples.yml/badge.svg)](https://github.com/matthewcarbone/crescendo/actions/workflows/examples.yml)
[![image](https://github.com/matthewcarbone/crescendo/actions/workflows/unit.yml/badge.svg)](https://github.com/matthewcarbone/crescendo/actions/workflows/unit.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d43b1194e52f42339cc0b896f53b1ec8)](https://app.codacy.com/gh/matthewcarbone/Crescendo/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


Crescendo provides a unified command line + API for training and evaluating Lightning models

</div>

------------

⚠️ **Crescendo is a work in progress and highly subject to change**

🙏 Some of our boilerplate is based on the wonderful template by [ashleve](https://github.com/ashleve)! See [here](https://github.com/ashleve/lightning-hydra-template).

## Summary

⭐️ Crescendo leverages the power of [Hydra](https://hydra.cc), [Lightning](https://lightning.ai) and the humble command line to make executing the training of neural networks as easy as possible.

⭐️ Hydra supports an incredible suite of tools such as powerful approaches for hyperparameter tuning. These are built in and accessible.

⭐️ Loading your models will be handled with the `crescendo.analysis` API, so you can train your models via the command line on a supercomputer, then load the results in your local Jupyter notebook.

## Install

⚠️ **Coming soon!**

You can easily install Crescendo via Pip!

```bash
pip install crescendo
```

Of particular note, this not only installs the `crescendo` module, but also the `cr` command line executable. A simple example to test that everything is working properly:

```bash
cr model=mlp data=california_housing
```

## Acknowledgement

This research is based upon work supported by the U.S. Department of Energy, Office of Science, Office Basic Energy Sciences, under Award Number FWP PS-030. This research used resources of the Center for Functional Nanomaterials (CFN), which is a U.S. Department of Energy Office of Science User Facility, at Brookhaven National Laboratory under Contract No. DE-SC0012704. This software is also based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-FG02-97ER25308.

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.

IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.
