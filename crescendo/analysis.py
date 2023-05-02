"""Core analysis module for Crescendo. Used for loading and analyzing models
and data."""

from functools import cached_property, cache
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
import torch
from yaml import safe_load

from crescendo import utils

console = Console()


class Estimator:
    """The result base class. Corresponds to a single .hydra file. It is
    assumed that any directory containing the .hydra directory also contains
    directories checkpoints and logs.
    """

    @cached_property
    def hydra_config(self):
        """Loads and returns the .hydra/config.yaml file as a dictionary."""

        path = Path(self._results_dir) / ".hydra" / "config.yaml"
        d = safe_load(open(path, "r"))
        console.log(f"Loaded config.yaml from {path}")
        return d

    @cached_property
    def hydra_internal_config(self):
        """Loads and returns the .hydra/hydra.yaml file as a dictionary."""

        path = Path(self._results_dir) / ".hydra" / "hydra.yaml"
        d = safe_load(open(path, "r"))
        console.log(f"Loaded hydra.yaml from {path}")
        return d

    @cached_property
    def hydra_overrides(self):
        """Loads and returns the .hydra/overrides.yaml file as a list."""

        path = Path(self._results_dir) / ".hydra" / "overrides.yaml"
        d = safe_load(open(path, "r"))
        console.log(f"Loaded overrides.yaml from {path}")
        return d

    @cached_property
    def metrics(self):
        """Loads and returns the metrics.csv file. This also does some
        lightweight parsing, since the default way that PyTorch Lightning saves
        things is a bit... strange."""

        path = Path(self._results_dir) / "logs" / "version_0" / "metrics.csv"
        return pd.read_csv(path)

    @property
    def results_dir(self):
        """Returns the directory location corresponding to the parent of the
        .hydra file.

        Returns
        -------
        str
        """

        return self._results_dir

    @cached_property
    def config(self):
        """Loads the config as an omegaconf.DictConfig. Possibly modifies the
        path to the data, since that might be different depending on which
        machine training/testing takes place on."""

        path = Path(self.results_dir) / "final_config.yaml"
        config = utils.omegaconf_from_yaml(path)

        # If the data_dir is set explicitly, we override the config here
        if self._data_dir is not None:
            console.log(
                f"Results data_dir is explicitly set to {self._data_dir}. "
                "Internal self.config will be appended with this directory.",
                style="bold yellow",
            )
            config.data["data_dir"] = self._data_dir

        return config

    @cached_property
    def best_checkpoint(self):
        path = Path(self.results_dir) / "checkpoints" / "best-v1.ckpt"
        if not path.exists():
            console.log(
                f"Note best_checkpoint path={path} does not exist! This will "
                "be set to None by default.",
                style="bold red",
            )
            return None
        return str(path)

    @utils.log_warnings()
    @cache
    def get_model(self, checkpoint=None):
        """Loads the model from the provided checkpoint file. If None, it will
        attempt to load the model from a checkpoint matching the signature
        best-v1.ckpt.

        Parameters
        ----------
        checkpoint : None, optional
            The path to the checkpoint file.

        Returns
        -------
        lightning.LightningModule
            The PyTorch Lightning model, possibly loaded from checkpoint.
        """

        if checkpoint is None:
            checkpoint = self.best_checkpoint

        return utils.instantiate_model(self.config, checkpoint=checkpoint)

    @cache
    def get_datamodule(self):
        """Gets the datamodule used for the work."""

        return utils.instantiate_datamodule(self.config)

    @classmethod
    def from_root(klass, root, **kwargs):
        """Gets all of the paths matching final_config.yaml via Path.rglob."""

        paths = [str(xx) for xx in Path(root).rglob("final_config.yaml")]
        console.log(f"Found hydra paths {paths}")
        if len(paths) > 1:
            raise ValueError(
                "Result can only correspond to one final_config.yaml directory"
            )
        if len(paths) == 0:
            raise ValueError("No .hydra directory found")
        return klass(str(Path(paths[0]).parent.resolve()), **kwargs)

    def __init__(self, results_dir, data_dir=None):
        self._results_dir = results_dir
        self._data_dir = data_dir

    def predict(self, x):
        """Runs forward prediction on the model.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        x = torch.Tensor(x).float()
        model = self.get_model()
        model.eval()
        with torch.no_grad():
            return model.forward(x).detach().numpy()


class Ensemble:
    """Similar to the results class, the Ensemble class assumes that every
    directory you point it to with a ``final_config.yaml`` file in it is part
    of a larger ensemble. Each individual model will be used as part of this
    ensemble during inference."""

    @cached_property
    def estimators(self):
        """Gets a list of Estimator objects."""

        return [
            Estimator.from_root(root, data_dir=self._data_dir)
            for root in self._results_dirs
        ]

    @classmethod
    def from_root(klass, root, **kwargs):
        """Gets all of the paths matching final_config.yaml via Path.rglob."""

        paths = [xx for xx in Path(root).rglob("final_config.yaml")]
        console.log(
            f"Found {len(paths)} hydra paths (n_estimators=={len(paths)})"
        )
        if len(paths) == 0:
            raise ValueError("No .hydra directory found")
        paths = [str(xx.parent) for xx in paths]
        return klass(paths, **kwargs)

    def __init__(self, results_dirs, data_dir=None):
        self._results_dirs = results_dirs
        self._data_dir = data_dir

    def predict(self, x):
        """Runs forward prediction on the estimators. The estimator index is
        the zeroth axis of the returned array.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """

        x = torch.Tensor(x).float()
        return np.array([est.predict(x) for est in self.estimators])
