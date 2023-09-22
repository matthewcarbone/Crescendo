"""Core analysis module for Crescendo. Used for loading and analyzing models
and data."""

from functools import cached_property, cache
from pathlib import Path

import numpy as np
import pandas as pd
from rich.jupyter import print
import torch
from yaml import safe_load

from crescendo import utils


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
        if self._verbose:
            print(f"Loaded config.yaml from {path}")
        return d

    @cached_property
    def hydra_internal_config(self):
        """Loads and returns the .hydra/hydra.yaml file as a dictionary."""

        path = Path(self._results_dir) / ".hydra" / "hydra.yaml"
        d = safe_load(open(path, "r"))
        if self._verbose:
            print(f"Loaded hydra.yaml from {path}")
        return d

    @cached_property
    def hydra_overrides(self):
        """Loads and returns the .hydra/overrides.yaml file as a list."""

        path = Path(self._results_dir) / ".hydra" / "overrides.yaml"
        d = safe_load(open(path, "r"))
        if self._verbose:
            print(f"Loaded overrides.yaml from {path}")
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
            if self._verbose:
                print(
                    f"Results data_dir is explicitly set to {self._data_dir}. "
                    "Internal self.config will be appended with this "
                    "directory.",
                    style="bold yellow",
                )
            config.data["data_dir"] = self._data_dir

        return config

    @cached_property
    def best_checkpoint(self):
        path = Path(self.results_dir) / "checkpoints"
        paths = sorted(list(path.rglob("*.ckpt")))

        if self._use_checkpoint == "best_epoch":
            path = paths[-2]
        elif self._use_checkpoint == "last":
            path = paths[-1]
        else:
            raise ValueError(f"Invalid use_checkpiont {self._use_checkpoint}")

        if not path.exists():
            print(
                f"Note best_checkpoint path={path} does not exist! This will "
                "be set to None by default.",
                style="bold red",
            )
            return None
        return str(path)

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

    @cached_property
    def X_train(self):
        dm = self.get_datamodule()
        return dm.X_train

    @cached_property
    def X_train_scaled(self):
        return self.get_datamodule().X_train_scaled

    @cached_property
    def X_val(self):
        return self.get_datamodule().X_val

    @cached_property
    def X_val_scaled(self):
        return self.get_datamodule().X_val_scaled

    @cached_property
    def X_test(self):
        return self.get_datamodule().X_test

    @cached_property
    def X_test_scaled(self):
        return self.get_datamodule().X_test_scaled

    @cached_property
    def Y_train(self):
        return self.get_datamodule().Y_train

    @cached_property
    def Y_val(self):
        return self.get_datamodule().Y_val

    @cached_property
    def Y_test(self):
        return self.get_datamodule().Y_test

    @classmethod
    def from_root(klass, root, verbose=True, **kwargs):
        """Gets all of the paths matching final_config.yaml via Path.rglob."""

        paths = [str(xx) for xx in Path(root).rglob("final_config.yaml")]
        if verbose:
            print(f"Found hydra paths {paths}")
        if len(paths) > 1:
            raise ValueError(
                "Result can only correspond to one final_config.yaml directory"
            )
        if len(paths) == 0:
            raise ValueError("No .hydra directory found")
        return klass(
            str(Path(paths[0]).parent.resolve()), verbose=verbose, **kwargs
        )

    def __init__(
        self,
        results_dir,
        data_dir=None,
        verbose=True,
        use_checkpoint="best_epoch",
    ):
        self._results_dir = results_dir
        self._data_dir = data_dir
        self._verbose = verbose
        self._use_checkpoint = use_checkpoint

    @staticmethod
    def _predict_dnn(model, x):
        x = torch.Tensor(x).float()
        with torch.no_grad():
            return model.forward(x).detach().numpy()

    def predict(self, x):
        """Runs forward prediction on the model.

        Parameters
        ----------
        x
            Depending on the type of prediction to perform, x might be a numpy
            array, or list of objects. For example, in the case of the MLPs,
            x is a numpy array. However, for MPNNs, it is a list of DGL graph
            objects.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            If the detected model type is not known.
        """

        model_type = self.config["model"]["_target_"]

        model = self.get_model()
        model.eval()

        if "crescendo.models.mlp" in model_type:
            return self._predict_dnn(model, x)

        raise ValueError(f"Model type {model_type} unknown")


class ModelSet:
    @cached_property
    def estimators(self):
        """Gets a list of Estimator objects."""

        return [
            Estimator.from_root(
                root,
                data_dir=self._data_dir,
                verbose=self._verbose,
                use_checkpoint=self._use_checkpoint,
            )
            for root in self._results_dirs
        ]

    @classmethod
    def from_root(klass, root, **kwargs):
        """Gets all of the paths matching final_config.yaml via Path.rglob."""

        paths = [xx for xx in Path(root).rglob("final_config.yaml")]
        print(f"Found {len(paths)} hydra paths (n_estimators=={len(paths)})")
        if len(paths) == 0:
            raise ValueError("No .hydra directory found")
        if "data_dir" in kwargs.keys():
            d = kwargs["data_dir"]
            print(f"data_dir will be overridden to {d}")
        paths = [str(xx.parent) for xx in paths]
        return klass(paths, **kwargs)

    def __init__(
        self,
        results_dirs,
        verbose=False,
        data_dir=None,
        use_checkpoint="best_epoch",
    ):
        self._results_dirs = results_dirs
        self._data_dir = data_dir
        self._verbose = verbose
        self._use_checkpoint = use_checkpoint


class EnsembleValTestMixin:
    """Defines the validation and test data properties for convenience."""

    @cached_property
    def X_val(self):
        if len(self.estimators) > 1:
            x1 = self.estimators[0].X_val
            x2 = self.estimators[1].X_val
            assert np.allclose(x1, x2)
        return self.estimators[0].X_val

    @cached_property
    def X_test(self):
        if len(self.estimators) > 1:
            x1 = self.estimators[0].X_test
            x2 = self.estimators[1].X_test
            assert np.allclose(x1, x2)
        return self.estimators[0].X_test

    @cached_property
    def Y_val(self):
        if len(self.estimators) > 1:
            x1 = self.estimators[0].Y_val
            x2 = self.estimators[1].Y_val
            assert np.allclose(x1, x2)
        return self.estimators[0].Y_val

    @cached_property
    def Y_test(self):
        if len(self.estimators) > 1:
            x1 = self.estimators[0].Y_test
            x2 = self.estimators[1].Y_test
            assert np.allclose(x1, x2)
        return self.estimators[0].Y_test


class HPTunedSet(ModelSet, EnsembleValTestMixin):
    """A set of models which are assumed to have used some Hydra sweeper to
    find optimal parameters."""

    @cached_property
    def X_val_scaled(self):
        if len(self.estimators) > 1:
            x1 = self.estimators[0].X_val_scaled
            x2 = self.estimators[1].X_val_scaled
            assert np.allclose(x1, x2)
        return self.estimators[0].X_val_scaled

    @cached_property
    def X_test_scaled(self):
        if len(self.estimators) > 1:
            x1 = self.estimators[0].X_test_scaled
            x2 = self.estimators[1].X_test_scaled
            assert np.allclose(x1, x2)
        return self.estimators[0].X_test_scaled

    def get_best_estimator(self, X=None, Y=None, metric=None):
        """Evaluates all found models on the validation set, and returns the
        estimator with the best performance.

        Parameters
        ----------
        X : TYPE
            Description
        Y : TYPE
            Description
        metric : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """

        if metric is None:

            def metric(pred, truth):
                return np.mean((pred - truth) ** 2)

        if X is None:
            X = self.X_val
        if Y is None:
            Y = self.Y_val

        preds = [est.predict(X) for est in self.estimators]
        results = [metric(pred, Y) for pred in preds]
        argmin = np.argmin(results)
        return self.estimators[argmin], results[argmin]


class Ensemble(ModelSet, EnsembleValTestMixin):
    """Similar to the results class, the Ensemble class assumes that every
    directory you point it to with a ``final_config.yaml`` file in it is part
    of a larger ensemble. Each individual model will be used as part of this
    ensemble during inference."""

    def predict(self, x):
        """Runs forward prediction on the estimators. The estimator index is
        the zeroth axis of the returned array.

        Parameters
        ----------
        x : numpy.ndarray
            Assumed to be unscaled.

        Returns
        -------
        numpy.ndarray
        """

        x = torch.Tensor(x).float()

        # Ignore the standard output
        return np.array(
            [est.predict(x, scale_forward=True) for est in self.estimators]
        )
