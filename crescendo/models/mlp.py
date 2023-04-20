"""
BSD 3-Clause License

Copyright (c) 2022, Brookhaven Science Associates, LLC, Brookhaven National
Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from datetime import datetime
from copy import deepcopy
from functools import cache
from math import floor
from io import StringIO
from joblib import Parallel, delayed
import json
from multiprocessing import cpu_count
from os import rename
from pathlib import Path
import time
import sys
import warnings

from monty.json import MSONable
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer as plTrainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint



def _activation_map(s):
    """Maps a string corresponding to an activation function to the activation
    function itself."""

    if s == "relu":
        return nn.ReLU()
    elif s == "leaky_relu":
        return nn.LeakyReLU()
    elif s == "sigmoid":
        return nn.Sigmoid()
    elif s == "softplus":
        return nn.Softplus()
    elif s is None:
        return None
    else:
        raise ValueError(s)


class FeedforwardLayer(nn.Module):
    def __init__(
        self,
        *,
        input_size,
        output_size,
        activation=nn.ReLU(),
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()
        layers = [torch.nn.Linear(input_size, output_size)]
        if activation is not None:
            layers.append(activation)
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_size))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        *,
        architecture,
        dropout=0.0,
        activation=nn.ReLU(),
        last_activation=None,
        batch_norm=False,
        last_batch_norm=False,
    ):
        super().__init__()
        assert len(architecture) > 1

        layers = []
        for ii, (n, n2) in enumerate(zip(architecture[:-1], architecture[1:])):
            if ii == len(architecture) - 2:
                a = last_activation
                b = last_batch_norm
            else:
                a = activation
                b = batch_norm
            layers.append(
                FeedforwardLayer(
                    input_size=n,
                    output_size=n2,
                    activation=a,
                    dropout=dropout,
                    batch_norm=b,
                )
            )

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class _OptimizerSetter:
    def set_optimizer(self, optimizer, scheduler, scheduler_kwargs):
        """Flexible method for setting the optimizer and scheduler parameters
        of the model. This configures the ``_optimizer_arguments`` attributes,
        which are then used in the ``configure_optimizers`` method.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        scheduler : torch.optim.Scheduler
        scheduler_kwargs : dict
            Optional arguments to pass to the pytorch lightning API for
            monitoring the scheduler. For example, ``{"monitor": "val_loss"}``.
        """

        if scheduler is None:
            self._optimizer_arguments = {"optimizer": optimizer}
        self._optimizer_arguments = {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, **scheduler_kwargs},
        }

    def configure_optimizers(self):
        """Core method for pytorch lightning. Returns the
        ``_optimizer_arguments`` attribute.

        Returns
        -------
        dict
        """

        return self._optimizer_arguments


class _GeneralPLModule:
    def training_step(self, batch, batch_idx):
        loss = self._single_forward_step(batch, batch_idx)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._single_forward_step(batch, batch_idx)
        return {"loss": loss}

    def on_train_epoch_start(self):
        self._epoch_dt = time.time()

    def _log_outputs(self, outputs, what="train", keys=["loss"]):

        d = {}
        for key in keys:
            tmp_loss = torch.tensor([x[key] for x in outputs]).mean().item()
            d[key] = tmp_loss
            self.log(f"{what}_{key}", tmp_loss, on_step=False, on_epoch=True)
        return d

    def _single_forward_step(self, batch, batch_index):
        raise NotImplementedError

    def training_epoch_end(self, outputs):
        d = self._log_outputs(outputs, "train")
        epoch = self.trainer.current_epoch + 1
        dt = time.time() - self._epoch_dt
        if self._print_every_epoch > 0:
            if epoch % self._print_every_epoch == 0:
                loss = d["loss"]
                print(
                    f"\ttr loss {loss:.03e} | {(dt/60.0):.02f} m", flush=True
                )

    def validation_epoch_end(self, outputs):
        d = self._log_outputs(outputs, "val")
        epoch = self.trainer.current_epoch + 1

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True)

        if self._print_every_epoch > 0:
            if epoch % self._print_every_epoch == 0:
                loss = d["loss"]
                print(f"Epoch {epoch:05}")
                print(f"\tlr: {lr:.03e}")
                print(f"\tcv loss {loss:.03e}")

    def on_train_end(self):
        """Logs information as training ends."""

        if self.trainer.global_rank == 0:
            epoch = self.trainer.current_epoch + 1
            if epoch < self.trainer.max_epochs:
                print(
                    "Early stopping criteria reached at "
                    f"epoch {epoch}/{self.trainer.max_epochs}"
                )


class MultilayerPerceptron(
    _OptimizerSetter, _GeneralPLModule, LightningModule
):

    def __init__(
        self,
        *,
        input_size,
        hidden_sizes,
        output_size,
        dropout=0.01,
        batch_norm=True,
        activation="relu",
        last_activation="softplus",
        criterion="mae",
        last_batch_norm=False,
    ):
        hidden_sizes = np.array(hidden_sizes).tolist()
        super().__init__()
        self.save_hyperparameters()

        activation = _activation_map(activation)
        last_activation = _activation_map(last_activation)

        self._model = FeedForwardNeuralNetwork(
            architecture=[input_size, *hidden_sizes, output_size],
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            last_activation=last_activation,
            last_batch_norm=last_batch_norm,
        )

        if criterion == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        elif criterion == "mae":
            self.criterion = nn.L1Loss(reduction="mean")
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def forward(self, x):
        return self._model(x)


class Trainer(plTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def export_csv_log(
        self, columns=["epoch", "train_loss", "val_loss", "lr"]
    ):
        """Custom method for exporting the trainer logs to something much more
        readable. Only executes on the 0th global rank for DDP jobs."""

        if self.global_rank > 0:
            return

        metrics = self.logger.experiment.metrics
        log_dir = self.logger.experiment.log_dir

        path = Path(log_dir) / Path("custom_metrics.csv")
        t = pd.DataFrame([d for d in metrics if "train_loss" in d])
        v = pd.DataFrame([d for d in metrics if "val_loss" in d])
        df = pd.concat([t, v], join="outer", axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        try:
            df = df[columns]
            df.to_csv(path, index=False)
        except KeyError:
            print(
                "might be running overfit_batches, not saving custom metrics"
            )
            print(f"Current columns: {list(df.columns)}")

    def fit(self, **kwargs):
        print_every_epoch = 0
        if "print_every_epoch" in kwargs.keys():
            print_every_epoch = kwargs.pop("print_every_epoch")
        kwargs["model"]._print_every_epoch = print_every_epoch
        super().fit(**kwargs)
        self.export_csv_log()


@cache
def load_MultilayerPerceptron_from_ckpt(path):
    """Loads the MultilayerPerceptron from path, but the results are
    cached so as to speed up the loading process dramatically.

    Parameters
    ----------
    path : os.PathLike

    Returns
    -------
    MultilayerPerceptron
    """

    return MultilayerPerceptron.load_from_checkpoint(path)


def set_optimizer(
    model,
    lr=1e-2,
    patience=10,
    min_lr=1e-7,
    factor=0.95,
    monitor="val_loss",
):
    local = {
        key: value for key, value in locals().items() if key != "model"
    }
    print(f"Setting OPTIMIZER and SCHEDULER: {local}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=patience,
        min_lr=min_lr,
        factor=factor,
    )
    scheduler_kwargs = {"monitor": monitor}
    model.set_optimizer(optimizer, scheduler, scheduler_kwargs)


class SingleEstimator(MSONable):
    """The single estimator is a container for metadata that "points" to a
    saved model, and handles all training and metadata tracking. The models
    are not contained in the Estimator."""

    def get_trainer(
        self,
        early_stopping_kwargs={
            "monitor": "val_loss",
            "check_finite": True,
            "patience": 100,
            "verbose": False,
        },
        gpus=None,
        max_epochs=10000,
    ):
        """Initializes and returns the trainer object.
        
        Parameters
        ----------
        early_stopping_kwargs : dict, optional
            Description
        gpus : None, optional
            Description
        max_epochs : int, optional
        monitor : str, optional
        early_stopper_patience : int, optional
        
        Returns
        -------
        pl.Trainer
        """

        local = {
            key: value for key, value in locals().items() if key != "self"
        }
        print(f"Setting TRAINER: {local}")

        logger = CSVLogger(self._root, name="Logs")
        early_stopper = EarlyStopping(**early_stopping_kwargs)

        cuda = torch.cuda.is_available()
        checkpointer = ModelCheckpoint(
            dirpath=f"{self._root}/Checkpoints",
            save_top_k=5,
            monitor="val_loss"
        )

        if gpus is None:
            gpus = int(cuda)
            auto_select_gpus = bool(cuda)
        else:
            assert isinstance(gpus, int)
            auto_select_gpus = True

        return Trainer(
            gpus=gpus,
            num_nodes=1,
            auto_select_gpus=auto_select_gpus,
            precision=32,
            max_epochs=max_epochs,
            enable_progress_bar=False,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[early_stopper, checkpointer],
            enable_model_summary=True,
        )

    def _set_root(self, root):
        if root is not None:
            self._root = str(root)
            Path(self._root).mkdir(exist_ok=True, parents=True)
        else:
            self._root = None

    @property
    def best_checkpoint(self):
        checkpoint = Path(self._best_checkpoint)
        if checkpoint.exists():
            print(f"Loading from checkpoint {checkpoint}")
            return str(checkpoint)

        # If the path does not exist, it might be because the absolute paths
        # were different.
        L = len(checkpoint.parts)
        for ii in range(1, L):
            new_path_test = Path(*checkpoint.parts[ii:])
            if new_path_test.exists():
                print(f"Loading from checkpoint {new_path_test}")
                return str(new_path_test)
        else:
            raise RuntimeError(f"Checkpoint {checkpoint} does not exist")

    @property
    def best_model(self):
        return load_MultilayerPerceptron_from_ckpt(
            self.best_checkpoint
        )

    def __init__(
        self,
        root=None,
        best_checkpoint=None,
        latest_train_args=None
    ):
        self._set_root(root)
        self._best_checkpoint = best_checkpoint
        self._latest_train_args = latest_train_args

    def train(
        self,
        *,
        model,
        data,
        batch_size=4096,
        persistent_workers=True,
        pin_memory=True,
        num_workers=3,
        epochs=100,
        lr=1e-2,
        patience=20,
        min_lr=1e-7,
        factor=0.95,
        monitor="val_loss",
        early_stopper_patience=100,
        gpus=None,
        print_every_epoch=10
    ):
        """Trains a model.

        Parameters
        ----------
        model : MultilayerPerceptron, optional
        training_data : numpy.ndarray
        validation_data : numpy.ndarray
        batch_size : int, optional
        persistent_workers : bool, optional
        pin_memory : bool, optional
        num_workers : int, optional
        epochs : int, optional
        lr : float, optional
        patience : int, optional
        min_lr : float, optional
        factor : float, optional
        monitor : str, optional
        early_stopper_patience : int, optional
        gpus : None, optional
        downsample_training_proportion : float, optional
        parallel : bool, optional
        print_every_epoch : int, optional
        """

        self._latest_train_args = {
            key: value for key, value in locals().items()
            if key not in [
                "model", "training_data", "validation_data", "self"
            ]
        }

        # Execute the training using a lot of defaults/boilerplate
        set_optimizer(
            model,
            lr=lr,
            patience=patience,
            min_lr=min_lr,
            factor=factor,
            monitor=monitor,
        )

        trainer = self.get_trainer(
            max_epochs=epochs,
            early_stopper_patience=early_stopper_patience,
            gpus=gpus,
            monitor=monitor,
        )

        # Execute training
        trainer.fit(
            model,
            datamodule=data,
            print_every_epoch=print_every_epoch   
        )

        self._best_checkpoint = trainer.checkpoint_callback.best_model_path

        path = Path(self._root) / Path("estimator.json")
        with open(path, 'w') as outfile:
            json.dump(self.as_dict(), outfile, indent=4, sort_keys=True)

    def predict(self, x, model=None):
        """Makes a prediction on the provided data.

        Parameters
        ----------
        x : numpy.array

        Returns
        -------
        numpy.array
        """

        if model is None:
            model = self.best_model

        x = torch.Tensor(x)
        model.eval()
        with torch.no_grad():
            return model.forward(x).detach().numpy()


class Ensemble(MSONable):

    @property
    def root(self):
        return self._root

    @property
    def estimators(self):
        return self._estimators

    def __init__(self, root, estimators=[]):
        self._root = str(root)
        self._estimators = estimators

    def _get_ensemble_model_root(self, estimator_index):
        return Path(self._root) / Path(f"{estimator_index:06}")

    def train(
        self,
        *,
        model,
        training_data,
        validation_data,
        estimator_index=None,
        **kwargs
    ):
        if estimator_index is None:
            # Check the length of the current dictionary
            L = len(self._estimators)
            if L == 0:
                estimator_index = 0
            else:
                estimator_index = L + 1
        root = self._get_ensemble_model_root(estimator_index)
        estimator = SingleEstimator(root=root)
        estimator.train(
            model=model,
            training_data=training_data,
            validation_data=validation_data,
            **kwargs,
        )
        self._estimators.append(estimator)

        # Dump current state
        path = Path(self._root) / Path("ensemble.json")
        save_json(self.as_dict(), path)

    def train_from_random_architecture(
        self,
        *,
        training_data,
        validation_data,
        estimator_index=None,
        min_layers=4,
        max_layers=8,
        min_neurons_per_layer=160,
        max_neurons_per_layer=300,
        dropout=0.0,
        batch_norm=True,
        activation="leaky_relu",
        last_activation="softplus",
        criterion="mae",
        last_batch_norm=False,
        sort_architecture=False,
        seed=None,
        **kwargs
    ):
        """Executes a training procedure but initializes the
        ``MultilayerPerceptron`` from a random architecture.

        Parameters
        ----------
        training_data : numpy.ndarray
        validation_data : numpy.ndarray
        estimator_index : int, optional
        **kwargs
            Extra keyword arguments to pass through to training.
        """

        if seed is not None:
            np.random.seed(seed)
        n_hidden_layers = np.random.randint(
            low=min_layers,
            high=max_layers + 1
        )
        architecture = np.random.randint(
            low=min_neurons_per_layer,
            high=max_neurons_per_layer,
            size=(n_hidden_layers,)
        )
        if sort_architecture:
            architecture = sorted(architecture.tolist())
        model = MultilayerPerceptron(
            input_size=training_data["x"].shape[1],
            hidden_sizes=architecture,
            output_size=training_data["y"].shape[1],
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            last_activation=last_activation,
            criterion=criterion,
            last_batch_norm=last_batch_norm,
        )
        self.train(
            model=model,
            training_data=training_data,
            validation_data=validation_data,
            estimator_index=estimator_index,
            **kwargs
        )

    def train_ensemble_serial_from_random_architectures(
        self,
        training_data,
        validation_data,
        n_estimators=10,
        from_random_architecture_kwargs={
            "min_layers": 4,
            "max_layers": 8,
            "min_neurons_per_layer": 160,
            "max_neurons_per_layer": 300,
            "dropout": 0.0,
            "batch_norm": True,
            "activation": "leaky_relu",
            "last_activation": "softplus",
            "criterion": "mae",
            "last_batch_norm": False,
            "sort_architecture": False,
        },
        use_seeds=False,
        **kwargs
    ):
        """Trains the entire ensemble in serial. Default behavior is to
        reload existing models from checkpoint and to train them using the
        provided learning rate, and other parameters.

        Parameters
        ----------
        training_data : TYPE
            Description
        validation_data : TYPE
            Description
        n_estimators : TYPE, optional
            Description
        from_random_architecture_kwargs : dict, optional
            Description
        use_seeds : bool, optional
            If True, uses ``seed==estimator_index`` for generating the random
            architectures.
        **kwargs
            Description
        """

        estimator_indexes = [ii for ii in range(n_estimators)]
        L = len(self._estimators)
        if L > 0:
            estimator_indexes = [xx + L for xx in estimator_indexes]
            print(
                "Ensemble already has estimators: new estimator indexes "
                f"are {estimator_indexes}"
            )
        for estimator_index in estimator_indexes:
            self.train_from_random_architecture(
                training_data=training_data,
                validation_data=validation_data,
                estimator_index=estimator_index,
                **from_random_architecture_kwargs,
                seed=estimator_index if use_seeds else None,
                **kwargs
            )

    def predict(self, x):
        """Predicts on the provided data in ``x`` by loading the best models
        from disk.

        Parameters
        ----------
        x : numpy.array
        """

        results = []
        for estimator in self._estimators:
            results.append(estimator.predict(x))
        return np.array(results)

    def predict_filter_outliers(
        self,
        x,
        sd_mult=3,
        threshold_sd=0.75,
        max_spectra_value=100.0,
        threshold_zero=0.5,
        min_spectra_value=0.05
    ):
        """Summary
        
        Parameters
        ----------
        x : TYPE
            Description
        sd_mult : int, optional
            Description
        threshold_sd : float, optional
            Description
        
        Returns
        -------
        numpy.ma.core.MaskedArray
            Description
        """

        # predictions is of shape N_ensembles x N_examples x M
        predictions = self.predict(x)

        # Get the mean and standard deviations over the ensembles
        # These are now of shape N_examples x M
        mu = predictions.mean(axis=0, keepdims=True)
        sd = predictions.std(axis=0, keepdims=True)
        _sd = sd_mult * sd

        # cond is of shape N_ensembles x N_examples x M
        # For every example, we want to drop certain estimator predictions
        c1 = (predictions > mu + _sd) | (predictions < mu - _sd)
        c1 = c1.mean(axis=2) > threshold_sd

        # We also have sensible heuristics... for example, none of the
        # predictions should be greater than 100. None can be negative due to
        # the usual softmax output
        c2 = np.any(predictions > max_spectra_value, axis=2)

        # Finally, if the majority of the predicted points are around 0, this
        # is clearly unphysical. This will catch some predictions where there
        # are random, sharply peaked features.
        c3 = (predictions < min_spectra_value).mean(axis=2) > threshold_zero

        # The total condition
        cond = c1 | c2 | c3

        # where_keep is of shape N_ensembles x N_examples (I think)
        where_discard = np.where(cond)

        # This is where it gets a little tricky. Set these values to nan
        predictions[where_discard] = np.nan

        # Now generate a MASKED array where the mask is where the predictions
        # are nans
        final_preds = np.ma.array(predictions, mask=np.isnan(predictions))

        # Taking the means/standard deviations will safely ignore the nan'd
        # values
        return final_preds
