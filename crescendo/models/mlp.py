"""Container for various LightningModule models
Code is modified based off of
https://github.com/ashleve/lightning-hydra-template/blob/
89194063e1a3603cfd1adafa777567bc98da2368/src/models/mnist_module.py

MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from time import perf_counter

import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MeanMetric

from crescendo import logger


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
        input_dims,
        architecture,
        output_dims,
        dropout=0.0,
        activation=nn.ReLU(),
        last_activation=None,
        batch_norm=False,
        last_batch_norm=False,
    ):
        super().__init__()
        assert len(architecture) >= 1
        architecture = [input_dims, *architecture, output_dims]

        layers = []
        for ii, (n, n2) in enumerate(zip(architecture[:-1], architecture[1:])):
            if ii == len(architecture) - 2:
                a = last_activation
                b = last_batch_norm
                c = 0.0
            else:
                a = activation
                b = batch_norm
                c = dropout
            layers.append(
                FeedforwardLayer(
                    input_size=n,
                    output_size=n2,
                    activation=a,
                    dropout=c,
                    batch_norm=b,
                )
            )

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class MultilayerPerceptron(LightningModule):
    """A standard multilayer perceptron.

    Previous kwargs

    TODO: docstring here

    input_dims,
    architecture,
    output_dims,
    optimizer,
    dropout=0.0,
    activation=nn.ReLU(),
    last_activation=None,
    batch_norm=False,
    last_batch_norm=False,
    scheduler=None,
    criterion=nn.MSELoss(),
    """

    def __init__(
        self,
        *,
        input_dims,
        architecture,
        output_dims,
        optimizer,
        dropout,
        activation,
        last_activation,
        batch_norm,
        last_batch_norm,
        scheduler,
        criterion,
        print_every,
        lr_scheduler_kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = FeedForwardNeuralNetwork(
            input_dims=input_dims,
            architecture=architecture,
            output_dims=output_dims,
            dropout=dropout,
            activation=activation,
            last_activation=last_activation,
            batch_norm=batch_norm,
            last_batch_norm=last_batch_norm,
        )
        self.criterion = criterion
        self._t_epoch_started = None
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        """Steps the model for one minibatch.

        Parameters
        ----------
        batch : tuple
            Contains the (x, y) feature-target data.

        Returns
        -------
        tuple
            The loss, predictions and ground truth y value.
        """

        x, y = batch
        ypred = self.forward(x)
        loss = self.criterion(ypred, y)
        return loss, ypred, y

    def on_train_start(self):
        self.train_loss.reset()
        self.val_loss.reset()

    def on_train_epoch_start(self):
        self._t_epoch_started = perf_counter()

    def training_step(self, batch, batch_idx):
        """Executes a single training step, logs information, etc.

        Parameters
        ----------
        batch : TYPE
            Description
        batch_idx : TYPE
            Description
        """

        loss, ypred, y = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss, ypred, y = self.model_step(batch)
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_epoch_end(self):
        # Sometimes this is called on the trainer.validate step, and in those
        # cases the train_loss has not been updated.
        if self._t_epoch_started is None or self.train_loss._update_count == 0:
            return

        dt = perf_counter() - self._t_epoch_started
        avg_loss = self.train_loss.compute()
        avg_val_loss = self.val_loss.compute()
        lr = self.optimizers().param_groups[0]["lr"]

        if self.current_epoch == 0:
            logger.info("Epoch  | t (s)  | T loss     | V loss     | LR      ")
            logger.info("----------------------------------------------------")

        if (
            self.current_epoch == 0
            or self.current_epoch % self.hparams.print_every == 0
        ):
            logger.info(
                f"{self.current_epoch:06d} | {dt:.02f}   | {avg_loss:.02e}   "
                f"| {avg_val_loss:.02e}   | {lr:.02e}   "
            )

    def test_step(self, batch, batch_idx):
        loss, ypred, y = self.model_step(batch)
        self.val_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        logger.debug(f"Optimizer configured {optimizer.__class__}")
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            logger.debug(f"Scheduler configured {scheduler.__class__}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.hparams.lr_scheduler_kwargs,
                },
            }
        return {"optimizer": optimizer}
