from time import perf_counter

from rich.console import Console

# import dgl
from dgllife.model import MPNNPredictor
from lightning import LightningModule
import torch
from torchmetrics import MeanMetric

from crescendo.models.mlp import FeedForwardNeuralNetwork

console = Console()


def collating_function_graph_to_vector(batch):
    """Collates cd / graph-fixed length vector combination. Recall that
    in this case, each element of batch is a three vector containing the
    graph and the target"""

    # Each target is the same length, so we can use standard batching for
    # it.
    targets = torch.FloatTensor([xx[1] for xx in batch])

    # However, graphs are not of the same "length" (diagonally on the
    # adjacency matrix), so we need to be careful. Usually, dgl's batch
    # method would work just fine here, but for multi-gpu training, we
    # need to catch some subtleties, since the batch itself is split apart
    # equally onto many GPU's, but torch doesn't know how to properly split
    # a batch of graphs. So, we manually partition the graphs here, and
    # will batch the output of the collating function before training.
    # This is now just a list of graphs.
    graphs = [xx[0] for xx in batch]

    return (graphs, targets)


class MessagePassingNeuralNetwork(LightningModule):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        node_out_feats,  # hidden node size
        edge_hidden_feats,
        n_tasks,  # output of the MPNN
        num_step_message_passing,
        num_step_set2set,
        num_layer_set2set,
        architecture,
        output_dims,
        dropout,
        activation,
        batch_norm,
        last_batch_norm,
        criterion,
        optimizer,
        last_activation,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MPNNPredictor(
            node_in_feats=node_in_feats,
            edge_in_feats=edge_in_feats,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            n_tasks=n_tasks,
            num_step_message_passing=num_step_message_passing,
            num_step_set2set=num_step_set2set,
            num_layer_set2set=num_layer_set2set,
        )
        self.net2 = FeedForwardNeuralNetwork(
            input_dims=n_tasks,
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

    def forward(self, g, n, e):
        return self.net2(self.net(g, n, e))

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

        (g, n, e, target) = batch
        ypred = self.forward(g, n, e)
        loss = self.criterion(ypred, target)
        return loss, ypred, target

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

        if not torch.isnan(avg_loss):
            console.log(
                f"{self.current_epoch} \t {dt:.02f} s \t T={avg_loss:.02e} "
                f"\t V={avg_val_loss:.02e} \t lr={lr:.02e}"
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
        console.log(f"Optimizer configured {optimizer.__class__}")
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            console.log(f"Scheduler configured {scheduler.__class__}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
