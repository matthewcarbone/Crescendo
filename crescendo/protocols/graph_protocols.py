#!/usr/bin/env python3

import warnings

import dgl
from dgllife.model import MPNNPredictor
import numpy as np
import torch

from crescendo.protocols.base_protocol import TrainProtocol
from crescendo.utils.logger import logger_default as log


class GraphToVectorProtocol(TrainProtocol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_MPNN(
        self, n_node_features, n_edge_features, output_size,
        hidden_node_size=64, hidden_edge_size=128
    ):
        """TODO: docstring."""

        self.model = MPNNPredictor(
            node_in_feats=n_node_features, edge_in_feats=n_edge_features,
            node_out_feats=hidden_node_size,
            edge_hidden_feats=hidden_edge_size, n_tasks=output_size
        )
        self._send_model_to_device()
        self._log_model_info()

    def initialize_model(self, model_name='MPNN', **kwargs):
        if model_name == 'MPNN':
            self.initialize_MPNN(**kwargs)
        else:
            raise NotImplementedError

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model'])
            self.best_model_state_dict = self.checkpoint['model']
            log.info(f"Model initialization from checkpoint successful")

    def _get_batch(self, batch):
        """Parses a batch from the Loaders to the model-compatible features."""

        dgl_batched = dgl.batch(batch[0])
        g = dgl_batched.to(self.device)
        n = dgl_batched.ndata['features'].float().to(self.device)
        e = dgl_batched.edata['features'].float().to(self.device)
        target = batch[1].to(self.device)
        ids = batch[2]

        return (g, n, e, target, ids)

    def _train_single_epoch(self, clip=None):
        """Executes the training of the model over a single full pass of
        training data.

        Parameters
        ----------
        clip : float
            Gradient clipping.

        Returns
        -------
        float
            The average training loss/batch.
        """

        self.model.train()  # Unfreeze weights, set model in train mode
        epoch_loss = []
        for idx, batch in enumerate(self.trainLoader):

            # Recall that batch[0] is a LIST of graphs now, so we need to batch
            # it properly.
            (g, n, e, target, ids) = self._get_batch(batch)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Run forward prop
            output = self.model.forward(g, n, e)

            # Compute the loss
            loss = self.criterion(output.flatten(), target.flatten())

            # Run back prop
            with warnings.catch_warnings():

                # Ignore a silly warning that I can't turn off in pytorch
                # See https://github.com/pytorch/pytorch/issues/43425
                warnings.filterwarnings("ignore", category=UserWarning)
                loss.backward()

            # Clip the gradients
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            # Step the optimizer
            self.optimizer.step()

            # Append the current loss tracker
            epoch_loss.append(loss.item())

        return sum(epoch_loss) / len(epoch_loss)  # mean loss over this epoch

    def _eval_valid_pass(self, loader, cache=False, target_metadata=None):
        """Performs the for loop in evaluating the validation sets. This allows
        for interchanging the passed loaders as desired.

        Parameters
        ----------
        loader : torch.utils.data.dataloader.DataLoader
            Input loader to evaluate on.
        cache : bool
            If true, will save every output on the full pass to a dictionary so
            the user can evaluate every result individually.

        Returns
        -------
        float, List[defaults.Result]
            The average loss on the validation data / batch. Also returns a
            cache of the individual evaluation results if cache is True.
        """

        total_loss = []
        cache_list = []

        if cache:
            mu = 0.0
            sd = 1.0
            if target_metadata is not None:
                mu = target_metadata[0]
                sd = target_metadata[1]

        for idx, batch in enumerate(loader):

            (g, n, e, target, ids) = self._get_batch(batch)
            output = self.model.forward(g, n, e)

            # Note that the batch size is target.shape[0]

            if cache:
                cache_list_batch = [
                    (
                        np.array(ids[ii]),
                        output[ii].cpu().detach().numpy() * sd + mu,
                        target[ii].cpu().detach().numpy() * sd + mu
                    ) for ii in range(target.shape[0])
                ]
                cache_list.extend(cache_list_batch)
            loss = self.criterion(output.flatten(), target.flatten())
            total_loss.append(loss.item())

        return sum(total_loss) / len(total_loss), cache_list
