#!/usr/bin/env python3

import numpy as np
import torch

from crescendo.models.mpnn import MPNN
from crescendo.protocols.base_protocol import TrainProtocol


class GraphToVectorProtocol(TrainProtocol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_model(self, **kwargs):
        self.model = MPNN(**kwargs)
        self._send_model_to_device()
        self._log_model_info()

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

            # Features
            g = batch[0].to(self.device)
            n = batch[0].ndata['features'].to(self.device)
            e = batch[0].edata['features'].to(self.device)
            target = batch[1].to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Run forward prop
            output = self.model.forward(g, n, e)

            # Compute the loss
            loss = self.criterion(output.flatten(), target.flatten())

            # Run back prop
            loss.backward()

            # Clip the gradients
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            # Step the optimizer
            self.optimizer.step()

            # Append the current loss tracker
            epoch_loss.append(loss.item())

        return sum(epoch_loss) / len(epoch_loss)  # mean loss over this epoch

    def _eval_valid_pass(self, loader, cache=False):
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

        for idx, batch in enumerate(loader):
            # Features
            g = batch[0].to(self.device)
            n = batch[0].ndata['features'].to(self.device)
            e = batch[0].edata['features'].to(self.device)
            target = batch[1].to(self.device)
            ids = batch[2]
            output = self.model.forward(g, n, e)

            if cache:
                cache_list_batch = [
                    (
                        np.array(ids[ii]), output[ii].numpy(),
                        target[ii].numpy()
                    ) for ii in range(len(batch))
                ]
                cache_list.extend(cache_list_batch)
            loss = self.criterion(output.flatten(), target.flatten())
            total_loss.append(loss.item())

        cache_list.sort(key=lambda x: x[4])
        return sum(total_loss) / len(total_loss), cache_list
