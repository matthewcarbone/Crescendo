#!/usr/bin/env python3

import numpy as np
import torch

from crescendo.models.feedforward import StandardPerceptron
from crescendo.protocols.base_protocol import TrainProtocol
from crescendo.utils.logger import logger_default as log


class Vec2VecProtocol(TrainProtocol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_model(self, model_type='mlp', best=False, **kwargs):
        """Initializes the standard MLP neural network.

        Parameters
        ----------
        best : bool
            If True, initializes the model from the best model parameters as
            evaluated on the validation set.
        """

        if model_type == 'mlp':
            self.model = StandardPerceptron(**kwargs)
        else:
            raise NotImplementedError

        self._send_model_to_device()
        self._log_model_info()

        if self.checkpoint is not None:
            m = 'best_model' if best else 'model'
            self.model.load_state_dict(self.checkpoint[m])
            self.best_model_state_dict = self.checkpoint[m]
            log.info("Model initialization from checkpoint successful")

    def _train_single_epoch(self, clip=None):
        """Executes the training of the model over a single full pass of
        training data.

        Parameters
        ----------
        clip : float, optional
            Gradient clipping.

        Returns
        -------
        float
            The average training loss/batch.
        """

        self.model.train()  # Unfreeze weights, set model in train mode
        epoch_loss = []
        for idx, batch in enumerate(self.trainLoader):

            (f, t, idx, meta) = batch
            f = f.float().to(self.device)
            t = t.float().to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Run forward prop
            output = self.model.forward(f)

            # Compute the loss
            loss = self.criterion(output.flatten(), t.flatten())

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

            (f, t, idx, meta) = batch
            f = f.float().to(self.device)
            t = t.float().to(self.device)

            output = self.model.forward(f)

            # Note that the batch size is target.shape[0]
            if cache:
                cache_list_batch = [
                    (
                        np.array(idx[ii]),
                        output[ii].cpu().detach().numpy() * sd + mu,
                        t[ii].cpu().detach().numpy() * sd + mu,
                        np.array(meta[ii])
                    ) for ii in range(t.shape[0])
                ]
                cache_list.extend(cache_list_batch)
            loss = self.criterion(output.flatten(), t.flatten())
            total_loss.append(loss.item())

        return sum(total_loss) / len(total_loss), cache_list
