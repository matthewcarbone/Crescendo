#!/usr/bin/env python3

import torch

from crescendo.models.feedforward import StandardPerceptron
from crescendo.protocols.base_protocol import TrainProtocol
from crescendo.utils.logger import logger_default as log


class Vec2VecProtocol(TrainProtocol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_model(
        self, n_features, n_targets, n_meta, model_type='mlp', best=False,
        **kwargs
    ):
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

        self.n_features = n_features
        self.n_targets = n_targets
        self.n_meta = n_meta

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
        fstart = 1 + self.n_meta
        fend = 1 + self.n_meta + self.n_features
        for __, batch in enumerate(self.trainLoader):

            f = batch[0][:, fstart:fend].to(self.device)
            t = batch[0][:, fend:].to(self.device)

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

    def _eval_valid_pass(
        self, loader, cache=False, feature_metadata=None, target_metadata=None
    ):
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

        fstart = 1 + self.n_meta
        fend = 1 + self.n_meta + self.n_features

        if cache:
            mu_targ = 0.0
            sd_targ = 1.0
            if target_metadata is not None:
                mu_targ = target_metadata[0]
                sd_targ = target_metadata[1]
            mu_feat = 0.0
            sd_feat = 0.0
            if feature_metadata is not None:
                mu_feat = feature_metadata[0]
                sd_feat = feature_metadata[1]

        for __, batch in enumerate(loader):

            idx = batch[0][:, 0]
            meta = batch[0][:, 1:fstart]
            f = batch[0][:, fstart:fend].to(self.device)
            t = batch[0][:, fend:].to(self.device)

            output = self.model.forward(f)

            # Note that the batch size is target.shape[0]
            if cache:
                cache_list_batch = [
                    (
                        int(idx[ii]), meta[ii],
                        f.cpu().detach().numpy() * sd_feat + mu_feat,
                        output[ii].cpu().detach().numpy() * sd_targ + mu_targ,
                        t[ii].cpu().detach().numpy() * sd_targ + mu_targ,
                    ) for ii in range(t.shape[0])
                ]
                cache_list.extend(cache_list_batch)
            loss = self.criterion(output.flatten(), t.flatten())
            total_loss.append(loss.item())

        return sum(total_loss) / len(total_loss), cache_list
