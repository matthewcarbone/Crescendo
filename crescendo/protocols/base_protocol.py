#!/usr/bin/env python3

import os
import time
import shutil
import uuid

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from crescendo import defaults
from crescendo.utils.logger import logger_default as log
from crescendo.utils import ml_utils
from crescendo.utils.py_utils import check_for_environment_variable

# At the time that the module is called, this should be a global variable
CUDA_AVAIL = torch.cuda.is_available()


class TrainProtocol:
    """Base class for performing ML training. Attributes that are not also
    inputs to __init__ are listed below. A standard use case for this type of
    class might look like this:

    Example
    -------
    > x = TrainProtocol(train_loader, valid_loader)
    > x.initialize_model(model_params)
    > x.initialize_support()  # inits the optimizer, loss and scheduler
    > x.train()

    Attributes
    ----------
    trainLoader : torch.utils.data.dataloader.DataLoader
    validLoader : torch.utils.data.dataloader.DataLoader
    device : str
        Will be 'cuda:0' if at least one GPU is available, else 'cpu'.
    parallel : bool
    model : torch.nn.Module
        Class inheriting the torch.nn.Module back end.
    self.criterion : torch.nn._Loss
        Note, I think this is the correct object type. The criterion is the
        loss function.
    self.optimizer : torch.optim.Optimizer
        The numerical optimization protocol. Usually, we should choose Adam for
        this.
    scheduler : torch.optim.lr_scheduler
        Defines a protocol for training, usually for updating the learning
        rate.
    best_model_state_dict : Dict
        The torch model state dictionary corresponding to the best validation
        result. Used as a lightweight way to store the model parameters.
    """

    def __init__(
        self, dsname, trainLoader, validLoader, directory=None, seed=None,
        parallel=True, trial=str(uuid.uuid4()), override=False
    ):
        """Initializer.

        Parameters
        ----------
        dsname : str
            The dataset name to which this training corresponds to.
        trainLoader : torch.utils.data.dataloader.DataLoader
            The training loader object.
        validLoader : torch.utils.data.dataloader.DataLoader
            The cross validation (or testing) loader object.
        directory : str, optional
            The cache directory for the dataset. The trials will be saved here.
            Default is None, and it will look for an environment variable if
            this is the case.
        seed : int, optional
            Seeds random, numpy and torch.
        parallel : bool
            Switch from parallel GPU training to single, if desired. Ignored if
            no GPUs are available. Default is True
        trial : str, optional
            The specific trial identifier. Useful for e.g. hyperparameter
            tuning. Defaults to a uuid random hash.
        override : bool
            If True, this will restart training by deleting any existing
            directory corresponding to the trial, and restarting training from
            the first epoch. If this is False, an attept is made to see if a
            previously-trained model exists, and training will continue from
            state. Else, if no model exists, training will begin at the first
            epoch.
        """

        self.trainLoader = trainLoader
        self.validLoader = validLoader
        self.device = torch.device('cuda:0' if CUDA_AVAIL else 'cpu')
        self.parallel = parallel and CUDA_AVAIL
        if self.parallel and not torch.cuda.device_count() > 1:
            self.paralel = False
        self._log_cuda_info()
        if seed is not None:
            ml_utils.seed_all(seed)

        self.dsname = dsname
        self.trial = trial

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_model_state_dict = None
        self.crit_str = None

        if directory is None:
            directory = check_for_environment_variable(defaults.QM9_DS_ENV_VAR)

        # Get the location of the directory for this particular trial.
        self.root = f"{directory}/{trial}"
        log.info(
            f"Root directory for saving model and states set to {self.root}"
        )
        if os.path.isdir(self.root):

            log.info(f"Trial path {self.root} exists")

            if override:
                log.warning("Override is True, deleting existing data")
                shutil.rmtree(self.root)

            else:
                log.info(
                    "Override is False, will resume training from checkpoint"
                )

            os.makedirs(self.root, exist_ok=True)

    # The following two functions are based off of:
    # https://medium.com/analytics-vidhya/
    # saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
    def save_checkpoint(self, state):
        """We'll save a model checkpoint only when the validation loss is less
        than all previous ones.

        Parameters
        ----------
        state : dict
            State dictionary of the format
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'crit_str': self.crit_str
            }
        """

        f_path = f"{self.root}/checkpoint.pt"
        torch.save(state, f_path)

    def load_checkpoint(self):
        """Loads a model checkpoint from state. Initializes the model,
        optimizer, schedule and criterion.

        Returns
        -------
        int
            The epoch to restart on.
        """

        checkpoint = torch.load(f"{self.root}/checkpoint.pt")
        log.info(f"Loading model from state at {checkpoint}")

        self.model.load_state_dict(checkpoint['state_dict'])
        log.info(f"Model initialization successful")

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        log.info(f"Optimizer initialization successful")

        self.scheduler.load_state_dict(checkpoint['scheduler'])
        log.info(f"Scheduler initialization successful")

        self.crit_str = checkpoint['crit_str']
        self._init_criterion(self.crit_str)
        log.info(f"Criterion initialization successful")

        epoch = checkpoint['epoch']
        log.info(f"Overall initialization successful at epoch {epoch}")

        return epoch

    def initialize_model(self):
        raise NotImplementedError

    def _eval_valid_pass(self):
        raise NotImplementedError

    def _train_single_epoch(self):
        raise NotImplementedError

    def _send_model_to_device(self):
        """Initializes the parallel model if specified, and sends the model to
        the used device."""

        if self.parallel:
            self.model = nn.DataParallel(self.model)
            log.info("Parallel model defined")
        self.model.to(self.device)
        log.info(f"Model sent to {self.device}")

    def _log_model_info(self):
        """Computes the number of trainable parameters and logs it."""

        n_trainable = sum(
            param.numel() for param in self.model.parameters()
            if param.requires_grad
        )
        log.info(f"model has {n_trainable} trainable parameters")

    def _log_cuda_info(self):
        """Informs the user about the available GPU status."""

        log.info(
            f"device is {self.device}; CUDA_AVAIL {CUDA_AVAIL}; "
            f"parallel {self.parallel}"
        )
        if CUDA_AVAIL:
            gpu_count = torch.cuda.device_count()
            log.info(f"number of GPUs is {gpu_count}")

    def _init_optimizer(self, opt_str, kwargs):
        """Initializes the optimizer.

        Parameters
        ----------
        opt_str : {'adam'}
            String indicating the optimizer.
        kwargs : Dict[str, Any]
            The parameters to be passed to the optimizer by keyword.

        Returns
        -------
        torch.optim.Optimizer
        """

        if opt_str == 'adam':
            self.optimizer = \
                torch.optim.Adam(self.model.parameters(), **kwargs)
            log.info("Initialized the Adam optimizer")
        else:
            critical = \
                "Invalid optimizer specified. Re-initialize the model using " \
                "known optimizer."
            log.critical(critical)
            raise RuntimeError(critical)

    def _init_criterion(self, crit_str):
        """Initializes the criterion.

        Parameters
        ----------
        crit_str : {'l1', 'l2'}
            String indicating the criterion.

        Returns
        -------
        torch.nn._Loss
        """

        self.crit_str = crit_str

        if self.crit_str == 'l1':
            self.criterion = nn.L1Loss()
            log.info("Initialized the L1 (MAE) loss")
        elif self.crit_str == 'l2':
            self.criterion = nn.MSELoss()
            log.info("Initialized the L2 (MSE) loss")
        else:
            critical = \
                "Invalid criterion specified. Re-initialize the model using " \
                "known criterion."
            log.critical(critical)
            raise RuntimeError(critical)

    def _init_scheduler(self, sched_string, kwargs):
        """Initializes the scheduler.

        Parameters
        ----------
        sched_string : {'rlrp'}
            String indexing the scheduler. Default is 'rlrp', for 'reduce
            learning rate on plateau'.
        kwargs : Dict[str, Any]
            The parameters to be passed to the optimizer by keyword.
        """

        if sched_string == 'rlrp':
            self.scheduler = ReduceLROnPlateau(self.optimizer, **kwargs)
            log.info('Initialized the ReduceLROnPlateau scheduler')
        else:
            critical = \
                "Invalid scheduler specified. Re-initialize the model using " \
                "known scheduler."
            log.critical(critical)
            raise RuntimeError(critical)

    def initialize_support(
        self, criterion=('l1', dict()), optimizer=('adam', {'lr': 1e-4}),
        scheduler=('rlrp', {'patience': 10, 'factor': 0.02, 'min_lr': 1e-6})
    ):
        """Initializes the criterion, optimize and scheduler.

        Parameters
        ----------
        criterion : Tuple[str, Dict[str, Any]]
            String (and kwargs) indexing which criterion to load (and its
            parameters). Default is l2.
        optimizer : Tuple[str, Dict[str, Any]]
            Tuple containing the initialization string for the optimizer, and
            optional dictionary containing the kwargs to pass to the optimizer
            initializer. Default is adam.
        scheduler : Tuple[str, Dict[str, Any]]
            Save as the optimizer, but for the scheduler. Default is to reduce
            learning rate upon plateau.
        """

        self._init_criterion(*criterion)
        self._init_optimizer(*optimizer)
        self._init_scheduler(*scheduler)

    def _eval_valid(self):
        """Similar to _train_single_epoch above, this method will evaluate a
        full pass on the validation data.

        Returns
        -------
        float
            The average loss on the validation data / batch.
        """

        self.model.eval()  # Freeze weights, set model to evaluation mode

        # Double check that torch will not be updating gradients or doing
        # anything we don't want it to do.
        with torch.no_grad():
            total_loss, __ = self._eval_valid_pass(self.validLoader)

        return total_loss

    def _update_state_dict(self, best_valid_loss, valid_loss, epoch):
        """Updates the best_model_state_dict attribute if the valid loss is
        less than the best up-until-now valid loss.

        Parameters
        ----------
        best_valid_loss : float
            The best validation loss so far.
        valid_loss : float
            The current validation loss on the provided epoch.
        epoch : int
            The current epoch.

        Returns
        -------
        float
            min(best_valid_loss, valid_loss)
        """

        if valid_loss < best_valid_loss or epoch == 0:
            self.best_model_state_dict = self.model.state_dict()
            log.info(
                f'\tVal. Loss: {valid_loss:.05e} < Best Val. Loss '
                f'{best_valid_loss:.05e}'
            )
            log.info("\tUpdating best_model_state_dict")
        else:
            log.info(f'\tVal. Loss: {valid_loss:.05e}')

        return min(best_valid_loss, valid_loss)

    def _step_scheduler(self, valid_loss):
        """Steps the scheduler and outputs information about the learning
        rate.

        Parameters
        ----------
        valid_loss : float
            The current epoch's validation loss.

        Returns
        -------
        float
            The current learning rate.
        """

        # Current learning rate
        clr = self.scheduler.optimizer.param_groups[0]['lr']
        log.info(f"\t Learning rate {clr:.02e}")
        self.scheduler.step(valid_loss)
        return clr

    def train(self, epochs, clip=None):
        """Executes model training.

        Parameters
        ----------
        epochs : int
            Number of full passes through the training data.
        clip : float, optional
            Gradient clipping.
        """

        # Keep track of the best validation loss so that we know when to save
        # the model state dictionary.
        best_valid_loss = float('inf')

        # Begin training
        train_loss_list = []
        valid_loss_list = []
        learning_rates = []
        for epoch in range(epochs):

            # Train a single epoch
            t0 = time.time()
            train_loss = self._train_single_epoch(clip)
            t_total = time.time() - t0
            log.info(f"Epoch {epoch:04} [{t_total:3.02f}s]")
            log.info(f'\tTrain Loss: {train_loss:.05e}')

            # Evaluate on the validation data
            valid_loss = self._eval_valid()

            # Step the scheduler
            clr = self._step_scheduler(valid_loss)

            # Update the best state dictionary of the model for loading in
            # later on in the process
            best_valid_loss = self._update_state_dict(
                best_valid_loss, valid_loss, epoch
            )

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            learning_rates.append(clr)

        log.info("Setting model to best state dict")
        self.model.load_state_dict(self.best_model_state_dict)

        return train_loss_list, valid_loss_list, learning_rates

    def eval(self, loader_override=None, meta=None):
        """Systematically evaluates the validation, or dataset corresponding to
        the loader specified in the loader_override argument, dataset."""

        if loader_override is not None:
            log.warning(
                "Default validation loader is overridden - ensure this is "
                "intentional, as this is likely evaluating on a testing set"
            )

        loader = self.validLoader if loader_override is None \
            else loader_override

        # defaults.Result
        with torch.no_grad():
            total_loss, cache = self._eval_valid_pass(
                loader, cache=True, target_metadata=meta
            )
        log.info(f"Eval complete: loss {total_loss:.02e}")

        return cache
