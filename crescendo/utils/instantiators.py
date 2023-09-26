"""Container for various LightningModule models
Code is modified based off of
https://github.com/ashleve/lightning-hydra-template/blob/
89194063e1a3603cfd1adafa777567bc98da2368/src/utils/instantiators.py

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

import hydra
from omegaconf import DictConfig
import torch

from crescendo.utils.modifiers import (
    seed_everything,
    update_architecture_in_out_,
    update_optimizer_lr_,
    update_scheduler_based_on_production_mode_,
)
from crescendo import logger


def instantiate_datamodule(config):
    datamodule = hydra.utils.instantiate(config.data)
    logger.debug(f"Datamodule instantiated {datamodule.__class__}")
    return datamodule


def instantiate_model(config, checkpoint=None):
    model = hydra.utils.instantiate(config.model)
    logger.debug(f"Model instantiated {model.__class__}")
    if checkpoint is not None:
        try:
            model = model.__class__.load_from_checkpoint(checkpoint)
        except RuntimeError:
            model = model.__class__.load_from_checkpoint(
                checkpoint, map_location=torch.device("cpu")
            )
        logger.debug(f"Model loaded from checkpoint {checkpoint}")
    return model


def instantiate_trainer(config, callbacks, loggers):
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers
    )
    logger.debug(f"Trainer instantiated {trainer.__class__}")
    return trainer


def instantiate_callbacks(config):
    """Instantiates callbacks from config."""

    callbacks_cfg = config.get("callbacks")

    callbacks = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    for callback in callbacks:
        logger.debug(f"Callbacks instantiated {callback.__class__}")
    del callback  # Remove from locals

    return callbacks


def instantiate_loggers(config):
    """Instantiates loggers from config."""

    logger_cfg = config.get("logger")

    _logger = []

    if not logger_cfg:
        return _logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            _logger.append(hydra.utils.instantiate(lg_conf))

    for ll in _logger:
        logger.debug(f"Logger instantiated {ll.__class__}")
    del ll

    return _logger


def instantiate_all_(config):
    """Core utility which instantiates the datamodule, model, callbacks,
    loggers and trainer from the hydra config. This makes modifications to the
    config where appropriate."""

    seed_everything(config)
    datamodule = instantiate_datamodule(config)
    update_architecture_in_out_(config, datamodule)
    update_optimizer_lr_(config)
    model = instantiate_model(config)
    callbacks = instantiate_callbacks(config)
    update_scheduler_based_on_production_mode_(config)
    loggers = instantiate_loggers(config)
    trainer = instantiate_trainer(config, callbacks, loggers)

    return datamodule, model, callbacks, loggers, trainer
