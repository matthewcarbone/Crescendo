"""Controls making modifications to hydra configurations. Modifiers come with
a bold yellow output style."""

from os import cpu_count

import lightning as L
from rich.console import Console


CONSOLE = Console()
OUTPUT_STYLE = "bold yellow"


def _log(msg):
    CONSOLE.log(msg, style=OUTPUT_STYLE)


def seed_everything(config):
    """Runs the Lightning ``seed_everything`` method.
    
    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
    """

    if config.get("seed"):
        L.seed_everything(config.seed, workers=True)
        _log(f"Config seed set: {config.seed}")


def update_architecture_in_out_(config, datamodule):
    """Given the datamodule with properties ``n_features`` and ``n_targets``,
    this updates the config if the model is of type mlp. This method will only
    execute on models matching the path ``crescendo.models.mlp``.
    
    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration file which will be modified in-place.
    datamodule : lightning.LightningDataModule
        The LightningDataModule corresponding to the dataset in use.
    """

    if "crescendo.models.mlp" not in config.model["_target_"]:
        return
    
    n_features = datamodule.n_features
    if config.model["net"]["input_dims"] == "auto":
        config.model["net"]["input_dims"] = n_features
        _log(
            "Input MLP dimensions automatically set from dataloader "
            f"to n_features={n_features}"
        )

    n_targets = datamodule.n_targets
    if config.model["net"]["output_dims"] == "auto":
        config.model["net"]["output_dims"] = n_targets
        _log(
            "Output MLP dimensions automatically set from dataloader "
            f"to n_targets={n_targets}"
        )


def compile_model(config, model):
    """Attempts to executes the pytorch 2.0 model compiling for further
    acceleration. Will fail gracefully."""

    if not config.compile:
        _log("Model compile==False")
        return model

    import torch
    from torch import _dynamo
    _dynamo.config.suppress_errors = True
    model = torch.compile(model)
    _log("Model compilation attempted... see logs above")
    return model
