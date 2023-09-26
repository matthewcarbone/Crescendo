"""Controls making modifications to hydra configurations. Modifiers come with
a bold yellow output style."""


import lightning as L
import numpy as np
from omegaconf import DictConfig, ListConfig, open_dict

from crescendo import logger


LOGGER_PREFIX = "<modifier>"


def seed_everything(config):
    """Runs the Lightning ``seed_everything`` method.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
    """

    if config["seed"] > -1:
        L.seed_everything(config.seed, workers=True)
        logger.success(f"{LOGGER_PREFIX} Config seed set: {config.seed}")


def _update_architecture_linear_ramp_(config, input_dims_key, output_dims_key):
    """If config.model["architecture"] is an integer, assumes a linear ramp
    between the input and output layers, with the integer providing the number
    of hidden layers."""

    # These will have been set beforehand even if previously unset
    n_in = config.model[input_dims_key]
    n_out = config.model[output_dims_key]

    # Make a random choice of the number of interior neurons
    n_interior = config.model["architecture"]

    # Now we interpolate
    x = [0, n_interior + 1]
    y = [n_in, n_out]
    x_eval = np.linspace(1, n_interior, n_interior)
    y_interp = np.interp(x_eval, x, y).astype(int).tolist()

    config.model["architecture"] = y_interp
    logger.success(
        f"{LOGGER_PREFIX} Architecture set to {y_interp} "
        f"given n_layers=={n_interior}"
    )


def _update_architecture_(config, input_dims_key, output_dims_key):
    """Helper function for parsing through the ``architecture`` argument. This
    can either be a list (the actual interior layers) or a dictionary
    containing parameters for random initialization. The function returns the
    new interior layers of the network.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration file which will be modified in-place.
    input_dims_key : str
        The name of the input_dimensions to the neural network for which the
        architecture is being updated. For example, for a standard MLP, this
        just "input_dims", but for a GNN, in which the neural network is
        actually after the GNN, this is "n_tasks".
    output_dims_key : str
        Similar to input_dims_key but for the output layer. Usually this is
        just "output_dims".

    Raises
    ------
    ValueError
        If various errors are present.
    """

    if isinstance(config.model["architecture"], int):
        _update_architecture_linear_ramp_(
            config, input_dims_key, output_dims_key
        )
        return

    if isinstance(config.model["architecture"], ListConfig):
        return

    if not isinstance(config.model["architecture"], DictConfig):
        t = type(config.model["architecture"])
        raise ValueError(
            f"Config's MLP architecture must be list or dict. Found type {t}"
        )

    if config.get("seed") is None:
        raise ValueError(
            "For reproducibility, seed must be set for the random "
            "architecture initialization"
        )

    # Otherwise, we do our magic. Note that the random state should've been
    # provided earlier (via ``seed_everything``)
    neurons_range = config.model["architecture"]["neurons_range"]
    ramp_std = config.model["architecture"]["ramp_std"]

    # These will have been set beforehand even if previously unset
    n_in = config.model[input_dims_key]
    n_out = config.model[output_dims_key]

    # Make a random choice of the number of interior neurons
    n_interior = np.random.randint(
        low=neurons_range[0], high=neurons_range[1] + 1
    )

    # Now we interpolate
    x = [0, n_interior + 1]
    y = [n_in, n_out]
    x_eval = np.linspace(1, n_interior, n_interior)
    y_interp = np.interp(x_eval, x, y)
    logger.success(
        f"{LOGGER_PREFIX} Using randomized architecture: from parameters "
        f"neurons_range={neurons_range} and ramp_std={ramp_std:.05f}. "
        f"Interior architecture before noise: {y_interp.tolist()}"
    )

    # Add random noise
    y_interp = y_interp + np.random.normal(scale=ramp_std, size=len(y_interp))
    y_interp = y_interp.astype(int)
    y_interp[y_interp <= 0] = 1
    logger.success(f"{LOGGER_PREFIX} Architecture after noise: {y_interp}")

    config.model["architecture"] = y_interp.tolist()


def update_architecture_in_out_(config, datamodule):
    """Given the datamodule with properties ``n_features`` and ``n_targets``,
    this updates the config if the model is of type mlp. Basically, if the
    ``input_dims`` and ``output_dims`` are set to "auto", this function will
    populate those values based on the dataloader. In addition,
    ``architecture`` can be a list (corresponding to the dimensions of the
    hidden layers themselves), or a dictionary. If this is a list, it is just
    the architecture itself. If it is a dictionary, the first required key is
    ``neurons_range``, which is a list of two integers (the low and high,
    inclusive) values for the number of interior layers, chosen in a uniform
    random way. The second key is ``ramp_std``. This is a standard deviation of
    a Guassian distribution which represents the spread of random noise around
    a linear interpolant between the number of input and output features. This
    is a way to generate some randomness in the architecture while still
    preserving the funnel-like structure of the network.

    Note
    ----
    This method will only execute on models matching the path
    ``crescendo.models.mlp`` and ``crescendo.models.gnn``

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration file which will be modified in-place.
    datamodule : lightning.LightningDataModule
        The LightningDataModule corresponding to the dataset in use.
    """

    if "crescendo.models.mlp" in config.model["_target_"]:
        n_features = datamodule.n_features
        if config.model["input_dims"] == "auto":
            config.model["input_dims"] = n_features
            logger.success(
                f"{LOGGER_PREFIX} Input MLP dimensions automatically set "
                f"from dataloader to n_features={n_features}"
            )

        n_targets = datamodule.n_targets
        if config.model["output_dims"] == "auto":
            config.model["output_dims"] = n_targets
            logger.success(
                f"{LOGGER_PREFIX} Output MLP dimensions automatically set "
                f"from dataloader to n_targets={n_targets}"
            )

        _update_architecture_(config, "input_dims", "output_dims")

    elif "crescendo.models.gnn" in config.model["_target_"]:
        if config.model["output_dims"] == "auto":
            n_targets = datamodule.n_targets
            config.model["output_dims"] = n_targets
            logger.success(
                f"{LOGGER_PREFIX} Output GNN dimensions automatically set "
                f"from dataloader to n_targets={n_targets}"
            )

        if config.model["node_in_feats"] == "auto":
            node_in_feats = datamodule.node_in_feats
            config.model["node_in_feats"] = node_in_feats
            logger.success(
                f"{LOGGER_PREFIX} GNN node size automatically set from "
                f"dataloader to node_in_feats={node_in_feats}"
            )

        if config.model["edge_in_feats"] == "auto":
            edge_in_feats = datamodule.edge_in_feats
            config.model["edge_in_feats"] = edge_in_feats
            logger.success(
                f"{LOGGER_PREFIX} GNN edge size automatically set from "
                f"dataloader to edge_in_feats={edge_in_feats}"
            )

        _update_architecture_(config, "n_tasks", "output_dims")


def update_optimizer_lr_(config):
    """This function checks the lr parameter of the optimizer, but also
    checks against another possible parameter, log10_lr. If one is provided,
    it is set at the lr, with appropriate conversions. If neither is provided
    or both are provided, an error is thrown.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration file which will be modified in-place.
    """

    lr = config.model["optimizer"].get("lr", None)
    log10_lr = config.model["optimizer"].get("log10_lr", None)

    if lr is not None and log10_lr is not None:
        raise ValueError("You must only provide one of lr and log10_lr")
    if lr is None and log10_lr is None:
        raise ValueError("Either lr or log10_lr must be provided")

    if lr is None:
        # Then log10_lr is set
        with open_dict(config):
            log10_lr = config.model["optimizer"].pop("log10_lr")
            new_lr = 10.0**log10_lr
            config.model["optimizer"]["lr"] = new_lr
        logger.success(
            f"{LOGGER_PREFIX} Optimizer lr set from log10_lr to {new_lr:.02e}"
        )
    else:
        # The lr is set and log10_lr doesn't exist. Do nothing
        pass


def update_scheduler_based_on_production_mode_(config):
    """If Crescendo is being run in production mode, this function changes
    early stopping criteria's monitor to the training loss, and also
    changes the scheduler's lr modifier to the training loss.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        The configuration file which will be modified in-place.
    """

    if not config.data["production_mode"]:
        return

    if "early_stopping" in config.callbacks.keys():
        config.callbacks["early_stopping"]["monitor"] = "train/loss"
        logger.success(
            f"{LOGGER_PREFIX} Early stopping monitor set to train/loss"
        )

    config.model["lr_scheduler_kwargs"]["monitor"] = "train/loss"
    logger.success(f"{LOGGER_PREFIX} lr scheduler monitor set to train/loss")


def compile_model(config, model):
    """Attempts to executes the pytorch 2.0 model compiling for further
    acceleration. Will fail gracefully."""

    if not config.compile:
        logger.warning(
            f"{LOGGER_PREFIX} PyTorch 2.0 model compile setting is False"
        )
        return model

    import torch
    from torch import _dynamo

    _dynamo.config.suppress_errors = True
    model = torch.compile(model)
    logger.warning(
        f"{LOGGER_PREFIX} Model compilation attempted... see logs above"
    )
    return model
