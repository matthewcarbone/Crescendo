from pathlib import Path
import warnings

import hydra
from omegaconf import OmegaConf
from rich.pretty import pprint
from pyrootutils import setup_root

from crescendo import utils, logger, __version__
from crescendo.logger import configure_loggers, NO_DEBUG_LEVELS

setup_root(__file__, indicator=".project-root", pythonpath=True)

IGNORE_WARNINGS = (
    "is an instance of `nn.Module` and is already saved during "
    "checkpointing"
)


WARNINGS_ATTR = [
    "category",
    "file",
    "filename",
    "line",
    "lineno",
    "message",
    "source",
]


def _train(config):
    if config.logging_mode == "debug":
        configure_loggers()
    else:
        configure_loggers(levels=NO_DEBUG_LEVELS)
    logger.info(f"v{__version__}")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out = hydra_cfg["runtime"]["output_dir"]
    logger.info(f"Output dir: {out}")

    # Hydra magic, basically. Instantiate all relevant Lightning objects via
    # the hydra instantiator. Everything's under the hood in Crescendo's
    # utils module.
    # dm is datamodule
    dm, model, callbacks, loggers, trainer = utils.instantiate_all_(config)

    # Save the processed configuration file as yaml
    yaml_path = Path(out) / "final_config.yaml"
    utils.omegaconf_to_yaml(config, yaml_path)
    logger.info(f"Final config saved to {yaml_path}")

    if config.logging_mode == "debug":
        logger.debug("OmegaConf config:")
        pprint(OmegaConf.to_container(config))

    # This a PyTorch 2.0 special. Compiles the model if possible for faster
    # runtime (if specified to try in the config). Will fail gracefully and
    # fall back on eager execution otherwise.
    model = utils.compile_model(config, model)

    # Fit the model, of course!
    logger.info(">>>>>> Training start")
    trainer.fit(model=model, datamodule=dm, ckpt_path=config.get("ckpt_path"))
    logger.success("<<<<<< Training success")

    # Evaluate on the validation set. This will be important for hyperparameter
    # tuning later. Requires that the .validate method is defined on the
    # model.
    best_ckpt = trainer.checkpoint_callback.best_model_path
    trainer.validate(model=model, datamodule=dm, ckpt_path=best_ckpt)
    val_metric = trainer.callback_metrics

    return val_metric["val/loss"].item()


def _log_warnings(warnings_caught, config):
    if warnings_caught:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        out = hydra_cfg["runtime"]["output_dir"]
        warnings_path = Path(out) / "warnings.yaml"
        logger.warning(f"Warnings were caught and saved to {warnings_path}")
        all_warnings = [
            {
                attribute: str(getattr(w, attribute))
                for attribute in WARNINGS_ATTR
            }
            for w in warnings_caught
            if IGNORE_WARNINGS not in str(w)
        ]
        if config.logging_mode == "debug":
            logger.debug("Warnings below")
            pprint(all_warnings)
        utils.save_yaml(all_warnings, warnings_path)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train.yaml"
)
def train(config):
    """Executes training powered by Hydra, given the configuration file. Note
    that Hydra handles setting up the config.

    Parameters
    ----------
    config : omegaconf.DictConfig

    Returns
    -------
    float
        Validation metrics on the best checkpoint.
    """

    with warnings.catch_warnings(record=True) as warnings_caught:
        _train(config)
    _log_warnings(warnings_caught, config)


def entrypoint():
    with utils.Timer() as dt:
        train()
    logger.info(f"PROGRAM END ({str(int(dt()))} s)")
