from datetime import datetime
from os import get_terminal_size
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import warnings

import hydra
from omegaconf import OmegaConf
from rich.pretty import pprint

# from pyrootutils import setup_root

from crescendo import utils, logger, __version__
from crescendo.logger import configure_loggers, NO_DEBUG_LEVELS

# setup_root(__file__, indicator=".project-root", pythonpath=True)

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


def _configure_loggers(config):
    """Configures the Crescendo logging system.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
    """

    if config.debug_mode:
        configure_loggers()
    else:
        configure_loggers(levels=NO_DEBUG_LEVELS)

    try:
        tsize = get_terminal_size().columns
    except OSError:
        tsize = 20
    msg = "NEW RUN"
    L = ((tsize - len(msg)) // 2) - 3
    arrows = ">" * L
    arrows_r = "<" * L
    logger.info(f"{arrows} {msg} {arrows_r}")

    if config.debug_mode:
        logger.info(f"Cresecendo v{__version__} running with debug mode on")
    else:
        logger.info(f"Cresecendo v{__version__}")


def _configure_cache(config):
    """Configures a cache which is available to any instance running train.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
    """

    # Get the resolved dictionary
    resolved = OmegaConf.to_container(config, resolve=True)

    # Pop everything that could pertain to a specific instance of train
    pop = ["data", "model", "callbacks", "logger", "trainer", "extras"]
    pop_paths = ["output_dir", "log_dir", "checkpoint_dir"]
    for key in pop:
        resolved.pop(key)
    for key in pop_paths:
        resolved["paths"].pop(key)

    cache = utils.GlobalCache(config.TEMPDIR_CACHE)
    d = cache.read()
    d["config"] = resolved
    logger.debug(f"Global cache created at {cache._path}")
    if config.debug_mode:
        logger.debug("Global cache:")
        pprint(resolved)

    d["now"] = str(datetime.now())
    cache.save(d)

    # Check loaded
    cache2 = utils.GlobalCache(config.TEMPDIR_CACHE)
    d = cache2.read()
    logger.debug(f"Now == {d['now']}; cache check pass")


def _save_final_config(config):
    """The modifiers and other utilities in Crescendo actually modify the
    config on the fly. This final configuration file is saved with the models
    and other metadata. Also prints out the final configuration if debug
    mode is on.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
        Description
    """

    # Save the processed configuration file as yaml
    out = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    yaml_path = Path(out) / "final_config.yaml"
    utils.omegaconf_to_yaml(config, yaml_path)
    logger.info(f"Final config saved to {yaml_path}")

    if config.debug_mode:
        logger.debug("OmegaConf config:")
        pprint(OmegaConf.to_container(config))


def _save_validation_score(config, val_metric):
    """Saves the validation score and model path to the global config.

    Parameters
    ----------
    config : omegaconf.dictconfig.DictConfig
    val_metric : float
    """

    cache = utils.GlobalCache(config.TEMPDIR_CACHE)
    out = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    d = cache.read()
    if "validation_results" not in d.keys():
        d["validation_results"] = {}
    d["validation_results"][out] = val_metric
    cache.save(d)


def _train(config):
    _configure_loggers(config)
    _configure_cache(config)

    # Hydra magic, basically. Instantiate all relevant Lightning objects via
    # the hydra instantiator. Everything's under the hood in Crescendo's
    # utils module. Note that "dm" is datamodule
    dm, model, callbacks, loggers, trainer = utils.instantiate_all_(config)

    _save_final_config(config)

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
    # best_ckpt = trainer.checkpoint_callback.best_model_path
    # trainer.validate(model=model, datamodule=dm, ckpt_path=best_ckpt)
    val_metric = trainer.callback_metrics

    _save_validation_score(config, val_metric["val/loss"].item())

    return {"val_metric": val_metric["val/loss"].item()}


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
        if config.debug_mode:
            logger.debug("Warnings below")
            pprint(all_warnings)
        utils.save_yaml(all_warnings, warnings_path)


@hydra.main(
    version_base="1.3", config_path="configs", config_name="train.yaml"
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
        state = _train(config)
    _log_warnings(warnings_caught, config)
    return state["val_metric"]


def _tune_cleanup(cache):
    """Optuna can perform the hyperparameter search but it doesn't actually
    tell the user which model is the best (for some reason). This actually
    does that."""

    d = cache.read()

    sweep_run_dir = d["config"]["paths"]["sweep_run_dir"]
    p = Path(sweep_run_dir) / "optimization_results.yaml"
    if not p.exists():
        return

    optimization_results = utils.read_yaml(p)

    res = [(key, value) for key, value in d["validation_results"].items()]
    res.sort(key=lambda xx: xx[1])

    best_result = res[0][0]  # Get the directory

    optimization_results["best_model"] = best_result

    utils.save_yaml(optimization_results, p)

    if d["config"]["debug_mode"]:
        logger.debug("Optimization results:")
        pprint(optimization_results)

    logger.success(f"HP tuning found best model path: {best_result}")


def entrypoint():
    with TemporaryDirectory() as tempdir:
        sys.argv.append(f"+TEMPDIR_CACHE={tempdir}")
        with utils.Timer() as dt:
            train()
        cache = utils.GlobalCache(tempdir)
        _tune_cleanup(cache)
    logger.info(f"PROGRAM END ({str(int(dt()))} s)")
