from pathlib import Path

import hydra
from omegaconf import OmegaConf
from rich.console import Console
from pyrootutils import setup_root

from crescendo import utils, __version__

setup_root(__file__, indicator=".project-root", pythonpath=True)
console = Console()


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train.yaml"
)
@utils.log_warnings("Warnings were caught during training")
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

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out = hydra_cfg["runtime"]["output_dir"]
    console.log(f"Output dir: {out}")

    # Hydra magic, basically. Instantiate all relevant Lightning objects via
    # the hydra instantiator. Everything's under the hood in Crescendo's
    # utils module.
    datamodule, model, callbacks, loggers, trainer = utils.instantiate_all_(
        config, log=True
    )

    # Save the processed configuration file as yaml
    yaml_path = Path(out) / "final_config.yaml"
    utils.omegaconf_to_yaml(config, yaml_path)
    console.log(f"Final config saved to {yaml_path}")

    console.log(model)
    console.log(OmegaConf.to_container(config))

    # This a PyTorch 2.0 special. Compiles the model if possible for faster
    # runtime (if specified to try in the config). Will fail gracefully and
    # fall back on eager execution otherwise.
    model = utils.compile_model(config, model)

    # Fit the model, of course!
    console.log(f"checkpoint path is {config.get('ckpt_path')}")
    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=config.get("ckpt_path")
    )

    # Evaluate on the validation set. This will be important for hyperparameter
    # tuning later. Requires that the .validate method is defined on the
    # model.
    best_ckpt = trainer.checkpoint_callback.best_model_path
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=best_ckpt)
    val_metric = trainer.callback_metrics
    console.log(f"Validation metric: {val_metric}")

    return val_metric["val/loss"].item()


def entrypoint():
    console.log(f"Crescendo version=={__version__}", style="bold red")
    with utils.Timer() as dt:
        train()
    console.log(f"PROGRAM END ({str(int(dt()))} s)", style="bold red")
