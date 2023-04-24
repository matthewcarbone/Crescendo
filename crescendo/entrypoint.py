import hydra
from pyrootutils import setup_root


from rich.console import Console


from crescendo import utils

setup_root(__file__, indicator=".project-root", pythonpath=True)
console = Console()


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train.yaml"
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
    dict
        Validation metrics on the best checkpoint.
    """

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out = hydra_cfg['runtime']['output_dir']
    console.log(f"Output dir: {out}")

    # Hydra magic, basically. Instantiate all relevant Lightning objects via
    # the hydra instantiator. Everything's under the hood in Crescendo's
    # utils module.
    utils.seed_everything(config)
    datamodule = utils.instantiate_datamodule(config)
    utils.update_architecture_in_out_(config, datamodule)
    model = utils.instantiate_model(config)
    callbacks = utils.instantiate_callbacks(config)
    loggers = utils.instantiate_loggers(config)
    trainer = utils.instantiate_trainer(config, callbacks, loggers)

    console.log(log_locals=True)

    # This a PyTorch 2.0 special. Compiles the model if possible for faster
    # runtime (if specified to try in the config). Will fail gracefully and
    # fall back on eager execution otherwise.
    model = utils.compile_model(config, model)

    # Fit the model, of course!
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.get("ckpt_path")
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
    train()
    console.log("PROGRAM END", style="bold red")
