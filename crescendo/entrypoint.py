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

    utils.seed_everything(config)
    datamodule = utils.instantiate_datamodule(config)
    utils.update_architecture_in_out_(config, datamodule)
    model = utils.instantiate_model(config)
    callbacks = utils.instantiate_callbacks(config)
    loggers = utils.instantiate_loggers(config)
    trainer = utils.instantiate_trainer(config, callbacks, loggers)

    console.log(log_locals=True)

    model = utils.compile_model(config, model)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.get("ckpt_path")
    )


def entrypoint():
    train()
    console.log("PROGRAM END", style="bold red")
