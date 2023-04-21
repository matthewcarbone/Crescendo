from contextlib import redirect_stderr
import hydra
from io import StringIO
from pyrootutils import setup_root
import os

import lightning as L
import torch

from crescendo import utils


setup_root(__file__, indicator=".project-root", pythonpath=True)
log = utils.get_pylogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train.yaml"
)
def main(config):
    # print("Working directory : {}".format(os.getcwd()))
    
    if config.get("seed"):
        L.seed_everything(config.seed, workers=True)
        log.warning(f"Config seed set: {config.seed}")

    datamodule = hydra.utils.instantiate(config.data)
    log.info(f"Datamodule instantiated {datamodule.__class__}")
    
    model = hydra.utils.instantiate(config.model)
    log.info(f"Model instantiated {model.__class__}")

    callbacks = utils.instantiate_callbacks(config.get("callbacks"))
    for callback in callbacks:
        log.info(f"Callbacks instantiated {callback.__class__}")

    logger = utils.instantiate_loggers(config.get("logger"))
    for _logger in logger:
        log.info(f"Logger instantiated {_logger.__class__}")

    f = StringIO()
    with redirect_stderr(f):
        trainer = hydra.utils.instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=logger
        )
    s = f.getvalue()
    log.info(f"Trainer instantiated {trainer.__class__}")
    if len(s) > 2:
        log.warning(s)

    # all_objects = {
    #     "cfg": config,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }

    # return

    # model = torch.compile(model)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.get("ckpt_path")
    )


def entrypoint():
    main()
