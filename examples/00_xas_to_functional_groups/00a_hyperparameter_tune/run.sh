#!/bin/bash

cr \
    model=mlp \
    data=regression \
    +data.data_dir=../data/C-XANES \
    trainer.max_epochs=10 \
    model.last_activation="{_target_: torch.nn.Sigmoid}" \
    trainer.accelerator=cpu \
    hparams_search=optuna \
    'model.architecture=choice(3, 6, 9, 12, 15)' \
    'data.dataloader_kwargs.batch_size=choice(32, 64, 128, 256)' \
    'model.batch_norm=choice(True, False)' \
    'model.optimizer.lr=interval(0.001, 0.01)' \
    "hydra.sweeper.n_trials=20"

