#!/bin/bash

cr \
    model=gnn \
    data=graph_to_vector \
    +data.data_dir=./data \
    trainer.max_epochs=10 \
    model.last_activation="{_target_: torch.nn.Sigmoid}" \
    trainer.accelerator=cpu \
    data.dataloader_kwargs.batch_size=2 \
    hparams_search=optuna \
    'model.architecture=choice(3, 6, 9, 12, 15)' \
    'model.batch_norm=choice(True, False)' \
    'model.optimizer.lr=interval(0.001, 0.01)' \
    "hydra.sweeper.n_trials=20"
