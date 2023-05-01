#!/bin/bash

for ii in {0..19}
do
    cr \
        model=mlp_random_architecture \
        data=regression \
        seed="$ii" \
        +data.ensemble_split_index="$ii" \
        +data.data_dir=../data/C-XANES \
        trainer.max_epochs=10 \
        model.last_activation="{_target_: torch.nn.Sigmoid}" \
        trainer.accelerator=gpu \
        "model.architecture.neurons_range=[2,5]" \
        "model.architecture.ramp_std=1.0" \
        data.dataloader_kwargs.batch_size=32 \
        model.batch_norm=True \
        model.optimizer.lr=0.002 \
        model.dropout=0.0
done
