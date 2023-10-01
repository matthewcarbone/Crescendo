# Hyperparameter tuning

In this example, we demonstrate how Crescendo can leverage the Hydra CLI to do straightforward hyperparamter optimization using Optuna.

The `run.sh` file looks like this:

```bash
cr \
    model=mlp \
    data=regression \
    +data.data_dir=../data/C-XANES \
    trainer.max_epochs=10 \
    model.last_activation="{_target_: torch.nn.Sigmoid}" \
    trainer.accelerator=cpu \
    hparams_search=optuna \
    'model.architecture=choice(3, 6, 9, 12, 15)' \
    'data.batch_size=choice(32, 64, 128, 256)' \
    'model.batch_norm=choice(True, False)' \
    'model.optimizer.lr=interval(0.001, 0.01)' \
    "hydra.sweeper.n_trials=20"
```

As follows, we will explain each of these options.

## Standard boilerplate

-   `model=mlp`: model is set to a MultiLayer Perceptron. This references a specific model in Crescendo: `crescendo.models.mlp:MultilayerPerceptron`. The input and output layers of the model are automatically determined from the structure of the data.
-   `data=regression`: indicates that the problem at hand is regression. This also references a specific object in Crescendo. In this case, a generic Lightning Datamodule: `crescendo.data:ArrayRegressionDataModule`.
-   `+data.data_dir=../data/C-XANES`: the `+` syntax is special to Hydra. It indicates to add a new attribute that isn't present by default in any of the configs. In this case, we're adding the `data_dir` attribute to the DataModule, which is required in order for the DataModule to know where to load the data from. Note that the directory we point to requires a certain structure. Particularly it needs to have `X_train.npy`, `Y_train.npy`, `X_val.npy` and `Y_val.npy` at minimum.
-   `trainer.max_epochs=10`: sets the maximum number of epochs to 10. Obviously you'll want to train this for real hyperparamter tuning.
-   `model.last_activation="{_target_: torch.nn.Sigmoid}"`: more Hydra-specific syntax for initializing an PyTorch object. In this case, it sets the `last_activation` attribute of the model to the `torch.nn.Sigmoid()` object. Hydra knows when it is passed the `_target_` key to try and initialize the provided value as an object.
-   `trainer.accelerator=cpu`: use the cpu as opposed to any other hardware that might be available.

## Optuna settings

Hydra uses an Optuna plugin for hyperparamter tuning. The standard Hydra sweeper is overridden by `hparams_search=optuna`, and the following model/data/and other parameters are overridden by the usual syntax. It's worth noting that

- `choice(A, B, C, ...)` indicates to Optuna to make a choice between a finite number of elements
- `interval(x0, xf)` indicates to Optuna to choose in the provided continuous range
