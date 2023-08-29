from time import perf_counter
from rich.console import Console
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from lightning import LightningModule  

console = Console()

class ConvolutionalNeuralNetwork(LightningModule):
    def __init__(
        self,
        input_channels,
        num_classes,
        architecture,
        fc_architecture,
        criterion,
        optimizer,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_channels = input_channels
        for out_channels, kernel_size, stride in architecture:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        self.fc_layers = nn.ModuleList()
        in_features = fc_architecture[0][0]
        for in_f, out_f in fc_architecture:
            self.fc_layers.append(nn.Linear(in_f, out_f))

        self.criterion = criterion
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)

        for layer in self.fc_layers:
            x = F.relu(layer(x))
        
        return x

    def model_step(self, batch):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        console.log(f"Optimizer configured {optimizer.__class__}")

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            console.log(f"Scheduler configured {scheduler.__class__}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
