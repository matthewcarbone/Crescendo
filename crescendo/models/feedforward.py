#!/usr/bin/env python3

import torch
import torch.nn as nn


class PerceptronFrame(nn.Module):
    """The standard MLP module."""

    def __init__(
        self, input_size, hidden_size, output_size, n_hidden_layers, dropout
    ):
        """Initializer.

        Parameters
        ----------
        input_size : int
            The size of the input layer. This is the number of features.
        hidden_size : int
            The size of every hidden layer.
        output_size : int
            The size of the output layer. This is the number of target
            features.
        n_hidden_layers : int
            The number of internal hidden layers of the network.
        dropout : float
            Percentage dropout.
        """

        super().__init__()

        self.drop = nn.Dropout(p=dropout)

        self.input_layer = torch.nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size)
            for __ in range(n_hidden_layers)
        ])

        self.predict_layer = torch.nn.Linear(hidden_size, output_size)


class StandardPerceptron(PerceptronFrame):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        """Forward propagation for the Perceptron. Applies relu activations
        to every layer, in addition to dropout."""

        x = self.drop(torch.relu(self.input_layer(x)))
        for layer in self.hidden_layers:
            x = self.drop(torch.relu(layer(x)))
        return self.predict_layer(x)  # Linear output
