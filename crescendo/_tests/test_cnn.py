# test_cnn_model.py

import unittest
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer

from cnn import ConvolutionalNeuralNetwork

class TestConvolutionalNeuralNetwork(unittest.TestCase):
    
    def test_initialization(self):
        """Test that the model initializes without error."""
        model = ConvolutionalNeuralNetwork(
            input_channels=1,
            num_classes=10,
            architecture=[(16, 3, 1), (32, 5, 1)],  
            fc_architecture=[(12 * 12 * 32, 128), (128, 64)], 
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
        )
        self.assertIsInstance(model, ConvolutionalNeuralNetwork)

    def test_forward_pass(self):
        """Test that a forward pass produces an output of the correct shape."""
        model = ConvolutionalNeuralNetwork(
            input_channels=1,
            num_classes=10,
            architecture=[(16, 3, 1), (32, 5, 1)],
            fc_architecture=[(12 * 12 * 32, 128), (128, 64)],
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
        )
        x = torch.randn(32, 1, 28, 28)  
        out = model(x)
        self.assertEqual(out.shape, (32, 10))  

    def test_training_step(self):
        """Test that a training step can be executed without crashing."""
        model = ConvolutionalNeuralNetwork(
            input_channels=1,
            num_classes=10,
            architecture=[(16, 3, 1), (32, 5, 1)],
            fc_architecture=[(12 * 12 * 32, 128), (128, 64)],
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
        )
        trainer = Trainer(max_epochs=1, fast_dev_run=True)  
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        trainer.fit(model, train_loader)

if __name__ == "__main__":
    unittest.main()
