import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    Model for training the mystery data
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        initialize the model for training
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.drop25 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1600, out_features=512)
        self.drop50 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        foward pass for the model
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max1(x)
        x = self.drop50(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.max1(x)
        x = self.drop50(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.drop50(x)
        x = self.relu(self.linear2(x))
        x = self.drop50(x)
        x = self.linear3(x)
        return x
