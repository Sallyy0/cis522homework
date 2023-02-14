import torch
from typing import Callable
import torch
from torch import Tensor


class MLP(torch.nn.Module):
    """
    creates a MLP
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, 84)
        self.l3 = torch.nn.Linear(84, num_classes)
        self.activation = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.l3(x)
        return torch.nn.functional.log_softmax(x, dim=1)
