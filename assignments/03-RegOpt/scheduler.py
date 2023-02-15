from typing import List

from torch.optim.lr_scheduler import _LRScheduler

# import torch.optim.lr_scheduler
import math


class CustomLRScheduler(_LRScheduler):
    """
    Creates a custom scheduler
    """

    def __init__(self, optimizer, step_size, gamma, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.step_size = step_size
        self.num_iter = 64
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        getting the updated learning rate
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
        #     return [group["lr"] for group in self.optimizer.param_groups]
        # return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
        # print(self.base_lrs)
        # learning_rate = 0.1
        # decay_rate = learning_rate / (self.last_epoch + 1)

        # return [
        #     0.001
        #     + (base_lr - 0.001)
        #     * (1 + math.cos(math.pi * (self.last_epoch + 1) / 2))
        #     / 2
        #     for base_lr in self.base_lrs
        # ]
        return [
            0.001
            + (base_lr - 0.001) * (1 + math.cos(math.pi * self.last_epoch / 500)) / 2
            for base_lr in self.base_lrs
        ]
