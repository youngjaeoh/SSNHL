import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.network = nn.Sequential(
            # nn.Linear(29, 32),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(32),
            #
            # nn.Linear(32, 64),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(64),
            #
            # nn.Linear(64, 128),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(128),
            #
            # nn.Linear(128, 32),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(32),
            #
            # nn.Linear(32, 1),
            # nn.Sigmoid()

            nn.Linear(29, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y
