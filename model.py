import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()

            # nn.Linear(29, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y
