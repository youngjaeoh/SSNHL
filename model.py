from torch import nn


class Model_32_64_128_32(nn.Module):
    def __init__(self):
        super(Model_32_64_128_32, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_64_32_32_16_8(nn.Module):
    def __init__(self):
        super(Model_64_32_32_16_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_64_64_32_16_8(nn.Module):
    def __init__(self):
        super(Model_64_64_32_16_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_32_16_16_8(nn.Module):
    def __init__(self):
        super(Model_128_32_16_16_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_32_32_16_8(nn.Module):
    def __init__(self):
        super(Model_128_32_32_16_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_64_32(nn.Module):
    def __init__(self):
        super(Model_128_64_32, self).__init__()

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

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_64_32_16(nn.Module):
    def __init__(self):
        super(Model_128_64_32_16, self).__init__()

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

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_64_32_16_8(nn.Module):
    def __init__(self):
        super(Model_128_64_32_16_8, self).__init__()

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

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_64_32_32_8(nn.Module):
    def __init__(self):
        super(Model_128_64_32_32_8, self).__init__()

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

            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_64_32_32_16(nn.Module):
    def __init__(self):
        super(Model_128_64_32_32_16, self).__init__()

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

            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_64_64_16_8(nn.Module):
    def __init__(self):
        super(Model_128_64_64_16_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_64_64_32_8(nn.Module):
    def __init__(self):
        super(Model_128_64_64_32_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 64),
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
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_64_64_32_16(nn.Module):
    def __init__(self):
        super(Model_128_64_64_32_16, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_128_32_16_8(nn.Module):
    def __init__(self):
        super(Model_128_128_32_16_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(32),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_128_64(nn.Module):
    def __init__(self):
        super(Model_128_128_64, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_128_64_16_8(nn.Module):
    def __init__(self):
        super(Model_128_128_64_16_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(8),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_128_64_32(nn.Module):
    def __init__(self):
        super(Model_128_128_64_32, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
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

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_128_64_32_8(nn.Module):
    def __init__(self):
        super(Model_128_128_64_32_8, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
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
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_128_64_32_16(nn.Module):
    def __init__(self):
        super(Model_128_128_64_32_16, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
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

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_256_128(nn.Module):
    def __init__(self):
        super(Model_128_256_128, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_256_128_64(nn.Module):
    def __init__(self):
        super(Model_128_256_128_64, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Model_128_256_128_64_16(nn.Module):
    def __init__(self):
        super(Model_128_256_128_64_16, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),

            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(16),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Meta(nn.Module):
    def __init__(self):
        super(Meta, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(22, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y
