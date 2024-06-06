import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, dim):
        super(ConvNet, self).__init__()

        self.code = nn.Sequential(

            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=1),
            nn.BatchNorm1d(10),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3, stride=1),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=1),

            nn.Flatten(),
            nn.Linear(in_features=20 * dim - 160, out_features=150),
            nn.Linear(in_features=150, out_features=25),
            nn.Linear(in_features=25, out_features=1),
        )

    def forward(self, x):
        x = self.code(x)

        return x
