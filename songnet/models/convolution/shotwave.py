import torch
import torch.nn as nn

FREQ = 44100
FRAGMENT_LENGTH = 1

class ShotWaveNet(nn.Module):
    """
    A convolution autoencoder network for short wave sequences
    """

    def __init__(self) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(1, 128, 128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, 64),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, 32),
            nn.LeakyReLU(),
            nn.Conv1d(64, 1, 5),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(1, 64, 5),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 128, 32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 128, 64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 1, 128),
            nn.Tanh()

        )

    def forward(self, x):
        return self.layers(x)
