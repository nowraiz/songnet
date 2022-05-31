import torch.nn as nn

class WaveConvolution(nn.Module):
    """
    A convolution based model that is applied directly to the wave, to produce another wave with possibly smaller
    sizes
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 256, 4096, 2048),
            nn.ReLU(),
            nn.Conv1d(256, 256, 128, 64),
            nn.ReLU(),
            nn.Conv1d(256, 128, 5, 1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, 1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 256, 5, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256, 128, 64, output_padding=6),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 1, 4096, 2048, output_padding=1952 ),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.layers(x)