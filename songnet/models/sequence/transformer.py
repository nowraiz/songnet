import torch
import torch.nn as nn

from songnet.utils.positional_encoding import PositionalEncoding

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SongNet(nn.Module):
    """
    A transformer based sequence model for audio signals
    """

    def __init__(self, features=1, num_heads=8, auto_encoder_layers=6, dropout=0.1, dim_feedforward=2048) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding(1).to(DEVICE)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout, nhead=1).to(DEVICE)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=auto_encoder_layers).to(DEVICE)

        self.transformer = nn.Transformer(
            d_model=1, 
            dim_feedforward=dim_feedforward, 
            custom_encoder=self.encoder,
            nhead=1).to(DEVICE)


    def forward(self, src, target):
        src = self.positional_encoding(src)
        target = self.positional_encoding(target)
        return self.transformer(src, target, tgt_mask=nn.Transformer.generate_square_subsequent_mask(256).to(DEVICE))

