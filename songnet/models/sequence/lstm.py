import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class LSTMSequenceModel(nn.Module):
    
    def __init__(self, num_features, song_library_size):
        super().__init__()
        self.num_features = num_features
        self.lstm = nn.LSTM(input_size=song_library_size+num_features, hidden_size=128, batch_first=True)
        self.decoder = nn.Linear(128, song_library_size)


    def forward(self, input, hidden=None):
        # output = self.encoder(input)
        if hidden is None:
            output, hidden = self.lstm(input)
        else:
            output, hidden = self.lstm(input, hidden)
        # output = F.relu(hidden[0])
        # print(output)
        if isinstance(output, PackedSequence):
            output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[:, -1, :]
        output = self.decoder(output)
        return output, hidden


