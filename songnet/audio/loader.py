import torch
import torchaudio
import torchaudio.functional as F

SEQ_SIZE = 256

class AudioLoader:
    """
    A helper class to load audio files into tensor. It also has some helper methods to convert it into
    batches of sequences of train samples and test samples
    """

    def __init__(self) -> None:
        pass

    def load(self, path):
        waveform, f = torchaudio.load(path)
        return waveform

    def load_resample(self, path, freq):
        waveform, f = torchaudio.load(path, normalize=True )
        if f == freq:
            return waveform

        return F.resample(waveform, f, freq)

    def create_sequences(self, waveform, seq_size=256):
        """
        Given a waveform, converts it into sequences of size seq_size with total sequences len(waveform) // seq_size
        """
        sequences = []
        i = 0
        length = waveform.shape[1]
        # print(waveform.shape[1]
        while length >= seq_size:
            sequences.append(waveform[0][i:i+seq_size].unsqueeze(1))
            i += seq_size
            length -= seq_size
        return torch.stack(sequences, dim=1)

