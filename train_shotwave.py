import os
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from songnet.audio.loader import AudioLoader
from songnet.models.convolution.shotwave import ShotWaveNet

AUDIO_DIR = "data/converted"
FREQ = 44100
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(src, target, dim=1):
    for idx in BatchSampler(RandomSampler(range(src.size(dim))), BATCH_SIZE, False):
        indices = torch.tensor(idx)
        yield torch.index_select(src, dim, indices), torch.index_select(target, dim, indices)

def sample_simple(src):
    for idx in BatchSampler(RandomSampler(range(len(src))), BATCH_SIZE, False):
        yield [src[i] for i in idx]

def pad(x):
    padd = nn.ConstantPad1d((0,44100*360-x.size(1)), 0)
    return padd(x)

def chunk_audio(audio, interval=1):
    size_in_seconds = FREQ*interval
    split = list(torch.split(audio, size_in_seconds, 1))
    if split[-1].size(1) < size_in_seconds:
        split = split[:-1]
    return torch.stack(split, dim=0)

def main():
    writer = SummaryWriter("runs/exp1")
    net = ShotWaveNet().to(DEVICE)
    optimizer = Adam(net.parameters(), lr=1e-4)

    mse = torch.nn.SmoothL1Loss()
    files = os.listdir(AUDIO_DIR)
    for epoch in range(10):
        avg_epoch_loss = 0
        waveforms = []
        batches = 0
        for fil in files:
            f = f"{AUDIO_DIR}/{fil}"
            # print(torchaudio.info(f))
            loader = AudioLoader()
            waveform= loader.load_resample(f, FREQ)
            waveforms = chunk_audio(waveform) 
            avg_loss = 0

            for wav, _ in sample(waveforms, waveforms, 0):
                wav = wav.to(DEVICE)
                output = net(wav)
                avg_loss = mse(output, wav)
                print(f"Epoch: {epoch}, Batch: {batches}, Loss: {avg_loss.item()}")
                avg_loss.backward()
                optimizer.step()
                batches += 1
                
        avg_epoch_loss += avg_loss.item()
        # avg_epoch_loss /= batches
        # writer.add_scalar("Loss/Train: ", avg_epoch_loss, epoch)
        

        

    

    # loader = AudioLoader("")

if __name__ == "__main__":
    main()