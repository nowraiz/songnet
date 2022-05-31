import os
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from songnet.audio.loader import AudioLoader
from songnet.models.convolution.waveconv import WaveConvolution

AUDIO_DIR = "data/converted"
FREQ = 44100
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(src, target):
    for idx in BatchSampler(RandomSampler(range(src.size(1))), BATCH_SIZE, False):
        yield src[:,idx], target[:,idx]

def sample_simple(src):
    for idx in BatchSampler(RandomSampler(range(len(src))), BATCH_SIZE, False):
        yield [src[i] for i in idx]

def pad(x):
    padd = nn.ConstantPad1d((0,44100*360-x.size(1)), 0)
    return padd(x)

def main():
    writer = SummaryWriter("runs/exp1")
    net = WaveConvolution().to(DEVICE)
    optimizer = Adam(net.parameters())

    mse = torch.nn.SmoothL1Loss()
    files = os.listdir(AUDIO_DIR)
    for epoch in range(10):
        avg_epoch_loss = 0
        waveforms = []
        batches = 0
        for fil in sample_simple(files):
            waveforms = []
            for f in fil:
                f = f"{AUDIO_DIR}/{f}"
                # print(torchaudio.info(f))
                loader = AudioLoader()
                waveform = loader.load_resample(f, FREQ)
                # print(audio)
                waveform = pad(waveform)
                waveforms.append(waveform)
            # print(seq.size(), target.size())
            # target = target[1:]
            # print(target)
            avg_loss = 0
            wave = torch.stack(waveforms, dim=0).to(DEVICE)
            print(wave.size())
            output = net(wave)
            avg_loss = mse(output, wave)
            print(f"Epoch: {epoch}, Batch: {batches}, Loss: {avg_loss.item()}")
            avg_loss.backward()
            optimizer.step()
            avg_epoch_loss += avg_loss.item()
            batches += 1
        avg_epoch_loss /= batches
        writer.add_scalar("Loss/Train: ", avg_epoch_loss, epoch)
        

        

    

    # loader = AudioLoader("")

if __name__ == "__main__":
    main()