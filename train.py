import os
import torch
import torchaudio
from torch.utils.data import BatchSampler, RandomSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from songnet.audio.loader import AudioLoader
from songnet.models.sequence.transformer import SongNet

AUDIO_DIR = "data/converted"
FREQ = 8000
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(src, target):
    for idx in BatchSampler(RandomSampler(range(src.size(1))), BATCH_SIZE, False):
        yield src[:,idx], target[:,idx]



def main():
    writer = SummaryWriter("runs/exp1")
    net = SongNet(features=1)
    optimizer = Adam(net.parameters())

    mse = torch.nn.MSELoss()
    files = os.listdir(AUDIO_DIR)
    for epoch in range(10):
        avg_epoch_loss = 0
        for i, f in enumerate(files[:100]):
            f = f"{AUDIO_DIR}/{f}"
            # print(torchaudio.info(f))
            loader = AudioLoader()
            waveform = loader.load_resample(f, FREQ)
            # print(audio)
            
            seq = loader.create_sequences(waveform).to(DEVICE)
            target = seq.roll(-1, 1)
            seq = seq[:,:-1]
            target = target[:,:-1]
            # print(seq.size(), target.size())
            # target = target[1:]
            # print(target)
            avg_loss = 0
            r = 0
            for s, t in sample(seq, target):
                optimizer.zero_grad()
                pred = net(s, t)
                loss = mse(pred, t)
                avg_loss += loss.item()
                r += 1
                loss.backward()
                optimizer.step()
            avg_loss /= r
            avg_epoch_loss += avg_loss
            print(i)
            writer.add_scalar("Loss/TrainPerFile", avg_loss, i)
            if (i % 20) == 0:
                torch.save(net, "saved.model")
        writer.add_scalar("Loss/Train", avg_epoch_loss/i, epoch)
        

    

    # loader = AudioLoader("")

if __name__ == "__main__":
    main()