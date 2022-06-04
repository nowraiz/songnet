import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import BatchSampler, RandomSampler
from songnet.data.loader import SpotifyDataLoader
from songnet.models.sequence.lstm import LSTMSequenceModel
from songnet.spotify.features import FEATURES as RELEVANT_FEATURES
import pprint
import matplotlib.pyplot as plt

DATA_FILE = "data/processedStreamingHistory.json"
BATCH_SIZE = 16
DEVICE = torch.device("cuda")

def sample(src, target):
    for idx in BatchSampler(RandomSampler(range(len(src))), BATCH_SIZE, False):
        yield [src[i] for i in idx], [target[i] for i in idx]

def create_data_set(sequences, index_map, expand_sequences=True):
    """
    Given a list of sequences from the SpotifyDataLoader, create a data set for training and testing as X and Y
    """
    X = []
    Y = []
    for sequence in sequences:
        if len(sequence) < 2:
            continue
        l = 1
        if expand_sequences:
            while l < len(sequence):
                x = sequence[:l] # assume the last element of the sequences as the target for the model
                y = sequence[l]
                l += 1
                X.append(x)
                Y.append(y)
        else:
            x = sequence[:-1]
            y = sequence[-1]
            X.append(x)
            Y.append(y)

    return convert_data_set_to_tensor(X, Y, index_map)

def convert_data_set_to_tensor(X, Y, index_map):
    """
    Given a set of records X and Y, convert it to a list of tensors data_x, and list of tensors data_y
    """
    data_x = []
    data_y = []
    assert len(X) == len(Y)
    for x, y in zip(X, Y):
        x = [convert_record_to_tensor(i) for i in x]
        x = torch.stack(x)
        y = convert_record_to_tensor(y)
        # y = convert_id_to_idx(y, index_map)
        # y = torch.tensor(y, device=DEVICE)
        data_x.append(x)
        data_y.append(y)
    return data_x, data_y

def convert_record_to_tensor(record):
    """
    Converts a loaded record with relevant spotify metadata to a tensor describing features
    """
    features = record["features"]
    feature_vector = [features[feature] for feature in RELEVANT_FEATURES]
    return torch.tensor(feature_vector, dtype=torch.float32, device=DEVICE)

def convert_id_to_idx(record, index_map):
    return index_map[record["id"]]


def generate_recommendation(starting_songs, model, data_loader, limit=20):
    """
    Generate recomendations based on the predictions of the model seeding from the starting song
    """
    songs = []
    # bootstrap
    hidden = None
    initial_songs = torch.stack(starting_songs)
    model.eval()
    with torch.no_grad():
        output, hidden = model(initial_songs.unsqueeze(0), hidden)
        # songs.append(data_loader.get_closest_song(output))
    # generate new songs
    print(output)
    # output = output[:, -1]
    song = data_loader.get_closest_song(output.squeeze())
    songs.append(song)
    # output = output.unsqueeze(0)
    output = convert_record_to_tensor(song).unsqueeze(0).unsqueeze(0)
    # print(hidden)
    h, c = hidden
    h = h[:,-1].unsqueeze(0)
    c = c[:,-1].unsqueeze(0)
    hidden = (h, c)
    while len(songs) < limit:
        with torch.no_grad():
            output, hidden = model(output, hidden)
            song = data_loader.get_closest_song(output.squeeze())
            songs.append(song)
            output = convert_record_to_tensor(song).unsqueeze(0).unsqueeze(0)
            # output = output.unsqueeze(0)

            # print(songs)
        
    return songs


def initial_songs(data_loader):
    # for now bootstrap based on id
    ids = ["0zQa7QXLpUZfrrsWbgDZll", "4xS7MhQnMv8ailfTUM347g", "1UWacd8x8tPPwmrPB1MoBI"]
    songs = []
    for id in ids:
        song = data_loader.get_song_by_id(id)
        song_vector = convert_record_to_tensor(song)
        songs.append(song_vector)
    
    return songs

def train(loader, epochs):
    data_x, data_y = create_data_set(loader.get_sequences(), loader.get_index_map())
    print(len(data_x))
    ratio = int(len(data_x) * 0.95)
    idx = list(RandomSampler(range(len(data_x))))
    train_idx = idx[:ratio]
    test_idx = idx[ratio:]
    train_x, train_y = [data_x[i] for i in train_idx], [data_y[i] for i in train_idx]
    test_x, test_y = [data_x[i] for i in test_idx], [data_y[i] for i in test_idx]
    model = LSTMSequenceModel(len(RELEVANT_FEATURES), loader.get_unique_songs())
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    for epoch in range(40):
        avg_loss = 0
        it = 0
        for batch_x, batch_y in sample(train_x, train_y):
            it += 1
            optimizer.zero_grad()
            packed_sequence = pack_sequence(batch_x, False)
            output, _ = model(packed_sequence)
            batch_y = torch.stack(batch_y)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()

        model.eval()
        packed_sequence = pack_sequence(test_x, False)
        y = torch.stack(test_y)
        output, _ = model(packed_sequence)
        test_loss = criterion(output.squeeze(), y)
        print(f"Epoch: {epoch}, Avg-loss: {avg_loss/it}, Test-loss: {test_loss.item()}")
        model.train()

    packed_sequence = pack_sequence(test_x, False)
    return model

def main():
    loader = SpotifyDataLoader(DATA_FILE, filter_skipped=True)
    model = train(loader, 40)
    starting_songs = initial_songs(loader)
    pprint.pprint(generate_recommendation(starting_songs, model, loader))

if __name__ == "__main__":
    main()