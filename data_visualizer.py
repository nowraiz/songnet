import json
from songnet.spotify.features import SpotifyFeatures
from sklearn.manifold import TSNE
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cosine
import pprint
import random

spotify = SpotifyFeatures()

data1 = "data/SpotifyData/StreamingHistory0.json.features.json"
data2 = "data/SpotifyData/StreamingHistory1.json.features.json"
unique_tracks = set()

def parse_data(filename):
    songs = {}
    raw_features = []
    names = []
    artists = [
        "Marshmello",
        "Alan Walker",
        "Sasha Alex Sloan"
    ]
    y = []
    with open(filename) as f:
        songs = json.load(f)

    for track in songs:
        feature_dict = track["features"]
        track_id = track["id"]
        if track_id in unique_tracks:
            continue
        artist = track["artist"]

        # print(artist)
        # if artist not in artists:
            # continue
        unique_tracks.add(track_id)
        feature_vec = spotify.get_feature_vector(feature_dict)
        if feature_vec is not None:
            raw_features.append(spotify.get_feature_vector(feature_dict))
            y.append(track["artist"])
            names.append(track["track"])
    
    return names, raw_features, y
        

def create_playlist(features, names, seed):
    """
    Genearte a playlist based on the first song
    """
    p = 0.5
    idx = -1
    for i, k in enumerate(names):
        if k == seed:
            idx = i

    n_items = 20
    play_list = [] 
    play_list.append(names[idx])
    for _ in range(n_items):
        vec = features[idx]
        distances = [(i, names[i], cosine(vec, song)) for i, song in enumerate(features)]
        distances = sorted(distances, key=lambda x: x[2])[1:20]
        # choose = random.randint(0, len(distances)-1)
        choose = 0
        # item = rand_choose(distances, p)
        item = distances[choose]
        while (item[1] in play_list):
            # choose = random.randint(0, len(distances)-1)
            choose += 1
            item = distances[choose]
        play_list.append(item[1])
        idx = item[0]

    pprint.pprint(play_list)



def rand_choose(elements, p):
    toss = random.random()
    i = 0
    while toss >= p:
        toss = random.random()
        i = (i + 1) % len(elements)
    return elements[i]


        
def recommendation(features, names):
    """
    Generate recommendations based on the similar songs
    """
    query = "Phir Bhi Tumko Chaahunga"
    idx = -1
    for i, k in enumerate(names):
        if k == query:
            idx = i
    vec = features[idx]
    distances = [(names[i], cosine(vec, song)) for i, song in enumerate(features)]
    distances = sorted(distances, key=lambda x: x[1])[1:]
    pprint.pprint(distances)



def main():
    names = []
    features = []
    y = []
    n, x, _y = parse_data(data1)
    features.extend(x)
    y.extend(_y)
    names.extend(n)
    n, x, _y = parse_data(data2)
    features.extend(x)
    y.extend(_y)
    names.extend(n)
    unique = len(set(y))
    # create_playlist(features, names, 'Arcade')
    embedded = TSNE().fit_transform(features)

    # print(embedded)

    embedded_x = [x[0] for x in embedded]
    embedded_y = [x[1] for x in embedded]
    
    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
    x=embedded_x, y=embedded_y,
    hue=y,
    palette=sns.color_palette("Paired", unique),
    legend="full",
    alpha=0.9
    )

    fig = plot.get_figure()
    fig.savefig("tsne.png")


if __name__ == "__main__":
    main()