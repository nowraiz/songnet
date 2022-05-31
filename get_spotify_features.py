from songnet.spotify.features import SpotifyFeatures
import pprint
import json

spotify = SpotifyFeatures()

history_file = "data/SpotifyData/StreamingHistory0.json"
history_file_2 = "data/SpotifyData/StreamingHistory1.json"


def parse_streaming_history(filename):
    data = []
    with open(filename) as f:

        history = json.load(f)
        for record in history:
            track_name = record["trackName"]
            artist_name = record["artistName"]
            print(f"{track_name} - {artist_name}")
            track_id, track_features = spotify.get_track_features(track_name)
            data.append({ "track": track_name, "artist": artist_name, "id": track_id, "features": track_features })
    with open(f"{filename}.features.json", "w") as f:
        json.dump(data, f)
            



def main():
    """
    Main
    """

    parse_streaming_history(history_file)
    parse_streaming_history(history_file_2)


if __name__ == "__main__":
    main()