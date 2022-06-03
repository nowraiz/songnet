from time import sleep
from songnet.spotify.features import SpotifyFeatures
import pprint
import json

spotify = SpotifyFeatures()

history_file = "data/SpotifyData/StreamingHistory0.json"
history_file_2 = "data/SpotifyData/StreamingHistory1.json"
processed_data_file = "data/processedStreamingHistory.json"


def parse_streaming_history(filename):
    data = []
    with open(filename, encoding="utf-8") as f:

        history = json.load(f)
        for record in history:
            track_name = record["trackName"]
            artist_name = record["artistName"]
            print(f"{track_name} - {artist_name}")
            try:
                track_id, metadata, track_features = spotify.get_track_features(track_name)
            except Exception:
                print("Exception")
                continue
            if track_id is None or track_features is None:
                print(f"Cannot find features for track: {track_name}")
            else:
                data.append({ "endTime": record["endTime"], "msPlayed": record["msPlayed"],
                     "track": track_name, "artist": artist_name, "id": track_id, "duration": metadata["duration"],
                 "features": track_features })
            sleep(0.01)
            
    return data



def main():
    """
    Main
    """
    # print(spotify.get_track_features("Beete Lamhein - KK"))

    print(f"Beginning feature extraction for tracks in the streaming history")
    data1 = parse_streaming_history(history_file)
    data2 = parse_streaming_history(history_file_2)
    data = data1 + data2
    print(f"Dumping processed data to file: {processed_data_file}")
    with open(processed_data_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()