import torch
from songnet.spotify.features import FEATURES as RELEVANT_FEATURES 
from datetime import datetime, timedelta
from torch.nn import CosineSimilarity
import json

class SpotifyDataLoader:
    """
    A helper class to load the spotify streaming data json and convert it into sequences of songs along
    with relevant features

        min_played:         The minimum fraction of the song played to be part of the sequence, from 0.0 to 1.0
        filter_skipped:     Whether to filter the (seemingly) skipped songs during streaming history
                            Skipped songs are flagged based on if the fraction streamed was less the min_played
        remove_duplicates:  Whether to remove duplicates from the streaming history. A duplicate songs is 
                            that is played back to back in one session
        normalize:          Type of normalization for the features
                                'min': min-max normalization
                                'mean': mean-std normalization
                                'none': No normalization
    """

    def __init__(self, data_file, min_played=0.15, filter_skipped=True, remove_duplicates=True, 
                        normalize="min") -> None:
        self.data_file = data_file
        self.data = None
        self.sequences = None
        self.min_played = min_played
        self.song_map = {}
        self.index_map = {}
        self.reverse_index_map = {}
        self.min_dict = {}
        self.max_dict = {}
        self.normalize = normalize

        self._load_data()
        if filter_skipped:
            self._filter_skipped()
        if remove_duplicates:
            self._remove_duplicates()
        self.split_data()
        self._build_song_map()
        self._build_index_map()
        self._calculate_min_max()
        self._normalize()


    def get_sequences(self):
        """
        Returns the loaded streaming history as a sequence of streams. A stream is just a json object describing
        the song along with the relevant data extracted from the Spotify API
        """
        return self.sequences


    def get_song_by_id(self, id):
        """
        Returns the json record for the song with the given id
        """
        return self.song_map.get(id)

    def get_song_by_idx(self, idx):
        """
        Returns the json record for the song with the given unique index
        """
        id_ = self.reverse_index_map[idx]
        return self.get_song_by_id(id_)

    def get_song(self, id):
        """
        Returns the song title for the given id if exists otherwise returns None.
        """
        song = self.song_map.get(id)
        name = None
        if song is not None:
            name = " - ".join([song["track"], song["artist"]])
        return name

    def get_closest_song(self, features):
        """
        Returns the song record for the closest song based on the consine similarity between the given
        features and the song features
        """
        max_similar = None
        max_similarity = float("-inf")
        cosine_similarity = CosineSimilarity(0)

        for song in self.song_map.values():
            song_features = song["features"]
            feature_vector = torch.tensor([song_features[feature] for feature in RELEVANT_FEATURES], device="cuda")
            cosine = cosine_similarity(features, feature_vector)
            if cosine >= max_similarity:
                max_similarity = cosine.item()
                max_similar = song
        
        return max_similar



    def get_sequences_as_tensor(self):
        """
        Returns the equivalent of get sequences as pytorch tensors. Where each tensor describes the relevant features
        of the song in the streaming sequence
        """
        pass

    def get_index_map(self):
        """
        Returns the index map
        """
        return self.index_map

    def get_unique_songs(self):
        """
        Returns the number of unique songs i.e. size of the index map
        """
        return len(self.index_map)
    
    def get_min_max_dict(self):
        """
        Returns the min and max dictionary for each of the feature value in the data
        """
        return self.min_dict, self.max_dict

    def split_data(self):
        """
        Splits the streaming history into multiple sequences. For our purpose, a streaming session of one hour
        is considered a sequence regardless of time. 
        """
        self.sequences = []
        assert len(self.data) > 0
        seq = []
        start_datetime = datetime.fromisoformat(self.data[0]["endTime"])
        seq.append(self.data[0])
        i = 1
        while i < len(self.data):
            record = self.data[i]
            time = datetime.fromisoformat(record["endTime"])
            if time > start_datetime + timedelta(hours=1):
                if seq:
                    self.sequences.append(seq)
                    seq = []
                start_datetime = time
            else:
                seq.append(record)
            i += 1
        
    def _filter_skipped(self):
        """
        Attempts to filter the songs that were skipped during the streaming history. Since its impossible to
        determine with utter certainty, we try to skip songs if they were played for less than 25% (the default
        fraction of 0.25) of their total time based on the streaming duration.
        """
        skipped = lambda record : record["msPlayed"] / record["duration"] >= self.min_played
        old_len = len(self.data)
        self.data = list(filter(skipped, self.data))
        print(f"Filtered {old_len - len(self.data)} skipped songs")

    def _remove_duplicates(self):
        """
        Removes any duplicates present in a single sequence in a successive manner (back to back)
        """
        data = self.data
        self.data = []
        for record in data:
            if not self.data or record["id"] != self.data[-1]["id"]:
                self.data.append(record)
        
        print(f"Removed {len(data) - len(self.data)} duplicates")

    def _normalize(self):
        """
        Normalizes the features based on the type of normalization specified in the constructor
        """
        if self.normalize == "none":
            return
        elif self.normalize == "min":
            # do min max normalization
            for record in self.data:
                features = record["features"]
                for feature in RELEVANT_FEATURES:
                    min_, max_ = self.min_dict[feature], self.max_dict[feature]
                    features[feature] = (features[feature] - min_) / (max_ - min_)
        elif self.normalize == "mean":
            raise NotImplementedError("Mean normalization not implemented yet")

        
    def _calculate_min_max(self):
        """
        Calculates the min max for each of the relevant features in a min vector (dict) and a max vector (dict)
        """
        self.min_dict = {feature: float("+inf") for feature in RELEVANT_FEATURES}
        self.max_dict = {feature: float("-inf") for feature in RELEVANT_FEATURES}
        
        for record in self.data:
            for feature in RELEVANT_FEATURES:
                feature_val = record["features"][feature]
                self.min_dict[feature] = min(self.min_dict[feature], feature_val)
                self.max_dict[feature] = max(self.max_dict[feature], feature_val)

        print(self.min_dict)
        print(self.max_dict)
            

    def _build_song_map(self):
        """
        Builds the mapping from song id to the song record
        """
        self.song_map = {}
        for record in self.data:
            self.song_map[record["id"]] = record

    def _build_index_map(self):
        idx = 0
        """
        Builds the index map by assigning each unique id to a integer starting from 0
        """
        for id in self.song_map:
            self.index_map[id] = idx
            self.reverse_index_map[idx] = id
            idx += 1

    def _load_data(self):
        """
        Loads the spotify data into a list of json objects where each object is an independent stream of a song
        """

        with open(self.data_file) as f:
            self.data = json.load(f)