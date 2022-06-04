import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

FEATURES = [
    'danceability', 'energy', 'loudness', 'mode', 'key', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature']

class SpotifyFeatures:
    """
    A helper class to get the features of a song from spotify
    """

    def __init__(self) -> None:
        with open("client_secret") as f:
            secret = f.read()
        # self.sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id="39c5275d30d044d68b705ab372f3ed5f", 
        # client_secret=secret))
        
        self.sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id="39c5275d30d044d68b705ab372f3ed5f", client_secret=
        "fdfa2b7e996a4003823525286e3f47e3"))
        

    def get_track_features(self, name):
        """
        Returns the metadata and features object of the given track name by returning the features of the first 
        search result
        """
        track_id, duration = self.get_id_and_duration(name)
        if track_id is None:
            return None, None, None
        
        features = self.sp.audio_features(track_id)
        if len(features) < 1:
            return None, None, None
        
        features = features[0]
        # for some reason, the track features for some tracks is None. 
        if features is None:
            return None, None, None
        filtered = {key: features[key] for key in FEATURES}
        metadata = {'duration': duration}
        return track_id, metadata, filtered

    
    def get_feature_vector(self, feature_dict):
        """
        Returns a feature vector from the given feature dictionary
        """
        if feature_dict is None:
            return None
        features = []

        
        features = [feature_dict[key] for key in FEATURES]
        return features
        

    def search(self, name):
        """
        Attempts to search for a given track name and returns the result from spotify
        """
        result = self.sp.search(name, limit=1)
        return result

    def get_id_and_duration(self, name):
        """
        Attempts to get the spotify id for a given track by returning the id of the first search result
        for the given track. Also return the duration of the track in (ms) if found
        """
        result = self.search(name)
        if "tracks" not in result:
            return None, None
        tracks = result["tracks"]
        if "items" not in tracks:
            return None, None
        items = tracks["items"]
        if len(items) < 1:
            return None, None
        item = items[0]
        if "id" not in item:
            return None, None
        return item["id"], item["duration_ms"]
