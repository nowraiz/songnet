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
        self.sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id="39c5275d30d044d68b705ab372f3ed5f", 
        client_secret="d825c7a8eb5e4938804d1b3c1f2e4be3"))
        

    def get_track_features(self, name):
        """
        Returns the features object of the given track name by returning the first search result
        """
        track_id = self.get_id(name)
        if track_id is None:
            return None, None
        
        features = self.sp.audio_features(track_id)
        if len(features) < 1:
            return None, None
        
        features = features[0]
        filtered = { key: features[key] for key in FEATURES }
        return track_id, filtered

    
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
        result = self.sp.search(name, limit=1, market="US")
        return result

    def get_id(self, name):
        """
        Attempts to get the spotify id for a given track by returning the id of the first search result
        for the given track
        """
        result = self.search(name)
        if "tracks" not in result:
            return None
        tracks = result["tracks"]
        if "items" not in tracks:
            return None
        items = tracks["items"]
        if len(items) < 1:
            return None
        item = items[0]
        if "id" not in item:
            return None
        return item["id"]
