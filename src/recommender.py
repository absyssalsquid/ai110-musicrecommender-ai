import csv
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float
    speechiness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    name: str
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

    genre: str
    mood: str
    energy: float
    tempo_bpm: int
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float
    speechiness: float

@dataclass
class Features:
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_norm: int
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float
    speechiness: float

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):

        # for normalization
        tempos = [s.tempo_bpm for s in songs]
        lo, hi = min(tempos), max(tempos)
        span = hi - lo
        self.tempo_lo, self.tempo_hi, self.tempo_span = lo, hi, span

        self.songs = songs
        self.songs_by_id = {song.id: song for song in songs}
        self.song_features_by_id = {song.id: self.extract_features(song) for song in songs}

        self.cat_selectors = ["artist", "genre", "mood"]
        self.num_selectors = ["energy","tempo_norm","valence","danceability","acousticness","instrumentalness","speechiness"]

        # low scores are better matches, high scores are hit harder
        self.weights = {
            "genre": 0.8,
            "mood": 1.5,
            "energy": 2,
            "valence": 3,
            "danceability": 2,
            "tempo_norm": 0.5,
            "speechiness": 0.5,
            # acousticness, instrumentalness, → implicit 1
        }
        self.penalty_weight = 0.7
        self.penalty_weights = {
            "artist": 2,
        }

    def extract_features(self, input) -> Features:
        """Extract normalized features from a song or user profile."""
        bpm = getattr(input, "tempo_bpm", None)
        tempo_norm = None
        if bpm is not None:
            tempo_norm = (bpm - self.tempo_lo) / self.tempo_span
            if tempo_norm > 1: tempo_norm = 1 # clamp
            if tempo_norm < 0: tempo_norm = 0 # clamp

        return Features(
            artist=getattr(input, "artist", None),
            genre=getattr(input, "genre", None),
            mood=getattr(input, "mood", None),
            energy=getattr(input, "energy", None),
            tempo_norm=tempo_norm,
            valence=getattr(input, "valence", None),
            danceability=getattr(input, "danceability", None),
            acousticness=getattr(input, "acousticness", None),
            instrumentalness=getattr(input, "instrumentalness", None),
            speechiness=getattr(input, "speechiness", None),
        )

    def feature_distance(self, a: Features, b: Features, weight: Optional[Dict] = None, invert=False) -> Dict:
        """Calculate weighted distance between two feature sets."""
        weight = self.weights if weight is None else weight
        distances = defaultdict(float)

        for s in self.cat_selectors:
            av, bv = getattr(a, s, None), getattr(b, s, None)
            if av is None or bv is None: continue
            distances[s] = int(av != bv)

        for s in self.num_selectors:
            av, bv = getattr(a, s, None), getattr(b, s, None)
            if av is None or bv is None: continue
            distances[s] = abs(av - bv)

        if invert:
            for s in distances:
                distances[s] = 1 - distances[s]

        # for s in distances:
        #     distances[s] = pow(distances[s], 2)

        for s, w in weight.items():
            if s in distances:
                distances[s] *=  w

        return distances
    
    def score(distances:Dict) -> int:
        """Sum all distance values into a single score."""
        total = 0
        for k, v in distances.items(): 
            total += v
        return total
    
    def song_vs_user_distance(self, song_id: int, user_features: Features) -> tuple[int, Features] :
        """Calculate distance between a song and user preferences."""
        song_features = self.song_features_by_id[song_id]
        return self.feature_distance(song_features, user_features, self.weights)

    def penalize(self, song_id, ref_id, song_distances):
        """Penalize a song for similarity to a reference song to promote diversity."""
        similarity = self.feature_distance(
            self.song_features_by_id[song_id], 
            self.song_features_by_id[ref_id],
            self.penalty_weights,
            invert=True,
            )

        penalty = self.penalty_weight * Recommender.score(similarity)
        # print(f"\tsong penalized: {penalty} ; {song_id} {self.songs_by_id[song_id]}")
        
        song_distances[song_id]["penalty"] += penalty

    def select_songs(self, song_distances, k: int = 5) -> List[int]:
        """Select top k songs using greedy selection with diversity penalties."""
        # returns list of song ids
        song_ids = [song.id for song in self.songs]

        ret = []
        for _ in range(k):
            song_ids.sort(key=lambda id: Recommender.score(song_distances[id]))
            selected_id = song_ids.pop(0)
            ret.append(selected_id)
            # print("selected", selected_id, self.songs_by_id[selected_id])

            # to ensure diversity, penalize songs that have similar metrics to the selected song
            for id in song_ids:
                self.penalize(id, selected_id, song_distances)

        song_ids = [song.id for song in self.songs]
        song_ids.sort(key=lambda id: Recommender.score(song_distances[id]))

        # for id in song_ids:
        #     dist = song_distances[id]
        #     print(id, self.songs_by_id[id].title, Recommender.score(dist), dist)
        return ret

    def recommend_plus (self, user: UserProfile, k: int = 5) -> List[Song]:
        """Recommend songs with distance scores and detailed feature distances."""
        # used in main.py
        user_features = self.extract_features(user)
        songs_dist = {id: self.song_vs_user_distance(id, user_features) for id in self.songs_by_id}
        selected_song_ids = self.select_songs(songs_dist)

        ret = []
        for id in selected_song_ids:
            song = self.songs_by_id[id]
            distance = songs_dist[id]
            score_wo_penalty = Recommender.score(distance) - distance.get("penalty", 0)
            ret.append((song, score_wo_penalty, distance))
        return ret

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Recommend top k songs matching user preferences."""
        user_features = self.extract_features(user)
        songs_dist = {id: self.song_vs_user_distance(id, user_features) for id in self.songs_by_id}
        selected_song_ids = self.select_songs(songs_dist)
        return [self.songs_by_id[id] for id in selected_song_ids]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Generate a human-readable explanation of why a song was recommended."""
        user_features = self.extract_features(user)
        song_features = self.extract_features(song)
        difference = self.feature_distance(song_features, user_features)

        ret = ""
        if "penalty" in difference:
            ret+=f" -- Penalty (+{v:.2f})"
        hi = []
        lo = []

        for k, v in difference.items():
            if k == "penalty": 
                continue
            if v < 0.05: hi.append(k)
            if v > 0.95: lo.append(k)

        if hi: ret += f" -- high alignment (<0.05): {", ".join(hi)}"
        if lo: ret += f" -- low alignment (>0.95): {", ".join(lo)}"
        return ret
    

def load_songs(csv_path: str) -> List[Song]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")

    # Columns that should be parsed as numbers rather than left as strings.
    int_fields = {"id", "tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness", "instrumentalness", "speechiness"}

    songs: List[Song] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fields = dict(row)
            for field in int_fields:
                if fields.get(field):
                    fields[field] = int(fields[field])
            for field in float_fields:
                if fields.get(field):
                    fields[field] = float(fields[field])
            # CSV headers match Song's field names, so unpack directly.
            songs.append(Song(**fields))

    print(f"Loaded {len(songs)} songs.")
    return songs


def load_profiles(csv_path: str) -> List[UserProfile]:
    """
    Loads user profiles from a CSV and returns a list of UserProfile.
    Blank cells become None so sparse profiles (only genre/mood set) are
    handled gracefully. The favorite_*/target_energy/likes_acoustic fields
    are derived from the feature columns.
    """
    feature_fields = ["name", "genre", "mood", "energy", "tempo_bpm", "valence",
                      "danceability", "acousticness", "instrumentalness", "speechiness"]
    int_fields = {"tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness",
                    "instrumentalness", "speechiness"}

    profiles: List[UserProfile] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attrs = {}
            for field in feature_fields:
                val = row.get(field, "")
                if val is None or val == "":
                    attrs[field] = None
                elif field in int_fields:
                    attrs[field] = int(val)
                elif field in float_fields:
                    attrs[field] = float(val)
                else:
                    attrs[field] = val

            acousticness = attrs["acousticness"]
            profiles.append(UserProfile(
                favorite_genre=attrs["genre"],
                favorite_mood=attrs["mood"],
                target_energy=attrs["energy"],
                likes_acoustic=(acousticness is not None and acousticness > 0.5),
                **attrs,
            ))

    print(f"Loaded {len(profiles)} profiles.")
    return profiles

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    # TODO: Implement scoring logic using your Algorithm Recipe from Phase 2.
    # Expected return format: (score, reasons)
    return []

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    # TODO: Implement scoring and ranking logic
    # Expected return format: (song_dict, score, explanation)
    return []