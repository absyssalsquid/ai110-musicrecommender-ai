import csv
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SpotifyTrack:
    track_id: str
    name: str
    artist_ids: str
    popularity: float
    release_date: str
    danceability: float
    energy: float
    valence: float
    acousticness: float
    tempo: float
    instrumentalness: float
    speechiness: float


@dataclass
class SpotifyArtist:
    artist_id: str
    name: str
    monthly_listeners: float
    popularity: float
    followers: float
    genres: List[str]
    first_release: str
    last_release: str


class SpotifyCorpus:
    """Index of Spotify data for fast lookup and retrieval."""

    def __init__(self):
        self.tracks: Dict[str, SpotifyTrack] = {}
        self.artists: Dict[str, SpotifyArtist] = {}
        self.artist_to_tracks: Dict[str, List[str]] = {}

    def load_tracks(self, csv_path: str) -> None:
        """Load Spotify tracks from CSV."""
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    track = SpotifyTrack(
                        track_id=row["ids"],
                        name=row["names"],
                        artist_ids=row["artists"],
                        popularity=float(row["popularity"]) if row["popularity"] else 0,
                        release_date=row["release_date"],
                        danceability=float(row["danceability"]) if row["danceability"] else 0,
                        energy=float(row["energy"]) if row["energy"] else 0,
                        valence=float(row["valence"]) if row["valence"] else 0,
                        acousticness=float(row["acousticness"]) if row["acousticness"] else 0,
                        tempo=float(row["tempo"]) if row["tempo"] else 0,
                        instrumentalness=float(row["instrumentalness"]) if row["instrumentalness"] else 0,
                        speechiness=float(row["speechiness"]) if row["speechiness"] else 0,
                    )
                    self.tracks[track.track_id] = track

                    if track.artist_ids not in self.artist_to_tracks:
                        self.artist_to_tracks[track.artist_ids] = []
                    self.artist_to_tracks[track.artist_ids].append(track.track_id)
                except (KeyError, ValueError):
                    continue

        print(f"Loaded {len(self.tracks)} Spotify tracks.")

    def load_artists(self, csv_path: str) -> None:
        """Load Spotify artists from CSV."""
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    genres_str = row.get("genres", "")
                    genres = [g.strip() for g in genres_str.split(",") if g.strip()]

                    artist = SpotifyArtist(
                        artist_id=row["ids"],
                        name=row["names"],
                        monthly_listeners=float(row["monthly_listeners"])
                        if row["monthly_listeners"]
                        else 0,
                        popularity=float(row["popularity"]) if row["popularity"] else 0,
                        followers=float(row["followers"]) if row["followers"] else 0,
                        genres=genres,
                        first_release=row["first_release"],
                        last_release=row["last_release"],
                    )
                    self.artists[artist.artist_id] = artist
                except (KeyError, ValueError):
                    continue

        print(f"Loaded {len(self.artists)} Spotify artists.")

    def get_artist(self, artist_id: str) -> Optional[SpotifyArtist]:
        """Look up artist by ID."""
        return self.artists.get(artist_id)

    def get_similar_artists(
        self, artist_id: str, max_results: int = 3
    ) -> List[SpotifyArtist]:
        """Find artists with similar genres."""
        artist = self.get_artist(artist_id)
        if not artist or not artist.genres:
            return []

        similar = []
        for other_id, other in self.artists.items():
            if other_id == artist_id or not other.genres:
                continue
            overlap = len(set(artist.genres) & set(other.genres))
            if overlap > 0:
                similar.append((overlap, other))

        similar.sort(key=lambda x: x[0], reverse=True)
        return [artist for _, artist in similar[:max_results]]

    def get_artists_by_genre(self, genre: str, depth: int = 2) -> set:
        """Find artist IDs by exploring genres to a given depth."""
        artist_ids = set()
        current_genres = {genre.lower()}

        for _ in range(depth):
            next_genres = set()
            for artist in self.artists.values():
                artist_genres = {g.lower() for g in artist.genres}
                if artist_genres & current_genres:
                    artist_ids.add(artist.artist_id)
                    next_genres.update(artist_genres)

            current_genres = next_genres - current_genres

        return artist_ids

    def find_artist_by_name(self, artist_name: str) -> Optional[SpotifyArtist]:
        """Rough lookup: find artist matching name (case-insensitive)."""
        name_lower = artist_name.lower()
        for artist in self.artists.values():
            if artist.name.lower() == name_lower:
                return artist
        return None