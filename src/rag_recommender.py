import re
import math
from typing import List, Optional
from dataclasses import dataclass
from recommender import Song, UserProfile
from spotify_corpus import SpotifyCorpus, SpotifyTrack, SpotifyArtist
import google.generativeai as genai
import os
from dotenv import load_dotenv
import random
import json

load_dotenv()

CANDIDATE_LIMIT = 50 # larger than result for diversity
FEATURES = [
    'energy',
    'valence',
    'danceability',
    'acousticness',
    'tempo',
    'speechiness',
    'instrumentalness',
]

@dataclass
class UserRecommendation:
    """Info about a recommended song from the corpus."""
    song_name: str
    artist_info: Optional[str]
    genres: list
    popularity_tier: str
    explanation: str

class RAGRecommender:
    """
    RAG-based recommender that uses Spotify corpus to generate recommendations.

    Flow:
    1. Retrieve: Sample songs from Spotify corpus matching user's profile
       (favorite_genre, mood, energy, danceability, acousticness)
    2. Augment: Enrich with artist metadata (popularity, similar artists, genres)
    3. Generate: Use Gemini to select top matches + personalized explanations
    """

    def __init__(self, corpus: SpotifyCorpus):
        self.corpus = corpus
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def recommend(self, user_profile: UserProfile, k: int = 10, temperature: int=0.3) -> List[UserRecommendation]:
        """
        Generate recommendations for a user using RAG over Spotify corpus.
        Args:
            user_profile: UserProfile with favorite_genre, favorite_mood, energy, etc.
            k: Number of recommendations to return
        Returns:
            List of UserRecommendation objects (includes explanation from Gemini)
        """
        # Step 1: Retrieve candidate songs from corpus matching user profile
        candidates = self._retrieve_candidates(user_profile, k, temperature)

        # Step 2: Augment candidates with Spotify metadata (keyed by track ID)
        augmented_by_id = self._augment_candidates(candidates)

        # Step 3: Ask Gemini to rank and explain based on user profile (returns raw song ids)
        raw_recs = self._generate_recommendations(augmented_by_id, user_profile, temperature, k)

        # Step 4: Hydrate into UserRecommendation objects for user
        return self._hydrate_recommendations(raw_recs, augmented_by_id)


    def _retrieve_candidates(
        self, user_profile: UserProfile, num_candidates: int=CANDIDATE_LIMIT, temperature: int=0.3
    ) -> List[SpotifyTrack]:
        """
        Retrieve candidates using two strategies:
        1. Genre-based: artists matching user's favorite genre + similar artists
        2. Feature-based: songs matching ≥3 features within max_error
        """
       
        # based on temperature, bias candidate set to genre or features
        ngc = math.ceil(num_candidates * (1 - temperature))
        if not user_profile.favorite_genre: ngc = 0
        nfc = num_candidates - ngc

        # Strategy 1: Genre-based retrieval
        candidates_genre = set()
        if user_profile.favorite_genre:
            if temperature > 0.7: # discovery mode
                artist_candidates = self.corpus.get_artists_by_genre(user_profile.favorite_genre, depth=2)
            else:
                artist_candidates = self._retrieve_by_genre(user_profile.favorite_genre)

            # Get track IDs by these artists
            for track in self.corpus.tracks.values():
                artist_ids = [a.strip() for a in track.artist_ids.split(",") if a.strip()]
                if any(aid in artist_candidates for aid in artist_ids):
                    candidates_genre.add(track.track_id)

        print(f"n tracks by genre: {len(candidates_genre)}")

        # Strategy 2: Feature-based retrieval
        candidates_feature = set()
        max_error = 0.0
        while len(candidates_feature) < nfc and max_error<=1:
            max_error += 0.05 # if count doesn't reach threshold, loosen limits
            candidates_feature = self._retrieve_by_features(user_profile, max_error=max_error, min_match=3)
            candidates_feature -= candidates_genre
        print(f"n tracks by features: {len(candidates_feature)} at {max_error} limit")
       
        ngc = min(ngc, len(candidates_genre))
        nfc = min(nfc, len(candidates_feature))

        candidates_set = \
            random.sample(list(candidates_genre), ngc) + \
            random.sample(list(candidates_feature), nfc)

        # Convert back to track objects
        return [self.corpus.tracks[tid] for tid in candidates_set if tid in self.corpus.tracks]

    def _retrieve_by_genre(self, genre: str) -> set:
        """Retrieve artist IDs by genre using artist lookup."""
        artist_candidates = set()

        # Find artists with matching genre
        for artist in self.corpus.artists.values():
            if artist.genres and genre.lower() in [g.lower() for g in artist.genres]:
                artist_candidates.add(artist.artist_id)
                # Add similar artists
                similar = self.corpus.get_similar_artists(artist.artist_id, max_results=2)
                for sim_artist in similar:
                    artist_candidates.add(sim_artist.artist_id)

        return artist_candidates

    def _retrieve_by_features(self, user_profile: UserProfile, max_error: float = 0.2, min_match: int = 3) -> set:
        """Retrieve track IDs matching ≥min_match features within max_error."""
        
        # count number of features set in user_profile
        n_set = sum([ 1 for f in FEATURES if getattr(user_profile, f, None) != None])
        min_match = min(n_set, len(FEATURES))
        
        candidates = set()
        for track in self.corpus.tracks.values():
            matches = 0
            for feat_name in FEATURES:
                user_val = getattr(user_profile, feat_name, None)
                track_val = getattr(track, feat_name, None)
                if user_val is None or track_val is None: continue

                error = abs(track_val - user_val)
                if feat_name == 'tempo': error /= user_val

                if error <= max_error:
                    matches += 1

            if matches >= min_match:
                candidates.add(track.track_id)

        return candidates

    def _augment_candidates(self, candidates: List[SpotifyTrack]) -> dict:
        """Enrich candidates with artist metadata, keyed by track ID."""
        augmented = {}

        for track in candidates:
            artist_ids = track.artist_ids.split(",") if track.artist_ids else []
            artist_info = None

            if artist_ids:
                artist_id = artist_ids[0].strip()
                artist_info = self.corpus.get_artist(artist_id)

            popularity_tier = self._get_popularity_tier(artist_info)

            augmented[track.track_id] = {
                "track": track,
                "artist_info": artist_info,
                "popularity_tier": popularity_tier,
            }

        return augmented

    def _generate_recommendations(
        self, augmented_by_id: dict, user_profile: UserProfile, temperature: int,  k: int
    ) -> List[dict]:
        """Use Gemini to select top k recommendations + generate explanations."""

        # Format augmented tracks for Gemini
        tracks_text = self._format_tracks_for_gemini(augmented_by_id)

        # Build user preference description
        user_desc = self._compress_features(user_profile)

        prompt = f"""

You are a music recommender. Select the TOP {k} songs that best match this user's taste profile.
Select based primarily based on song features. only use artist popularity in the absence of other data,
or for a tiebreaker. Want some diversity to selection; don't select only the most popular songs

User feature preferences: {user_desc}
Temperature: {temperature} (0 is strong alignment with user preference, 1 is high diversity)

Here are {len(augmented_by_id)} candidate songs from the corpus:
They are formatted as [[track_id]] [song title] by [ [artist & popularity | None ] | [song features]

{tracks_text}

Return JSON array with exactly {k} objects:
[
  {{
    "track_id": "...",
    "reason": "1-sentence explanation (under 80 chars)"
  }},
  ...
]

Only return the JSON array, no other text."""

        try:
            print(f"querying gemini at T={temperature}...")
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=800),
            )
            text = response.text.strip()
            print(text)

            # Extract JSON
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = []

            return data[:k]

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

    def _format_tracks_for_gemini(self, augmented_by_id: dict) -> str:
        """Format augmented tracks as readable text for Gemini."""
        lines = []
        for i, (track_id, aug) in enumerate(list(augmented_by_id.items())[:CANDIDATE_LIMIT], 1):  # Limit to CANDIDATE_LIMIT for context
            track = aug["track"]
            artist = aug["artist_info"]
            artist_str = f"{artist.name} ({aug['popularity_tier']})" if artist else "Unknown"
            features = self._compress_features(track)

            lines.append(
                # Include track_id for Gemini to reference with full audio features
                f"{i}. [{track_id}] {track.name} by {artist_str} | {features}"
            )

        return "\n".join(lines)

    def _compress_features(self, obj):
        """Format features from object or dict for display."""
        features = []
        for f in FEATURES:
            value = getattr(obj, f, None)
            if value is not None:
                features.append(f"{f[0].upper()}:{value:.2f}")

        return " | ".join(features)

    def _get_popularity_tier(self, artist_info: Optional[SpotifyArtist]) -> str:
        """Determine artist popularity tier."""
        if not artist_info:
            return "unknown"

        listeners = artist_info.monthly_listeners
        if listeners > 5_000_000:
            return "major (5M+)"
        elif listeners > 1_000_000:
            return "established (1M+)"
        elif listeners > 100_000:
            return "rising (100K+)"
        else:
            return "niche"

    def _hydrate_recommendations(self, raw_recs: List[dict], augmented_by_id: dict) -> List[UserRecommendation]:
        """Convert raw recommendation data (track_id + reason) into UserRecommendation objects."""
        results = []
        for rec in raw_recs:
            track_id = rec.get("track_id", "")
            reason = rec.get("reason", "Recommended match")
            if track_id in augmented_by_id:
                aug = augmented_by_id[track_id]
                track = aug["track"]
                genres = aug["artist_info"].genres if aug["artist_info"] else []
                results.append(
                    UserRecommendation(
                        song_name=track.name,
                        artist_info=f"{aug['artist_info'].name} ({aug['popularity_tier']})" if aug["artist_info"] else "Unknown",
                        genres=genres[:3],
                        popularity_tier=aug["popularity_tier"],
                        explanation=reason,
                    )
                )
        return results