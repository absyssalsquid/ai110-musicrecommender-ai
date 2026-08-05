# RAG Layer for Music Recommender

## Overview

**Retrieval-Augmented Generation (RAG)** enhances your distance-based recommender by:
1. **Retrieving** context about songs/artists from your 15K Spotify corpus
2. **Augmenting** recommendation scores with metadata (popularity, similar artists, genres, trends)
3. **Generating** human-readable explanations using Claude

This turns mechanical scores into **explainable, contextual recommendations**.

---

## High-Level Flow

```
User Profile (pop/happy/energetic)
         │
         ▼
Distance-Based Recommender
(finds 5 songs with lowest weighted distance)
         │
         ▼
RAG Layer
├─ Retrieve: Look up artist popularity, genres, release trends from Spotify corpus
├─ Augment: Boost scores based on context (e.g., +0.2 if similar to favorite artist)
└─ Generate: Use Claude to write explanation: "Neon Echo (2.3M listeners, Synthwave)
              shares 4 genres with Depeche Mode..."
         │
         ▼
Enhanced Recommendations
(song, original_score, rag_score, explanation)
```

---

## Implementation Plan

### Part 1: Load Spotify Corpus & Build Index

**New file: `src/spotify_corpus.py`**

```python
import csv
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SpotifyTrack:
    track_id: str
    name: str
    artist_ids: str  # comma-separated
    popularity: float
    genres: List[str]
    release_date: str
    danceability: float
    energy: float
    valence: float
    acousticness: float
    # ... other audio features

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
        self.genre_to_tracks: Dict[str, List[str]] = {}  # genre -> [track_ids]
        self.artist_to_tracks: Dict[str, List[str]] = {}  # artist -> [track_ids]
    
    def load_tracks(self, csv_path: str) -> None:
        """Load Spotify tracks from CSV."""
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                track = SpotifyTrack(
                    track_id=row["ids"],
                    name=row["names"],
                    artist_ids=row["artists"],  # may be comma-separated
                    popularity=float(row["popularity"]) if row["popularity"] else 0,
                    genres=self._parse_genres(row.get("playlists_found", "")),
                    release_date=row["release_date"],
                    danceability=float(row["danceability"]) if row["danceability"] else 0,
                    energy=float(row["energy"]) if row["energy"] else 0,
                    valence=float(row["valence"]) if row["valence"] else 0,
                    acousticness=float(row["acousticness"]) if row["acousticness"] else 0,
                )
                self.tracks[track.track_id] = track
                
                # Index by artist
                if track.artist_ids not in self.artist_to_tracks:
                    self.artist_to_tracks[track.artist_ids] = []
                self.artist_to_tracks[track.artist_ids].append(track.track_id)
    
    def load_artists(self, csv_path: str) -> None:
        """Load Spotify artists from CSV."""
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                genres_str = row.get("genres", "")
                genres = [g.strip() for g in genres_str.split(",") if g.strip()]
                
                artist = SpotifyArtist(
                    artist_id=row["ids"],
                    name=row["names"],
                    monthly_listeners=float(row["monthly_listeners"]) if row["monthly_listeners"] else 0,
                    popularity=float(row["popularity"]) if row["popularity"] else 0,
                    followers=float(row["followers"]) if row["followers"] else 0,
                    genres=genres,
                    first_release=row["first_release"],
                    last_release=row["last_release"],
                )
                self.artists[artist.artist_id] = artist
                
                # Index by genre
                for genre in genres:
                    if genre not in self.genre_to_tracks:
                        self.genre_to_tracks[genre] = []
    
    def get_artist(self, artist_id: str) -> Optional[SpotifyArtist]:
        """Look up artist by ID."""
        return self.artists.get(artist_id)
    
    def get_similar_artists(self, artist_id: str, max_results: int = 3) -> List[SpotifyArtist]:
        """Find artists with similar genres."""
        artist = self.get_artist(artist_id)
        if not artist:
            return []
        
        similar = []
        for other_id, other in self.artists.items():
            if other_id == artist_id:
                continue
            # Count overlapping genres
            overlap = len(set(artist.genres) & set(other.genres))
            if overlap > 0:
                similar.append((overlap, other))
        
        # Sort by overlap (descending) and take top N
        similar.sort(reverse=True)
        return [artist for _, artist in similar[:max_results]]
    
    def get_tracks_by_genre(self, genre: str, max_results: int = 5) -> List[SpotifyTrack]:
        """Find popular tracks in a genre."""
        track_ids = self.genre_to_tracks.get(genre, [])
        tracks = [self.tracks[tid] for tid in track_ids if tid in self.tracks]
        # Sort by popularity
        tracks.sort(key=lambda t: t.popularity, reverse=True)
        return tracks[:max_results]
    
    def _parse_genres(self, playlist_str: str) -> List[str]:
        """Extract genre info from playlist name (hacky for now)."""
        # Could improve: parse from playlists_found column
        return []
```

---

### Part 2: RAG Context Builder

**New file: `src/rag_context.py`**

```python
from typing import List, Dict
from recommender import Song
from spotify_corpus import SpotifyCorpus, SpotifyArtist
import anthropic

@dataclass
class UserRecommendation:
    """Context about a recommended song from the corpus."""
    song: Song
    original_score: float
    artist_info: Optional[SpotifyArtist]
    similar_artists: List[SpotifyArtist]
    genre_context: str  # text summary
    popularity_tier: str  # "trending", "established", "niche"
    explanation: str  # Claude-generated

class RAGContextBuilder:
    """Builds context for recommendations using the Spotify corpus."""
    
    def __init__(self, corpus: SpotifyCorpus):
        self.corpus = corpus
        self.client = anthropic.Anthropic()
    
    def build_context(self, song: Song, original_score: float, 
                      user_favorite_artist: str = None) -> UserRecommendation:
        """
        Gather context about a song from the corpus.
        """
        # Try to find artist info (rough lookup by name since we don't have track->artist mapping)
        artist_info = self._find_artist_by_name(song.artist)
        
        similar_artists = []
        if artist_info:
            similar_artists = self.corpus.get_similar_artists(artist_info.artist_id, max_results=3)
        
        # Generate genre context
        genre_context = self._build_genre_context(song.genre, artist_info)
        
        # Determine popularity tier
        if artist_info:
            listeners = artist_info.monthly_listeners
            if listeners > 1_000_000:
                tier = "established (1M+ monthly listeners)"
            elif listeners > 100_000:
                tier = "rising (100K-1M monthly listeners)"
            else:
                tier = "niche (< 100K monthly listeners)"
        else:
            tier = "unknown"
        
        # Generate explanation with Claude
        explanation = self._generate_explanation(
            song, original_score, artist_info, similar_artists, tier, user_favorite_artist
        )
        
        return UserRecommendation(
            song=song,
            original_score=original_score,
            artist_info=artist_info,
            similar_artists=similar_artists,
            genre_context=genre_context,
            popularity_tier=tier,
            explanation=explanation,
        )
    
    def _find_artist_by_name(self, artist_name: str) -> Optional[SpotifyArtist]:
        """Rough lookup: find artist matching name (case-insensitive)."""
        name_lower = artist_name.lower()
        for artist in self.corpus.artists.values():
            if artist.name.lower() == name_lower:
                return artist
        return None
    
    def _build_genre_context(self, song_genre: str, artist_info: Optional[SpotifyArtist]) -> str:
        """Create a text summary of genre context."""
        if not artist_info:
            return f"Genre: {song_genre}"
        
        genres_str = ", ".join(artist_info.genres[:3])  # Top 3 genres
        return f"{artist_info.name} is tagged as: {genres_str}"
    
    def _generate_explanation(self, song: Song, score: float, 
                             artist_info: Optional[SpotifyArtist],
                             similar_artists: List[SpotifyArtist],
                             popularity_tier: str,
                             user_favorite_artist: str) -> str:
        """Use Claude to write a human-readable explanation."""
        
        artist_summary = ""
        if artist_info:
            artist_summary = f"{artist_info.name} ({popularity_tier})"
        else:
            artist_summary = f"{song.artist} ({popularity_tier})"
        
        similar_summary = ""
        if similar_artists:
            names = ", ".join(a.name for a in similar_artists)
            similar_summary = f"Similar artists: {names}"
        
        favorite_context = ""
        if user_favorite_artist:
            favorite_context = f"You like {user_favorite_artist}."
        
        prompt = f"""
You are explaining why a song was recommended. Be concise and engaging (2 sentences max).

Song: "{song.title}" by {artist_summary}
Genre: {song.genre}, Mood: {song.mood}
Match Score: {score:.2f}/10 (lower is better fit)

{favorite_context}
{similar_summary}

Write a brief explanation for why this song was recommended. Focus on:
- Genre/mood fit
- Artist context (popularity, similar artists if relevant)
- Why the user might like it

Keep it under 2 sentences.
"""
        
        response = self.client.messages.create(
            model="claude-opus-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
```

---

### Part 3: Enhanced Recommender with RAG

**Update: `src/recommender.py`**

Add this method to the `Recommender` class:

```python
def recommend_with_rag(self, user: UserProfile, k: int = 5, 
                       rag_context_builder = None) -> List[UserRecommendation]:
    """
    Recommend songs with RAG augmentation.
    Returns UserRecommendation objects with explanations.
    """
    # Get base recommendations
    base_recs = self.recommend_plus(user, k=k)
    
    # If no RAG, return as-is
    if rag_context_builder is None:
        return base_recs
    
    # Build RAG context for each recommendation
    rag_recs = []
    for song, score, distance in base_recs:
        context = rag_context_builder.build_context(
            song, 
            score,
            user_favorite_artist=user.favorite_genre  # or extract artist name
        )
        rag_recs.append(context)
    
    return rag_recs
```

---

### Part 4: CLI Integration

**Update: `src/main.py`**

```python
from spotify_corpus import SpotifyCorpus
from rag_context import RAGContextBuilder

def main_with_rag() -> None:
    songs = load_songs("data/songs.csv")
    rec = Recommender(songs)
    
    # Load Spotify corpus
    corpus = SpotifyCorpus()
    corpus.load_tracks("data/featured_Spotify_track_info.csv")
    corpus.load_artists("data/featured_Spotify_artist_info.csv")
    
    # Initialize RAG builder
    rag_builder = RAGContextBuilder(corpus)
    
    profiles = load_profiles("data/test_profiles.csv")
    
    for profile in profiles:
        print_profile_summary(profile)
        print()
        
        # Get recommendations WITH RAG
        rag_recommendations = rec.recommend_with_rag(profile, k=5, 
                                                      rag_context_builder=rag_builder)
        
        print_rag_recommendations_table(rag_recommendations)
        print()

def print_rag_recommendations_table(rag_recs: List[UserRecommendation]) -> None:
    """Print enhanced recommendations with RAG context."""
    print("=" * 80)
    print("Top recommendations (with context):")
    print("=" * 80)
    
    for i, rec in enumerate(rag_recs, 1):
        song = rec.song
        print(f"\n{i}. {song.title} by {song.artist}")
        print(f"   Score: {rec.original_score:.2f} | Popularity: {rec.popularity_tier}")
        print(f"   Genre: {rec.genre_context}")
        if rec.similar_artists:
            print(f"   Similar: {', '.join(a.name for a in rec.similar_artists)}")
        print(f"   Why: {rec.explanation}")

if __name__ == "__main__":
    main_with_rag()
```

---

## What This Delivers (in 4 hours)

✅ **Load & index 15K Spotify songs/artists**  
✅ **Retrieve artist/genre context for each recommendation**  
✅ **Generate Claude-powered explanations**  
✅ **Print enhanced recommendations with context**  
✅ **Before/after comparison**: distance-only vs. RAG-augmented  

---

## Sample Output

**Before (distance-only):**
```
Song               | Artist      | Score | Penalty
-------------------+-------------+-------+--------
Sunrise City       | Neon Echo   | 0.39  | -
```

**After (with RAG):**
```
1. Sunrise City by Neon Echo
   Score: 0.39 | Popularity: established (2.3M monthly listeners)
   Genre: Synthwave, Darkwave, Electronic
   Similar: Perturbator, Carpenter Brut, Gost
   Why: Neon Echo's Synthwave sound matches your love of energetic, moody music.
        The artist has gained 40% more listeners in the past year.
```

---

## Why This Is RAG

- **Retrieval**: Look up artist/genre/popularity from 15K corpus ✓
- **Augmentation**: Add context that improves the recommendation ✓
- **Generation**: Use Claude to synthesize into natural language ✓

This is a **lightweight RAG** (not vector embeddings or semantic search), but it's practical and explainable.

---

## Stretch Ideas (if you have time)

1. **Vector embeddings**: Use Claude's embeddings API to do semantic genre/mood search
2. **Trend analysis**: Compare release dates, find "rising" artists in user's taste profile
3. **Collaborative context**: "Users who liked X also listened to Y"
4. **A/B testing**: Show recommendations with/without explanations to measure engagement