"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, load_profiles, Recommender
from spotify_corpus import SpotifyCorpus
from rag_recommender import RAGRecommender


def main() -> None:
    songs = load_songs("data/songs.csv")
    rec = Recommender(songs)

    profiles = load_profiles("data/test_profiles.csv")

    for profile in profiles:
        print_profile_summary(profile)
        print()

        recommendations = rec.recommend_plus(profile, k=5)
        print_recommendations_table(recommendations)
        print()


def main_with_rag() -> None:
    """Run recommender with RAG augmentation."""
    print("Loading Spotify corpus...")
    corpus = SpotifyCorpus()
    corpus.load_tracks("data/featured_Spotify_track_info.csv")
    corpus.load_artists("data/featured_Spotify_artist_info.csv")

    rag_recommender = RAGRecommender(corpus)

    profiles = load_profiles("data/test_profiles.csv")

    for profile in profiles[:2]:
        print_profile_summary(profile)
        print()

        # Test candidate retrieval only
        # candidates = rag_recommender._retrieve_candidates(profile, num_candidates=100, temperature=0.9)
        # print(f"Retrieved {len(candidates)} candidates")
        # augmented_by_id = rag_recommender._augment_candidates(candidates)
        # tracks_text = rag_recommender._format_tracks_for_gemini(augmented_by_id)
        # print("Formatted tracks for Gemini:")
        # print(tracks_text[:500])
        # print()

        rag_recommendations = rag_recommender.recommend(profile, k=5)
        print_rag_recommendations_table(rag_recommendations)
        print()

# -------------------------- without RAG ---------------------------------

def print_profile_summary(profile) -> None:
    """Print the user profile the recommendations are based on."""
    fields = [
        ("name", "name"), ("genre", "genre"), ("mood", "mood"), ("energy", "energy"),
        ("tempo_bpm", "tempo"), ("valence", "valence"), ("danceability", "danceability"),
        ("acousticness", "acousticness"), ("instrumentalness", "instrumentalness"),
        ("speechiness", "speechiness"),
    ]

    print("=" * 60)
    print("USER PROFILE")
    print("=" * 60)
    for key, label in fields:
        val = getattr(profile, key, None)
        if val is None:
            shown = "(any)"
        elif isinstance(val, float):
            shown = f"{val:.2f}"
        else:
            shown = str(val)
        print(f"  {label:<16}: {shown}")

def print_recommendations_table(recommendations) -> None:
    """
    Render recommend_plus() output as a fixed-width table with one column per
    feature. Each feature cell shows:
        o  high alignment (distance < 0.05)
        x  low alignment  (distance > 0.95)
        ·  in between
    """
    # (feature key, short column header)
    features = [
        ("genre", "gen"), ("mood", "mod"), ("energy", "enr"), ("tempo_norm", "tmp"),
        ("valence", "val"), ("danceability", "dnc"), ("acousticness", "acu"),
        ("instrumentalness", "ins"), ("speechiness", "spc"),
    ]

    headers = ["Song", "Artist", "Score", "Penalty"] + [label for _, label in features]

    rows = []
    for song, score_no_penalty, distance in recommendations:
        penalty = distance.get("penalty", 0.0)

        marks = []
        for key, _ in features:
            v = distance.get(key)
            if v is None:  marks.append("-")      # feature not compared (unset preference)
            elif v < 0.05: marks.append("o")      # high alignment
            elif v > 0.95: marks.append("x")      # low alignment
            else:          marks.append("·")      # partial

        rows.append([
            song.title,
            song.artist,
            f"{score_no_penalty:.2f}",
            f"{penalty:.2f}" if penalty else "-",
            *marks,
        ])

    # Column widths sized to the widest cell (header or value) in each column.
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("=" * 60)
    print("Top recommendations:")
    print("=" * 60)
    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    print("\nlegend:\no = aligned (<0.05)\nx = unaligned (>0.95)\n· = partial")
    print("\ngen=genre mod=mood enr=energy tmp=tempo val=valence "
          "dnc=danceability acu=acousticness ins=instrumentalness spc=speechiness")


# -------------------------- with RAG ---------------------------------

def print_rag_recommendations_table(recs) -> None:
    """Print recommendations from RAG recommender."""
    print("=" * 80)
    print("Top recommendations (RAG-powered from Spotify corpus):")
    print("=" * 80)

    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec.song_name} by {rec.artist_info}")
        print(f"   Genres: {', '.join(rec.genres)}")
        print(f"   Popularity: {rec.popularity_tier}")
        print(f"   ? {rec.explanation}")


if __name__ == "__main__":
    # Uncomment one:
    # main()
    main_with_rag()  # Run with RAG augmentation
