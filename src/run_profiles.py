"""
Test harness for the recommender's discriminatory ability.

For each profile in data/test_profiles.csv this ranks ALL songs by pure
relevance distance (no diversity penalty) and reports:
  - whether the #1 song matches expected_top (PASS/FAIL)
  - the winning score
  - the confidence margin: how far ahead #1 is of #2
    (a small margin = the algorithm barely discriminated)

Run from the project root:  python src/run_profiles.py
"""

import csv

from recommender import load_songs, load_profiles, Recommender

LOW_MARGIN = 0.05  # below this, #1 and #2 are effectively tied

PROFILES_CSV = "data/test_profiles.csv"


def load_profile_meta(csv_path):
    """Read the test-only columns (name/expected_top/tests) in CSV order.

    load_profiles() returns UserProfile objects, which don't carry test
    metadata — so we read those columns here and zip by position.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            {"name": row.get("name"),
             "expected_top": row.get("expected_top"),
             "tests": row.get("tests")}
            for row in csv.DictReader(f)
        ]


def ranked_scores(rec, profile):
    """All songs scored by pure relevance (no penalty), ascending — best first."""
    uf = rec.extract_features(profile)
    scored = [
        (Recommender.score(rec.song_vs_user_distance(sid, uf)), rec.songs_by_id[sid].title)
        for sid in rec.songs_by_id
    ]
    scored.sort(key=lambda x: x[0])
    return scored


def main() -> None:
    songs = load_songs("data/songs.csv")
    rec = Recommender(songs)
    profiles = load_profiles(PROFILES_CSV)
    meta = load_profile_meta(PROFILES_CSV)

    print("\n" + "=" * 74)
    print("DISCRIMINATION TEST — pure-relevance ranking, top match + confidence")
    print("=" * 74)

    passed = 0
    low_margin_count = 0
    for profile, p in zip(profiles, meta):
        ranked = ranked_scores(rec, profile)
        (top_score, top_title) = ranked[0]
        (second_score, second_title) = ranked[1]
        margin = second_score - top_score

        ok = top_title == p["expected_top"]
        passed += ok
        tight = margin < LOW_MARGIN
        low_margin_count += tight

        status = "PASS" if ok else "FAIL"
        flag = "  <-- LOW MARGIN" if tight else ""
        print(f"\n[{status}] {p['name']}{flag}")
        print(f"       expected : {p['expected_top']}")
        print(f"       #1       : {top_title}  (score {top_score:.3f})")
        print(f"       #2       : {second_title}  (score {second_score:.3f})")
        print(f"       margin   : {margin:.3f}   ({p['tests']})")
        if not ok:
            print(f"       full top-3: " + " | ".join(f"{t} {s:.2f}" for s, t in ranked[:3]))

    print("\n" + "=" * 74)
    print(f"RESULT: {passed}/{len(profiles)} matched expected top   "
          f"| {low_margin_count} decided by a low margin (<{LOW_MARGIN})")
    print("=" * 74)


if __name__ == "__main__":
    main()
