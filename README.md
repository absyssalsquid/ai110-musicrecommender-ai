# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
   - energy, tempo_bpm, valence, danceability, acousticness, instrumentalness, speechiness
- What information does your `UserProfile` store
  - it stores all the same features that a song has
- How does your `Recommender` compute a score for each song
  - for each feature, it calculates a weighted distance between user preferences and the song
- How do you choose which songs to recommend
  - Select songs that are most in alignment with the user preference (lowest distance)

<!-- You can include a simple diagram or bullet list if helpful. -->

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:

    ```bash
    python -m src.main
    ```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
============================================================
USER PROFILE
============================================================
  genre           : pop
  mood            : happy
  energy          : 0.80
  tempo           : 110
  valence         : 0.70
  danceability    : 0.66
  acousticness    : 0.40
  instrumentalness: 0.02
  speechiness     : 0.80

============================================================
Top recommendations:
============================================================
Song               | Artist      | Score | Penalty | gen | mod | enr | tmp | val | dnc | acu | ins | spc
-------------------+-------------+-------+---------+-----+-----+-----+-----+-----+-----+-----+-----+----
Sunrise City       | Neon Echo   | 0.39  | -       | o   | o   | o   | o   | o   | o   | o   | o   | ·  
Gym Hero           | Max Pulse   | 1.48  | 2.09    | o   | x   | o   | o   | o   | o   | ·   | o   | ·  
Concrete Verses    | Blockprint  | 4.24  | 2.88    | x   | x   | o   | o   | ·   | o   | ·   | o   | o  
Spacewalk Thoughts | Orbit Bloom | 5.88  | 2.15    | x   | x   | ·   | ·   | o   | ·   | ·   | ·   | ·  
Iron Cathedral     | Ashen Crown | 4.93  | 4.54    | x   | x   | ·   | ·   | ·   | o   | ·   | o   | ·  

legend:
o = aligned (<0.05)
x = unaligned (>0.95)
· = partial

gen=genre mod=mood enr=energy tmp=tempo val=valence dnc=danceability acu=acousticness ins=instrumentalness spc=speechiness
```

<!-- **Screenshot or video** *(optional)*: Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran.
- Reducing weight on genre from 2.0 to 0.8
  - for some profiles the output seemed better and more diverse, while remaining reasonably coherent, but for others the recommendations seemed nonsensical. possibly due to small data
- Removing energy and valence
  - caused a lot of the profiles that had strong energy and valence preferences to become nonsensical

---

## Limitations and Risks

Summarize some limitations of your recommender.
- No lyrics or language
- no release year
- recommendations are in a bubble. small amount for diveristy penalty, but still very close to user profile. Never comes up with something unexpected that a user may still like.
- no feedback function. cannot evolve over time with use
- hand tuned weights cannot learn what a user prefers more, e.g. if they find acousticness more important than other factors
- catalog is very small, and is insufficiently diverse for the music that exists

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
  - There are a relatively limited number of factors with which to categorize a song, but the way that they can be processed is infinite. It requires a lot of research to determining what extant algorithms to implement, or what mathematical models to use. The input data is also very important, because if the input data is not good, even if the algorithm is good, the outputs will not be good. Strong biases in the inputs will show up in the ouputs, even if the algorithm itself is not biased. It is also quite difficult to determine how exactly a large amount of data is interacting with the algorithm itself.


