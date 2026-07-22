# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **Music Matcher 1.0**  

---

## 2. Intended Use  

Prompts:  

- What kind of recommendations does it generate  
    - makes music recommendations based on a defined user profile
- What assumptions does it make about the user
    - it assumes users already have a good idea of what they like. recommendations are essentially random otherwise
- Is this for real users or classroom exploration  
    - definitely more for just classroom use

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
    - genre, mood, energy, tempo_bpm, valence, danceability, acousticness, instrumentalness, speechiness
- What user preferences are considered
    - user preferences have all the same music features
- How does the model turn those into a score  
    - it calculates an absolute distance (L1) between the user features and song features, then applies weights to them
- What changes did you make from the starter logic  
    - added more weights and a penalty to encourage diversity in recommendations. 

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
    - 70
- What genres or moods are represented
    - 16 genres: pop, lofi, rock, ambient, jazz, synthwave, indie pop, hip hop, classical, edm, reggae, metal, funk, folk, drum and bass, country, gospel
    - 14 moods: happy, chill, intense, relaxed, focused, moody, melancholy, groovy, energetic, nostalgic, uplifting, aggressive, triumphant, hypnotic, romantic
- Did you add or remove data  
    - added a lot of data
- Are there parts of musical taste missing in the dataset 
    - yes, the data is rather anglo focused. missing genres include: blues, R&B, latin/reggaeton, kpop. also missing. indie/alt rock and punk.

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
    - pop/energetic/positive, mellow acoustic, ambient/lofi, metal 
- Any patterns you think your scoring captures correctly
    -  the diversity penalty results in a nice oscillation in scores selected songs. like 1,3,2,6,4. if there were a large enough library i think this would be good to listen to
- Cases where the recommendations matched your intuition  
    - if there are enough songs to recommend, most were pretty good 

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
    - lyrics/language, release year, loudness, duration
- Genres or moods that are underrepresented  
    - missing genres include: blues, R&B, latin/reggaeton, kpop, indie/alt rock and punk.
- Cases where the system overfits to one preference 
    - for the genres that are over represented in the data, those genres show up randomly in the recommendations for users where there arent enough songs to satisfy their preferences
- Ways the scoring might unintentionally favor some users 
    - for genres that are overrepresented in the data, those users will always get music they like.

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
    - created several profiles, they are listed in the profiles csv
- What you looked for in the recommendations
    - whether if i were in the mood aligning with that profile, that i would like that selection of songs 
- What surprised you  
    - there's a gospel song that keeps showing up unexpectedly for different profiles. need to look into that
- Any simple tests or comparisons you ran
    - enabling/disabling some features, changing weights

No need for numeric metrics unless you created some.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
    - songs can have multiple categories, release year, language(s)
    - tastes changing over time
- Better ways to explain recommendations  
    - tabular rather than the string format given in the starter code
- Improving diversity among the top results  
    - more complex diversity algorithm, but also requires user 
- Handling more complex user tastes  
    - users can have multiple favorite genres and moods
    - users can weight some features more strongy

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
    - there are so many data features to look at, and even more ways to look at them
- Something unexpected or interesting you discovered  
    - there a lot of mathetmatical models and algorithms to consider that i would like to do research into
- How this changed the way you think about music recommendation apps
    - way more goes into them than i thought. definitely a big data sort of thing
