# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **Music Matcher 2.0**  

---
## AI collaborations 

### use during development
AI was used to brainstorm what AI feature was best to implement for this project, and general debugging. 

### helpful suggestion
was good for getting the project set up, creating the mmd diagram

### flawed suggestion
this assignment seemed to be a hard one for claude to help with. it often misunderstood what i was asking or identified code as buggy when it wasnt. 

## System limitations
due to the token and context limits of the free tier of gemini, the input data had to be cut down significantly before being given to the AI. As a result, many songs are left out of consideration. The recommender gives recommendations from a subset of the data rather than the whole corpus. Hopefully, the pre-filter did a sufficient job of filtering out the not-so-good songs so that of the data presented, it is at least better than the rest.
It is also hard to know exactly how the AI is selecting songs. Testing for this would be difficult in itself.