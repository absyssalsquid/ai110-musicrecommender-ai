# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Agentic Workflow (SF8)
> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**
Diversity and Fairness Logic

**Prompts used:**
describe different ways to apply scoring penalties to songs, for the purpose of preventing all recommendations from being too similar 

**What did the agent generate or change?**
refactored the code so that the function for calculating feature difference can be used in both song vs user and song vs song

**What did you verify or fix manually?**
added an inversion parameter to calculate similarity vs difference

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

Visual Summary Table

**Prompts used:**

in main.py, format the top recommendations into a table. 
columns: Song, Artist , Score w/o penalty, Similarity penalty, genre, mood, energy....
for each feature, mark o if it is high alignment, and an x if it is low alignment

**What did the agent generate or change?**
added a function to main.py to print the recommendations

**What did you verify or fix manually?**  
the legend format was originally messed up, requiired manual formatting

---