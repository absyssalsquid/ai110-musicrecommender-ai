# Agentic Loop Design for Music Recommender

## Overview

This doc outlines how to add an agentic feedback loop to your recommender. The agent will:
1. **Observe**: Collect feedback on recommendations (thumbs up/down, ratings)
2. **Analyze**: Identify patterns in what worked and what didn't
3. **Hypothesize**: Propose new weight adjustments based on the feedback
4. **Test**: Validate the new weights against held-out profiles
5. **Iterate**: Repeat until improvements plateau

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User provides feedback on recommendations                   │
│  e.g., "liked: Sunrise City (pop/happy)"                    │
│       "disliked: Concrete Verses (hip-hop/sad)"             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Feedback Collector (new module)                             │
│  - Stores (user_profile, song, rating)                      │
│  - Maintains feedback history                               │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Claude Agent (agentic loop)                                │
│  - Analyzes feedback vs. current weights                     │
│  - Generates hypotheses about misalignment                   │
│  - Proposes new weight configurations                        │
│  - Runs A/B tests on validation set                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Weight Optimizer (new module)                              │
│  - Tests proposed weights                                   │
│  - Measures impact on validation profiles                    │
│  - Returns metrics (e.g., % of recommendations liked)        │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Updated Recommender                                         │
│  - Uses new weights                                          │
│  - Generates next batch of recommendations                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Feedback Collection

**New file: `src/feedback.py`**

```python
@dataclass
class Feedback:
    """Record of user feedback on a recommendation."""
    user_name: str
    song_id: int
    song_title: str
    rating: float  # 0.0 (disliked) to 1.0 (loved)
    timestamp: datetime
    explanation: str  # optional, e.g., "too energetic"

class FeedbackStore:
    """Accumulates feedback across sessions."""
    def __init__(self):
        self.feedback: List[Feedback] = []
    
    def add(self, user_name: str, song_id: int, song_title: str, 
            rating: float, explanation: str = "") -> None:
        """Record feedback."""
        self.feedback.append(Feedback(
            user_name=user_name,
            song_id=song_id,
            song_title=song_title,
            rating=rating,
            timestamp=datetime.now(),
            explanation=explanation,
        ))
    
    def get_by_user(self, user_name: str) -> List[Feedback]:
        """Retrieve all feedback for a user."""
        return [f for f in self.feedback if f.user_name == user_name]
    
    def get_stats(self) -> Dict:
        """Summary: % liked, common complaints, etc."""
        if not self.feedback:
            return {"total": 0}
        
        liked = sum(1 for f in self.feedback if f.rating >= 0.7)
        return {
            "total": len(self.feedback),
            "liked_pct": 100 * liked / len(self.feedback),
            "avg_rating": sum(f.rating for f in self.feedback) / len(self.feedback),
        }
```

**Integration point in `main.py`:**
```python
store = FeedbackStore()

for profile in profiles:
    recommendations = rec.recommend_plus(profile, k=5)
    print_recommendations_table(recommendations)
    
    # Collect feedback (could be interactive or simulated)
    for song, score, distance in recommendations:
        # In interactive mode: user provides rating
        # In simulation: agent simulates based on profile fit
        rating = simulate_user_feedback(profile, song, distance)
        store.add(profile.name, song.id, song.title, rating)
```

---

## Phase 2: Feedback Analysis & Hypothesis Generation

**New file: `src/agent_loop.py`**

The agent reads feedback and proposes weight changes:

```python
from anthropic import Anthropic

class RecommenderAgent:
    def __init__(self, recommender: Recommender, feedback_store: FeedbackStore):
        self.recommender = recommender
        self.feedback_store = feedback_store
        self.client = Anthropic()
        self.current_weights = recommender.weights.copy()
    
    def analyze_and_propose(self) -> Dict:
        """
        Use Claude to analyze feedback and propose new weights.
        Returns a dict with:
            - analysis: text explanation
            - proposed_weights: new weight config
            - rationale: why these changes
        """
        stats = self.feedback_store.get_stats()
        
        prompt = f"""
You are tuning a music recommender system. Currently:
- % of recommendations users liked: {stats['liked_pct']:.1f}%
- Average rating: {stats['avg_rating']:.2f}

Current weights (lower = prioritize more):
{json.dumps(self.current_weights, indent=2)}

Feedback summary (disliked recommendations):
[Show sample of low-rated recommendations and their feature mismatches]

Based on this, propose a NEW set of weights that would likely improve the system.
Focus on:
1. Which features are causing misalignment?
2. Should certain features matter more or less?
3. Are there emergent patterns?

Return your proposal as a JSON block with:
{
  "analysis": "what you noticed",
  "proposed_weights": { ... },
  "rationale": "why you made these changes"
}
"""
        
        response = self.client.messages.create(
            model="claude-opus-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response (extract JSON)
        return self.extract_json(response.content[0].text)
    
    def extract_json(self, text: str) -> Dict:
        """Pull JSON from Claude's response."""
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No JSON found in response")
```

---

## Phase 3: Weight Validation

**New file: `src/weight_optimizer.py`**

Test proposed weights on a held-out validation set:

```python
class WeightValidator:
    def __init__(self, recommender: Recommender, validation_profiles: List[UserProfile]):
        self.recommender = recommender
        self.validation_profiles = validation_profiles
    
    def test_weights(self, new_weights: Dict, old_weights: Dict) -> Dict:
        """
        Compare old vs. new weights on validation set.
        Returns metrics showing if new weights are better.
        """
        metrics = {"old": {}, "new": {}}
        
        for weights, label in [(old_weights, "old"), (new_weights, "new")]:
            self.recommender.weights = weights
            
            aligned_count = 0
            total_recs = 0
            
            for profile in self.validation_profiles:
                recs = self.recommender.recommend_plus(profile, k=5)
                
                # Measure alignment: how many recs have mostly low distance?
                for song, score, distance in recs:
                    aligned = sum(1 for d in distance.values() 
                                 if isinstance(d, float) and d < 0.3)
                    aligned_count += aligned
                    total_recs += len(distance)
            
            metrics[label]["alignment_score"] = aligned_count / total_recs if total_recs else 0
        
        improvement = (metrics["new"]["alignment_score"] - 
                      metrics["old"]["alignment_score"])
        
        return {
            "improved": improvement > 0.02,  # threshold: 2% improvement
            "improvement_pct": improvement * 100,
            "new_metrics": metrics["new"],
            "old_metrics": metrics["old"],
        }
```

---

## Phase 4: Integration & Iteration Loop

**New file: `src/agentic_recommender.py`**

Orchestrates the full loop:

```python
class AgenticRecommender:
    def __init__(self, recommender: Recommender, 
                 feedback_store: FeedbackStore,
                 validation_profiles: List[UserProfile]):
        self.recommender = recommender
        self.feedback_store = feedback_store
        self.agent = RecommenderAgent(recommender, feedback_store)
        self.validator = WeightValidator(recommender, validation_profiles)
        self.iteration = 0
    
    def run_iteration(self) -> Dict:
        """One full loop: analyze → propose → test → update."""
        print(f"\n=== Iteration {self.iteration + 1} ===")
        
        # Step 1: Analyze feedback
        proposal = self.agent.analyze_and_propose()
        print(f"Agent analysis: {proposal['analysis']}")
        print(f"Proposed weights: {proposal['proposed_weights']}")
        
        # Step 2: Validate
        validation = self.validator.test_weights(
            proposal["proposed_weights"],
            self.recommender.weights
        )
        
        print(f"Validation result: {validation['improvement_pct']:.2f}% improvement")
        
        # Step 3: Accept or reject
        if validation["improved"]:
            self.recommender.weights = proposal["proposed_weights"]
            print("✓ Weights updated!")
        else:
            print("✗ No improvement, keeping old weights.")
        
        self.iteration += 1
        return validation
    
    def run_n_iterations(self, n: int = 5) -> None:
        """Run multiple iterations."""
        for _ in range(n):
            self.run_iteration()
```

---

## Phase 5: CLI Integration

**Update `main.py`:**

```python
def main_with_agent_loop():
    songs = load_songs("data/songs.csv")
    recommender = Recommender(songs)
    
    profiles = load_profiles("data/test_profiles.csv")
    validation_profiles = profiles[:3]  # Hold out first 3 for validation
    training_profiles = profiles[3:]
    
    feedback_store = FeedbackStore()
    agentic_rec = AgenticRecommender(recommender, feedback_store, validation_profiles)
    
    # Initial pass: collect feedback
    print("=== INITIAL RECOMMENDATIONS ===")
    for profile in training_profiles:
        recs = recommender.recommend_plus(profile, k=5)
        print_recommendations_table(recs)
        
        # Simulate user feedback based on alignment
        for song, score, distance in recs:
            # Simple heuristic: better alignment = higher rating
            alignment = 1.0 - (score / 10.0)  # normalize
            rating = max(0.0, min(1.0, alignment + random.gauss(0, 0.1)))
            feedback_store.add(profile.name, song.id, song.title, rating)
    
    # Run agent loop
    print("\n=== AGENT TUNING LOOP ===")
    agentic_rec.run_n_iterations(n=3)
    
    # Final recommendations with updated weights
    print("\n=== FINAL RECOMMENDATIONS (with tuned weights) ===")
    for profile in training_profiles:
        recs = recommender.recommend_plus(profile, k=5)
        print_recommendations_table(recs)
```

---

## Summary: What You'll Build

| Component | Purpose |
|-----------|---------|
| **FeedbackStore** | Accumulate user feedback (likes/dislikes) |
| **RecommenderAgent** | Use Claude to analyze patterns and propose weight changes |
| **WeightValidator** | Test proposed weights on validation set |
| **AgenticRecommender** | Orchestrate the full loop (analyze → propose → test → update) |
| **CLI runner** | Display iterations, show before/after metrics |

---

## Why This Matters

- **Autonomy**: The agent doesn't need you to manually tweak weights—it learns from feedback
- **Traceability**: You see the agent's reasoning (analysis + rationale) at each step
- **Validation**: New weights are tested before adoption, preventing regressions
- **Iteration**: Multiple rounds can compound improvements

---

## Stretch Ideas

1. **Multi-objective tuning**: Optimize for both accuracy AND diversity
2. **User-specific weights**: Different weight sets for different user clusters
3. **Explainability**: Agent generates human-readable explanations of weight changes
4. **Online learning**: Add feedback in real-time without running full iterations
