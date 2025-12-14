# ODI Wicket Predictor (Ball-by-Ball) — Streamlit App + ML

Predict the probability of a wicket falling on the next **legal delivery** in an ODI using ball-by-ball match context and a simplified bowling “profile”. Includes an interactive Streamlit app with:

- **Tab 1:** single-ball wicket probability
- **Tab 2:** compare/rank bowling options in the same match situation

---

## What this project does

This project parses ODI ball-by-ball YAML scorecards and builds:

1. **Exploratory analysis** of wicket rates by:
   - pace vs spin
   - match phase (powerplay / middle / death)
   - batter handedness matchups
   - spin direction (into vs away) and style-group effects

2. **Feature engineering** that converts raw deliveries into deployable match-state features:
   - legal-ball indexing (handles overs with 7/8/9 deliveries due to wides/no-balls)
   - balls remaining / ball index in innings
   - current RR, required RR (chase-only), RR pressure
   - caps/clipping for stability in modelling + UI

3. A **supervised ML model** trained to estimate `P(wicket=1)` for the next ball.

4. A **Streamlit app** that exposes the model in a usable decision-support form.

---

## Streamlit app

### Tab 1 — Single Ball Predictor
Inputs:
- innings, over, delivery number, wickets in hand, runs so far, required RR (chase only)
- bowling profile (pace/spin, arm, batter hand, tactical style, style group)

Outputs:
- wicket probability for the next ball

### Tab 2 — Bowling Advisor
Inputs:
- same match situation as Tab 1
- multiple bowling options (tactical style + style group)

Outputs:
- ranked list of options by predicted wicket probability

---

## Repository structure (suggested)

