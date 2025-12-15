import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="ODI Wicket Predictor", layout="wide")


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent  
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "wicket_model_pipe_hgb_v2.joblib"
SCHEMA_PATH = ARTIFACTS_DIR / "model_features_v2.json"


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    return model, schema

pipe_hgb, schema = load_artifacts()

FEATURES = schema["feature_cols"]
CAT_COLS = schema["cat_cols"]
NUM_COLS = schema["num_cols"]


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.caption(f"Model: {MODEL_PATH.name}")
st.sidebar.caption(f"Features expected: {len(FEATURES)}")
with st.sidebar.expander("Model feature list"):
    st.write(FEATURES)


# -----------------------------
# Helpers
# -----------------------------
def compute_phase(over_in_innings: int) -> str:
    if over_in_innings <= 10:
        return "powerplay"
    if over_in_innings <= 40:
        return "middle"
    return "death"


def compute_matchup(tactical_style_4: str, batter_hand: str) -> str:
    suffix = "RHB" if batter_hand == "right" else "LHB"
    if tactical_style_4 in ["left arm pace", "right arm pace"]:
        return f"{tactical_style_4} vs {suffix}"
    return f"{tactical_style_4.replace('_', ' ')} vs {suffix}"


def derive_state(
    innings: str,
    over_in_innings: int,
    delivery_in_over: int,   
    runs_innings_before: int,
    req_rr_input: float
) -> dict:
    
    legal_ball = int(np.clip(delivery_in_over, 1, 9))
    legal_ball_for_calc = min(legal_ball, 6)

    balls_faced_before = (int(over_in_innings) - 1) * 6 + (legal_ball_for_calc - 1)
    balls_faced_before = int(np.clip(balls_faced_before, 0, 299))

    ball_num_in_innings = float(balls_faced_before)
    balls_remaining = float(300 - balls_faced_before)

    curr_rr = np.nan if balls_faced_before == 0 else (runs_innings_before / balls_faced_before) * 6

    phase = compute_phase(int(over_in_innings))

    is_chase = (innings == "2nd innings") and (req_rr_input > 0)
    req_rr_val = float(req_rr_input) if is_chase else np.nan
    rr_pressure_val = (req_rr_val - curr_rr) if is_chase and not np.isnan(curr_rr) else np.nan

    runs_innings_before_cap = float(np.clip(runs_innings_before, 0, 500))
    curr_rr_cap = np.nan if np.isnan(curr_rr) else float(np.clip(curr_rr, 0, 15))
    req_rr_cap = np.nan if np.isnan(req_rr_val) else float(np.clip(req_rr_val, 0, 20))
    rr_pressure_cap = np.nan if np.isnan(rr_pressure_val) else float(np.clip(rr_pressure_val, -15, 15))

    return {
        "phase": phase,
        "ball_num_in_innings": ball_num_in_innings,
        "balls_remaining": balls_remaining,
        "curr_rr": curr_rr,
        "req_rr_val": req_rr_val,
        "rr_pressure_val": rr_pressure_val,
        "runs_innings_before_cap": runs_innings_before_cap,
        "curr_rr_cap": curr_rr_cap,
        "req_rr_cap": req_rr_cap,
        "rr_pressure_cap": rr_pressure_cap,
        "legal_ball_for_calc": legal_ball_for_calc,  
    }


def predict_one_ball(model, inputs: dict) -> float:
    row = {f: inputs.get(f, np.nan) for f in FEATURES}
    X = pd.DataFrame([row]).replace([np.inf, -np.inf], np.nan)
    return float(model.predict_proba(X)[:, 1][0])


def rank_bowling_options(model, match_state: dict, options: list) -> pd.DataFrame:
    rows = []
    for opt in options:
        merged = {**match_state, **opt}
        p = predict_one_ball(model, merged)
        rows.append({**opt, "p_wicket": p})
    return pd.DataFrame(rows).sort_values("p_wicket", ascending=False).reset_index(drop=True)


# -----------------------------
# UI
# -----------------------------
st.title("ODI Wicket Probability Tool")
tab1, tab2 = st.tabs(["Tab 1 — Single Ball Predictor", "Tab 2 — Bowling Advisor"])


# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.subheader("Single Ball Predictor")

    st.markdown("### Ball + match state")
    c0, c1, c2, c3, c4, c5 = st.columns(6)

    with c0:
        innings = st.selectbox("Innings", ["1st innings", "2nd innings"])
    with c1:
        over_in_innings = st.number_input("Over (in innings)", min_value=1, max_value=50, value=48)
    with c2:
        ball_in_over = st.number_input("Ball in over (delivery #, can be 7/8/9)", min_value=1, max_value=9, value=3)
    with c3:
        wkts_in_hand_before = st.number_input("Wickets in hand (before ball)", min_value=0, max_value=10, value=4)
    with c4:
        runs_innings_before = st.number_input("Runs in innings (before ball)", min_value=0, max_value=500, value=255)
    with c5:
        req_rr_input = st.number_input("Required RR (2nd inns chase only, else 0)", min_value=0.0, max_value=50.0, value=0.0)

    st.markdown("### Players / bowling plan")
    d1, d2, d3 = st.columns(3)
    e1, e2 = st.columns(2)

    with d1:
        bowler_bowling_type = st.selectbox("Bowler bowling type", ["pace", "spin"])
    with d2:
        bowler_arm = st.selectbox("Bowler arm", ["right", "left"])
    with d3:
        batter_hand = st.selectbox("Batter hand", ["right", "left"])

    with e1:
        tactical_style_4 = st.selectbox(
            "Tactical style",
            ["left arm pace", "right arm pace", "spin_into", "spin_away"]
        )
    with e2:
        bowler_style_group = st.selectbox(
            "Bowler style group",
            [
                "left arm pace",
                "right arm pace",
                "left arm wrist spin",
                "slow left arm orthodox",
                "right arm offbreak",
                "right arm legbreak",
                "right arm legbreak googly",
            ],
        )

    matchup_type = compute_matchup(tactical_style_4, batter_hand)
    derived = derive_state(
        innings=innings,
        over_in_innings=int(over_in_innings),
        delivery_in_over=int(ball_in_over),
        runs_innings_before=int(runs_innings_before),
        req_rr_input=float(req_rr_input),
    )

    st.markdown("### Auto-derived context")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Phase", derived["phase"])
    a2.metric("Matchup type", matchup_type)
    a3.metric("Legal ball used", str(derived["legal_ball_for_calc"]))
    a4.metric("Current RR", "NA" if np.isnan(derived["curr_rr"]) else f"{derived['curr_rr']:.2f}")
    a5.metric("RR pressure", "NA" if np.isnan(derived["rr_pressure_val"]) else f"{derived['rr_pressure_val']:.2f}")

    if st.button("Predict wicket probability", key="btn_predict"):
        inputs = {
            "phase": derived["phase"],
            "tactical_style_4": tactical_style_4,
            "bowler_bowling_type": bowler_bowling_type,
            "bowler_arm": bowler_arm,
            "batter_hand": batter_hand,
            "bowler_style_group": bowler_style_group,
            "matchup_type": matchup_type,

            "ball_num_in_innings": derived["ball_num_in_innings"],
            "balls_remaining": derived["balls_remaining"],
            "wkts_in_hand_before": float(wkts_in_hand_before),
            "runs_innings_before_cap": derived["runs_innings_before_cap"],
            "curr_rr_cap": derived["curr_rr_cap"],
            "req_rr_cap": derived["req_rr_cap"],
            "rr_pressure_cap": derived["rr_pressure_cap"],
        }

        p = predict_one_ball(pipe_hgb, inputs)
        st.success(f"Predicted wicket probability: {p:.4f} ({p*100:.2f}%)")


# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    st.subheader("Bowling Advisor (rank bowling profiles for this situation)")
    st.caption("Note: this ranks **bowling profiles** (arm/type/style), not individual bowler quality.")

    st.markdown("### Situation")
    s0, s1, s2, s3, s4, s5 = st.columns(6)

    with s0:
        innings2 = st.selectbox("Innings", ["1st innings", "2nd innings"], key="t2_innings")
    with s1:
        over_in_innings2 = st.number_input("Over (in innings)", min_value=1, max_value=50, value=48, key="t2_over")
    with s2:
        ball_in_over2 = st.number_input(
            "Ball in over (delivery #, can be 7/8/9)",
            min_value=1, max_value=9, value=3, key="t2_ball"
        )
    with s3:
        wkts_in_hand_before2 = st.number_input("Wickets in hand", min_value=0, max_value=10, value=4, key="t2_wkts")
    with s4:
        runs_innings_before2 = st.number_input("Runs in innings (before ball)", min_value=0, max_value=500, value=255, key="t2_runs")
    with s5:
        req_rr_input2 = st.number_input(
            "Required RR (2nd inns chase only, else 0)",
            min_value=0.0, max_value=50.0, value=0.0, key="t2_req"
        )

    batter_hand2 = st.selectbox("Batter hand", ["right", "left"], key="t2_bhand")

    derived2 = derive_state(
        innings=innings2,
        over_in_innings=int(over_in_innings2),
        delivery_in_over=int(ball_in_over2),
        runs_innings_before=int(runs_innings_before2),
        req_rr_input=float(req_rr_input2),
    )

    st.markdown("### Auto-derived context")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Phase", derived2["phase"])
    b2.metric("Balls remaining", int(derived2["balls_remaining"]))
    b3.metric("Legal ball used", str(derived2["legal_ball_for_calc"]))
    b4.metric("Current RR", "NA" if np.isnan(derived2["curr_rr"]) else f"{derived2['curr_rr']:.2f}")
    b5.metric("RR pressure", "NA" if np.isnan(derived2["rr_pressure_val"]) else f"{derived2['rr_pressure_val']:.2f}")

    match_state = {
        "phase": derived2["phase"],
        "batter_hand": batter_hand2,

        "ball_num_in_innings": derived2["ball_num_in_innings"],
        "balls_remaining": derived2["balls_remaining"],
        "wkts_in_hand_before": float(wkts_in_hand_before2),

        "runs_innings_before_cap": derived2["runs_innings_before_cap"],
        "curr_rr_cap": derived2["curr_rr_cap"],
        "req_rr_cap": derived2["req_rr_cap"],
        "rr_pressure_cap": derived2["rr_pressure_cap"],
    }

    # -----------------------------
    # Archetypes 
    # -----------------------------
    st.markdown("### Bowling profiles to compare")
    st.caption("Spin direction is set automatically from batter hand.")

    def opposite_spin(tac: str) -> str:
        return "spin_away" if tac == "spin_into" else "spin_into"

    def tactical_for_spin(dir_vs_rhb: str, batter_hand: str) -> str:
        tac_rhb = "spin_into" if dir_vs_rhb == "into" else "spin_away"
        return tac_rhb if batter_hand == "right" else opposite_spin(tac_rhb)

    ARCHETYPES = [
        dict(label="Left-arm pace", bowler_bowling_type="pace", bowler_arm="left",
             bowler_style_group="left arm pace", tactical_style_fixed="left arm pace"),
        dict(label="Right-arm pace", bowler_bowling_type="pace", bowler_arm="right",
             bowler_style_group="right arm pace", tactical_style_fixed="right arm pace"),

        dict(label="SLA (slow left-arm orthodox)", bowler_bowling_type="spin", bowler_arm="left",
             bowler_style_group="slow left arm orthodox", dir_vs_rhb="away"),
        dict(label="LA wrist spin", bowler_bowling_type="spin", bowler_arm="left",
             bowler_style_group="left arm wrist spin", dir_vs_rhb="into"),
        dict(label="Offbreak", bowler_bowling_type="spin", bowler_arm="right",
             bowler_style_group="right arm offbreak", dir_vs_rhb="into"),
        dict(label="Legbreak", bowler_bowling_type="spin", bowler_arm="right",
             bowler_style_group="right arm legbreak", dir_vs_rhb="away"),
        dict(label="Googly", bowler_bowling_type="spin", bowler_arm="right",
             bowler_style_group="right arm legbreak googly", dir_vs_rhb="into"),
    ]

    default_selected = [a["label"] for a in ARCHETYPES]
    selected = st.multiselect(
        "Select profiles to rank",
        options=[a["label"] for a in ARCHETYPES],
        default=default_selected,
        key="t2_profiles"
    )

    options = []
    for a in ARCHETYPES:
        if a["label"] not in selected:
            continue

        if a["bowler_bowling_type"] == "pace":
            tactical_style_4 = a["tactical_style_fixed"]
        else:
            tactical_style_4 = tactical_for_spin(a["dir_vs_rhb"], batter_hand2)

        matchup = compute_matchup(tactical_style_4, batter_hand2)

        options.append({
            "label": a["label"],
            "tactical_style_4": tactical_style_4,
            "bowler_bowling_type": a["bowler_bowling_type"],
            "bowler_arm": a["bowler_arm"],
            "bowler_style_group": a["bowler_style_group"],
            "matchup_type": matchup,
        })

    if st.button("Rank profiles", key="btn_rank_profiles"):
        if len(options) == 0:
            st.error("Select at least one profile.")
        else:
            ranked = rank_bowling_options(pipe_hgb, match_state, options)
            ranked["p_wicket_%"] = ranked["p_wicket"] * 100

            best_p = ranked.loc[0, "p_wicket"]
            ranked["edge_vs_best_pp"] = (ranked["p_wicket"] - best_p) * 100  

            if len(ranked) >= 2:
                gap_pp = (ranked.loc[0, "p_wicket"] - ranked.loc[1, "p_wicket"]) * 100
                if gap_pp < 0.30:
                    st.info(f"Top profiles are basically tied (gap {gap_pp:.2f} percentage points). Treat as a tie.")
                elif gap_pp < 0.75:
                    st.warning(f"Only a small edge (gap {gap_pp:.2f} percentage points).")

            best = ranked.iloc[0]
            st.success(f"Best profile: **{best['label']}** — {best['p_wicket_%']:.2f}% wicket chance")

            st.dataframe(
                ranked[
                    ["label", "p_wicket_%", "edge_vs_best_pp",
                     "tactical_style_4", "bowler_bowling_type", "bowler_arm",
                     "bowler_style_group", "matchup_type"]
                ],
                use_container_width=True
            )

