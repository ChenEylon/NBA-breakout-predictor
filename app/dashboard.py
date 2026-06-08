"""
Phase 5: Streamlit Dashboard
NBA Breakout Player Predictor — 3-view interactive dashboard.

Usage:
    streamlit run app/dashboard.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- path setup so imports work from /app/ --------------------------------
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.features.engineer import load_features, get_feature_columns, LAG_STATS
from src.explainability.shap_explainer import run as get_shap_candidates

MODELS_DIR = ROOT / "models"

# --------------------------------------------------------------------------
st.set_page_config(page_title="NBA Breakout Predictor 2027", layout="wide", page_icon="🏀")

# --------------------------------------------------------------------------
# Data loaders (cached)
# --------------------------------------------------------------------------

@st.cache_data
def load_candidates():
    return get_shap_candidates()


@st.cache_data
def load_model_metrics():
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_all_models():
    models = {}
    for name in ["xgboost", "random_forest", "lightgbm"]:
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
    return models


@st.cache_data
def load_val_data():
    df = load_features()
    feat_cols = get_feature_columns()
    val = df[df["season"] == "2025-26"].copy()
    available = [c for c in feat_cols if c in val.columns]
    X = val[available].fillna(val[available].median(numeric_only=True))
    return val, X, available


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------

st.sidebar.title("🏀 NBA Breakout Predictor")
st.sidebar.markdown("**2026-27 Season Predictions**")
st.sidebar.divider()

min_prob = st.sidebar.slider("Min breakout probability", 0, 100, 50, 5, format="%d%%") / 100
age_range = st.sidebar.slider("Age range", 18, 35, (18, 28))
view = st.sidebar.radio("View", ["🏆 Top 20 Candidates", "🔍 Player Deep Dive", "📊 Model Performance"])

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

with st.spinner("Loading model & predictions…"):
    candidates = load_candidates()
    metrics = load_model_metrics()

# Apply sidebar filters
filtered = [
    c for c in candidates
    if c["breakout_prob"] >= min_prob
    and (c.get("age") is None or age_range[0] <= c["age"] <= age_range[1])
]

# --------------------------------------------------------------------------
# View 1: Top 20 Candidates
# --------------------------------------------------------------------------

if "Top 20" in view:
    st.title("🏆 2027 Breakout Candidates")
    st.caption("Players predicted to have a breakout season (≥20% jump in PTS, PIE, or TS%) in 2026-27.")

    if not filtered:
        st.warning("No players match the current filters. Adjust the sidebar sliders.")
    else:
        # --- summary bar chart ---
        df_plot = pd.DataFrame(filtered)
        fig = px.bar(
            df_plot.sort_values("breakout_prob", ascending=True).tail(20),
            x="breakout_prob",
            y="name",
            orientation="h",
            color="breakout_prob",
            color_continuous_scale="RdYlGn",
            range_color=[0, 1],
            labels={"breakout_prob": "Breakout Probability", "name": ""},
            title="Top Breakout Candidates — 2026-27",
        )
        fig.update_traces(
            text=[f"{v:.1%}" for v in df_plot.sort_values("breakout_prob", ascending=True).tail(20)["breakout_prob"]],
            textposition="outside",
        )
        fig.update_layout(height=max(400, len(filtered) * 28), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- table ---
        st.subheader("Full Table")
        table_data = []
        for c in sorted(filtered, key=lambda x: -x["breakout_prob"]):
            table_data.append({
                "Player": c["name"],
                "Team": c.get("team", ""),
                "Age": c.get("age"),
                "PPG (25-26)": f"{c.get('pts', 0) or 0:.1f}",
                "TS% (25-26)": f"{(c.get('ts_pct', 0) or 0):.1%}",
                "USG% (25-26)": f"{(c.get('usg_pct', 0) or 0):.1%}",
                "PIE (25-26)": f"{c.get('pie', 0) or 0:.3f}",
                "Breakout Prob": f"{c['breakout_prob']:.1%}",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# View 2: Player Deep Dive
# --------------------------------------------------------------------------

elif "Deep Dive" in view:
    st.title("🔍 Player Deep Dive")

    player_names = [c["name"] for c in candidates]
    selected = st.selectbox("Select a player", player_names)
    ctx = next(c for c in candidates if c["name"] == selected)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Breakout Prob", f"{ctx['breakout_prob']:.1%}")
    col2.metric("Age", ctx.get("age", "N/A"))
    col3.metric("Team", ctx.get("team", "N/A"))
    col4.metric("Years in League", ctx.get("years_in_league", "N/A"))

    st.divider()

    # --- Radar chart: current vs prior season ---
    st.subheader("Stat Radar: 2025-26 vs Prior Season")
    with st.expander("📖 What do these stats mean?"):
        st.markdown("""
| Stat | Full Name | What it measures |
|---|---|---|
| **PTS** | Points per game | Raw scoring volume |
| **AST** | Assists per game | Playmaking and court vision |
| **REB** | Rebounds per game | Presence on the boards (offensive + defensive) |
| **TS%** | True Shooting % | Shooting efficiency accounting for 2s, 3s, and free throws — more accurate than FG% |
| **USG%** | Usage Rate | % of team possessions used by the player while on the floor — proxy for role and trust |
| **PIE** | Player Impact Estimate | NBA's all-in-one impact metric — how much a player contributes to winning relative to all players on the court |

The radar shows **2025-26 (blue)** vs **prior season (red)**, normalised so the league leader in each stat = 1.0. A growing shape means the player is trending up.
        """)
    radar_stats = ["pts", "ast", "reb", "ts_pct", "usg_pct", "pie"]
    radar_labels = ["PTS", "AST", "REB", "TS%", "USG%", "PIE"]

    val_df, _, _ = load_val_data()
    player_row = val_df[val_df["name"] == selected]

    if not player_row.empty:
        row = player_row.iloc[0]
        current_vals = []
        prior_vals = []
        for stat in radar_stats:
            c_val = row.get(stat, 0) or 0
            p_val = row.get(f"prev_{stat}", 0) or 0
            current_vals.append(float(c_val))
            prior_vals.append(float(p_val))

        # Normalise to 0-1 scale per stat (using val_df col max)
        def norm(vals, col_list):
            normed = []
            for v, stat in zip(vals, col_list):
                col_max = val_df[stat].replace(0, np.nan).max()
                normed.append(v / col_max if col_max else 0)
            return normed

        cur_norm = norm(current_vals, radar_stats)
        pri_norm_cols = [f"prev_{s}" for s in radar_stats]
        pri_norm = []
        for v, stat in zip(prior_vals, pri_norm_cols):
            col_max = val_df[stat.replace("prev_", "")].replace(0, np.nan).max() if stat.replace("prev_", "") in val_df.columns else 1
            pri_norm.append(v / col_max if col_max else 0)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=cur_norm + [cur_norm[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="2025-26",
            line_color="royalblue",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=pri_norm + [pri_norm[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Prior season",
            line_color="tomato",
            opacity=0.5,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"{selected} — Normalized Stats",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- SHAP waterfall ---
    st.subheader("SHAP Feature Importance")
    with st.expander("📖 What is SHAP?"):
        st.markdown("""
**SHAP** (SHapley Additive exPlanations) explains *why* the model gave a player their breakout probability.

Each bar shows how much a single feature pushed the prediction **up** (green) or **down** (red).
- A long green bar for *Prior PIE* means "this player's PIE last season is a strong reason to predict a breakout"
- A red bar means that feature is actually working against them

The values are in log-odds units — think of them as the raw signal before converting to a probability.
This makes the model transparent: you're not just getting a number, you're seeing the reasoning behind it.
        """)
    shap_feats = ctx.get("top_shap_features", [])
    if shap_feats:
        shap_df = pd.DataFrame(shap_feats)
        shap_df["color"] = shap_df["shap_value"].apply(lambda v: "green" if v > 0 else "red")
        fig_shap = go.Figure(go.Bar(
            x=shap_df["shap_value"],
            y=shap_df["feature"].str.replace("prev_", "Prior "),
            orientation="h",
            marker_color=shap_df["color"],
        ))
        fig_shap.update_layout(
            title=f"Top SHAP Drivers — {selected}",
            xaxis_title="SHAP Value (log-odds contribution)",
            yaxis={"categoryorder": "total ascending"},
            height=350,
        )
        st.plotly_chart(fig_shap, use_container_width=True)


# --------------------------------------------------------------------------
# View 3: Model Performance
# --------------------------------------------------------------------------

elif "Performance" in view:
    st.title("📊 Model Performance")
    st.caption("Evaluated on 2025-26 held-out validation season.")
    with st.expander("📖 How to read these results"):
        st.markdown("""
Three models were trained on 2010–2025 season data and evaluated on the **2025-26 held-out season** (data the model never saw during training).

| Metric | What it means |
|---|---|
| **Precision** | Of the players we flagged as breakout candidates, what % actually broke out. Higher = fewer false alarms. |
| **Recall** | Of all players who actually broke out, what % did we catch. Higher = fewer missed breakouts. |
| **F1** | Harmonic mean of precision and recall — balances both. |
| **AUC-ROC** | How well the model separates breakouts from non-breakouts across *all* thresholds (1.0 = perfect, 0.5 = random). |
| **Avg Precision** | Area under the Precision-Recall curve — better than AUC-ROC for imbalanced datasets like this one (~14% breakout rate). |

**Why LightGBM was selected:** Precision was prioritised over recall. In a scouting tool, a short list of high-confidence picks is more useful than a long list with lots of noise. LightGBM's 56% precision is 4× above the 14% random baseline.
        """)

    if not metrics:
        st.error("metrics.json not found. Run `python src/models/train.py` first.")
    else:
        model_results = metrics.get("models", {})
        best = metrics.get("best", "")

        # --- comparison table ---
        NAME_DISPLAY = {"xgboost": "XGBoost", "random_forest": "Random Forest", "lightgbm": "LightGBM"}

        rows = []
        for name, m in model_results.items():
            rows.append({
                "Model": ("⭐ " if name == best else "") + NAME_DISPLAY.get(name, name.replace("_", " ").title()),
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1": f"{m['f1']:.3f}",
                "AUC-ROC": f"{m['auc_roc']:.3f}",
                "Avg Precision": f"{m['avg_precision']:.3f}",
                "Threshold": f"{m['threshold']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # --- precision/recall bar comparison ---
        fig_comp = go.Figure()
        model_names = list(model_results.keys())
        display_names = [NAME_DISPLAY.get(n, n.replace("_", " ").title()) for n in model_names]
        prec_vals = [model_results[n]["precision"] for n in model_names]
        rec_vals = [model_results[n]["recall"] for n in model_results]
        auc_vals = [model_results[n]["auc_roc"] for n in model_results]

        fig_comp.add_trace(go.Bar(name="Precision", x=display_names, y=prec_vals, marker_color="royalblue"))
        fig_comp.add_trace(go.Bar(name="Recall", x=display_names, y=rec_vals, marker_color="tomato"))
        fig_comp.add_trace(go.Bar(name="AUC-ROC", x=display_names, y=auc_vals, marker_color="seagreen"))
        fig_comp.update_layout(barmode="group", title="Model Comparison — Validation Season 2025-26", yaxis_range=[0, 1])
        st.plotly_chart(fig_comp, use_container_width=True)

        # --- compute precision-recall curves ---
        st.subheader("Precision-Recall Curves")
        try:
            from sklearn.metrics import precision_recall_curve

            models_loaded = load_all_models()
            val_df, X_val, feat_cols = load_val_data()
            y_val = val_df["breakout"]

            fig_pr = go.Figure()
            for name, model in models_loaded.items():
                probs = model.predict_proba(X_val)[:, 1]
                prec_c, rec_c, _ = precision_recall_curve(y_val, probs)
                fig_pr.add_trace(go.Scatter(
                    x=rec_c, y=prec_c,
                    mode="lines",
                    name=name.replace("_", " ").title(),
                ))
            baseline = y_val.mean()
            fig_pr.add_hline(y=baseline, line_dash="dash", line_color="gray",
                             annotation_text=f"Baseline ({baseline:.2f})")
            fig_pr.update_layout(
                title="Precision-Recall Curves — 2025-26 Validation",
                xaxis_title="Recall",
                yaxis_title="Precision",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render PR curves: {e}")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.caption(
    "**Training:** 2010-11 → 2024-25 (15 seasons)  \n"
    "**Val/Prediction base:** 2025-26  \n"
    "**Best model:** LightGBM  \n"
    "**Breakout = ≥20% jump in PTS, PIE, or TS%**"
)
