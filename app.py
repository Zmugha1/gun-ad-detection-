"""
Weapons Detection Content Moderation — Streamlit app.
Five tabs: Three Approaches, Live Detection Demo, Calibration Analysis,
Business Impact Simulator, Monitoring & Drift.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import config
from data.synthetic_ads import generate_synthetic_ads
from data.cost_matrices import get_cost_matrix, total_cost
from approaches.baseline_model import BaselineWeaponsClassifier
from approaches.calibrated_model import CalibratedWeaponsClassifier
from approaches.theory_constrained_model import TheoryConstrainedWeaponsClassifier
from evaluation.calibration_metrics import ece, mce, brier_score, reliability_diagram_data
from evaluation.cost_analysis import compute_total_cost, human_review_queue_size
from evaluation.drift_detection import kl_divergence_bins, detect_drift
from visualization.calibration_plots import plot_reliability_diagram, plot_probability_histogram
from visualization.dashboard_components import cost_matrix_fig

st.set_page_config(page_title="Weapons Detection Moderation", layout="wide")

# ----- Data & models (cached) -----
@st.cache_data
def get_data():
    if config.SYNTHETIC_ADS_PATH.exists():
        df = pd.read_csv(config.SYNTHETIC_ADS_PATH)
    else:
        df = generate_synthetic_ads()
    return df

@st.cache_resource
def get_models():
    base = BaselineWeaponsClassifier()
    cal = CalibratedWeaponsClassifier(base_classifier=base)
    theory = TheoryConstrainedWeaponsClassifier(base_classifier=base)
    return base, cal, theory

def fit_calibration_models(base, cal, theory, df):
    """Fit calibration on a validation split (once)."""
    from sklearn.model_selection import train_test_split
    texts = (df["title"] + " " + df["description"] + " " + df["keywords"].fillna("")).tolist()
    y = df["label"].values
    _, X_val, _, y_val = train_test_split(texts, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    cal.fit_calibration(X_val, y_val)
    theory.fit_calibration(X_val, y_val)
    return cal, theory

# ----- Tab 1: The Three Approaches -----
def tab_three_approaches():
    st.header("The Three Approaches")
    df = get_data()
    base, cal, theory = get_models()
    # Fit calibration if we have enough data (use same split for both)
    texts = (df["title"] + " " + df["description"] + " " + df["keywords"].fillna("")).tolist()
    y = df["label"].values
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(texts, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    cal.fit_calibration(X_val, y_val)
    theory.fit_calibration(X_val, y_val)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Approach 1: Baseline")
        st.markdown("BERT + 0.5 threshold. No calibration, no cost awareness.")
    with col2:
        st.subheader("Approach 2: Calibrated")
        st.markdown("BERT + Platt scaling. Threshold 0.7. Better probabilities.")
    with col3:
        st.subheader("Approach 3: Theory-Constrained")
        st.markdown("Isotonic calibration + 3-tier + age-dependent thresholds.")

    st.markdown("---")
    st.subheader("Cost matrix")
    audience_sel = st.selectbox("Audience for cost matrix", ["general", "minors", "adults"])
    st.plotly_chart(cost_matrix_fig(audience_sel), use_container_width=True)

    st.subheader("Calibration curves (reliability diagrams)")
    # Get scores on validation set
    p1 = base.predict_proba(X_val)
    p2 = cal.predict_proba(X_val)
    p3 = theory.predict_proba(X_val)
    if p1.ndim > 1:
        p1, p2, p3 = p1.ravel(), p2.ravel(), p3.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (probs, name) in zip(axes, [(p1, "Baseline"), (p2, "Calibrated"), (p3, "Theory-Constrained")]):
        plot_reliability_diagram(y_val, probs, n_bins=10, title=name, ax=ax, label=name)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Comparison table
    from sklearn.metrics import precision_score, recall_score, f1_score
    thresh = 0.5
    pred1 = (p1 > thresh).astype(int)
    pred2 = (p2 > config.THRESHOLD_CALIBRATED).astype(int)
    pred3 = (p3 >= config.THRESHOLD_HUMAN_REVIEW_LOW).astype(int)
    def safe_metric(fn, y, p):
        try:
            return round(fn(y, p, zero_division=0), 4)
        except Exception:
            return 0.0
    rows = [
        {"Approach": "Baseline", "ECE": round(ece(y_val, p1), 4), "Precision": safe_metric(precision_score, y_val, pred1), "Recall": safe_metric(recall_score, y_val, pred1), "F1": safe_metric(f1_score, y_val, pred1), "Brier": round(brier_score(y_val, p1), 4)},
        {"Approach": "Calibrated", "ECE": round(ece(y_val, p2), 4), "Precision": safe_metric(precision_score, y_val, pred2), "Recall": safe_metric(recall_score, y_val, pred2), "F1": safe_metric(f1_score, y_val, pred2), "Brier": round(brier_score(y_val, p2), 4)},
        {"Approach": "Theory-Constrained", "ECE": round(ece(y_val, p3), 4), "Precision": safe_metric(precision_score, y_val, pred3), "Recall": safe_metric(recall_score, y_val, pred3), "F1": safe_metric(f1_score, y_val, pred3), "Brier": round(brier_score(y_val, p3), 4)},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ----- Tab 2: Live Detection Demo -----
def tab_live_demo():
    st.header("Live Detection Demo")
    ad_copy = st.text_area("Paste ad copy (title, description, keywords)", height=120, placeholder="e.g. tactical gear private sale no serial cash only...")
    audience = st.selectbox("Target audience", ["general", "minors", "adults"])
    if st.button("Run detection"):
        if not ad_copy.strip():
            st.warning("Enter some ad text.")
            return
        base, cal, theory = get_models()
        # Fit calibration on first run (small val set from cached data)
        df = get_data()
        texts = (df["title"] + " " + df["description"] + " " + df["keywords"].fillna("")).tolist()
        y = df["label"].values
        from sklearn.model_selection import train_test_split
        _, X_val, _, y_val = train_test_split(texts, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
        cal.fit_calibration(X_val, y_val)
        theory.fit_calibration(X_val, y_val)

        s1 = float(base.predict_proba(ad_copy).ravel()[0])
        s2 = float(cal.predict_proba(ad_copy).ravel()[0])
        s3 = float(theory.predict_proba(ad_copy).ravel()[0])
        t1 = "ban" if s1 > 0.5 else "allow"
        t2 = "ban" if s2 > config.THRESHOLD_CALIBRATED else "allow"
        t3 = theory.decision_tier(s3, audience)

        col1, col2, col3 = st.columns(3)
        for col, name, score, decision in [(col1, "Baseline", s1, t1), (col2, "Calibrated", s2, t2), (col3, "Theory-Constrained", s3, t3)]:
            with col:
                st.metric(name, f"{score:.3f}", decision)
                color = "red" if decision in ("ban", "auto_ban") else ("orange" if decision == "human_review" else "green")
                st.markdown(f"**Decision:** :{color}[{decision}]")

        st.subheader("Decision visualization (Theory-Constrained)")
        if t3 == "auto_ban":
            st.error("Auto-ban (score ≥ 0.90)")
        elif t3 == "human_review":
            st.warning("Human review (0.30–0.90)")
        else:
            st.success("Allow with monitoring (< 0.30/0.40)")

        st.subheader("Keywords that may trigger detection")
        weapons_words = ["tactical", "sale", "serial", "cash", "firearm", "rifle", "ammo", "private", "no background", "magazine", "receiver", "concealed", "carry", "reloading"]
        found = [w for w in weapons_words if w in (ad_copy or "").lower()]
        if found:
            st.write(", ".join(found))
        else:
            st.write("None of the tracked keywords found in this text.")

# ----- Tab 3: Calibration Analysis -----
def tab_calibration():
    st.header("Calibration Analysis")
    df = get_data()
    base, cal, theory = get_models()
    texts = (df["title"] + " " + df["description"] + " " + df["keywords"].fillna("")).tolist()
    y = df["label"].values
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(texts, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    cal.fit_calibration(X_val, y_val)
    theory.fit_calibration(X_val, y_val)

    p1 = base.predict_proba(X_val).ravel()
    p2 = cal.predict_proba(X_val).ravel()
    p3 = theory.predict_proba(X_val).ravel()

    st.subheader("Expected Calibration Error (ECE)")
    ece1, ece2, ece3 = ece(y_val, p1), ece(y_val, p2), ece(y_val, p3)
    st.dataframe(pd.DataFrame([
        {"Approach": "Baseline", "ECE": round(ece1, 4), "MCE": round(mce(y_val, p1), 4), "Brier": round(brier_score(y_val, p1), 4)},
        {"Approach": "Calibrated", "ECE": round(ece2, 4), "MCE": round(mce(y_val, p2), 4), "Brier": round(brier_score(y_val, p2), 4)},
        {"Approach": "Theory-Constrained", "ECE": round(ece3, 4), "MCE": round(mce(y_val, p3), 4), "Brier": round(brier_score(y_val, p3), 4)},
    ]), use_container_width=True)

    st.subheader("Reliability diagrams")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (probs, name) in zip(axes, [(p1, "Baseline"), (p2, "Calibrated"), (p3, "Theory-Constrained")]):
        plot_reliability_diagram(y_val, probs, n_bins=10, title=name, ax=ax, label=name)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Predicted probability histograms")
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (probs, name) in zip(axes2, [(p1, "Baseline"), (p2, "Calibrated"), (p3, "Theory-Constrained")]):
        plot_probability_histogram(probs, ax=ax, title=name, label=name)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ----- Tab 4: Business Impact Simulator -----
def tab_business_impact():
    st.header("Business Impact Simulator")
    cost_fn = st.slider("Cost of False Negative ($)", 1000, 100000, 10000, 1000)
    cost_fp = st.slider("Cost of False Positive ($)", 10, 500, 100, 10)
    volume = st.slider("Ad volume (per day)", 1000, 1_000_000, 10000, 1000)
    hr_cost = st.slider("Human review cost ($/hour)", 5, 50, 25, 5)

    df = get_data()
    base, cal, theory = get_models()
    texts = (df["title"] + " " + df["description"] + " " + df["keywords"].fillna("")).tolist()
    y = df["label"].values
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(texts, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    cal.fit_calibration(X_val, y_val)
    theory.fit_calibration(X_val, y_val)

    p1 = base.predict_proba(X_val).ravel()
    p2 = cal.predict_proba(X_val).ravel()
    p3 = theory.predict_proba(X_val).ravel()
    pred1 = (p1 > 0.5).astype(int)
    pred2 = (p2 > config.THRESHOLD_CALIBRATED).astype(int)
    pred3 = (p3 >= config.THRESHOLD_HUMAN_REVIEW_LOW).astype(int)
    hr_count = human_review_queue_size(p3, 0.40, 0.90)
    n_val = len(y_val)
    fn1 = ((pred1 == 0) & (y_val == 1)).sum()
    fp1 = ((pred1 == 1) & (y_val == 0)).sum()
    fn2 = ((pred2 == 0) & (y_val == 1)).sum()
    fp2 = ((pred2 == 1) & (y_val == 0)).sum()
    fn3 = ((pred3 == 0) & (y_val == 1)).sum()
    fp3 = ((pred3 == 1) & (y_val == 0)).sum()
    # For cost we use pred3 (flag for review or ban); FN = missed weapons, FP = wrong flag
    # Scale to daily volume (approximate)
    scale = volume / max(n_val, 1)
    cost1 = (fn1 * cost_fn + fp1 * cost_fp) * scale
    cost2 = (fn2 * cost_fn + fp2 * cost_fp) * scale
    cost3 = (fn3 * cost_fn + fp3 * cost_fp) * scale
    hr_daily_cost = hr_count * scale * (hr_cost / 60) * 2  # assume ~2 min per review
    cost3_total = cost3 + hr_daily_cost

    st.subheader("Estimated daily cost (scaled from validation set)")
    fig = go.Figure(data=[
        go.Bar(name="Approach 1 (Baseline)", x=["Cost"], y=[cost1], text=[f"${cost1:,.0f}"], textposition="auto"),
        go.Bar(name="Approach 2 (Calibrated)", x=["Cost"], y=[cost2], text=[f"${cost2:,.0f}"], textposition="auto"),
        go.Bar(name="Approach 3 (Theory) + HR queue", x=["Cost"], y=[cost3_total], text=[f"${cost3_total:,.0f}"], textposition="auto"),
    ])
    fig.update_layout(barmode="group", title="Cost comparison", yaxis_title="$/day", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Approach 3 includes human review queue cost. FN/FP use your chosen cost sliders.")

# ----- Tab 5: Monitoring & Drift -----
def tab_monitoring():
    st.header("Monitoring & Drift")
    df = get_data()
    base, cal, theory = get_models()
    texts = (df["title"] + " " + df["description"] + " " + df["keywords"].fillna("")).tolist()
    y = df["label"].values
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(texts, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    cal.fit_calibration(X_val, y_val)
    theory.fit_calibration(X_val, y_val)

    p_train = theory.predict_proba(X_tr).ravel()
    p_prod = theory.predict_proba(X_val).ravel()
    # Simulate 30 days drift: add small shift to prod
    np.random.seed(42)
    days = 30
    kl_per_day = []
    for d in range(days):
        shift = 0.02 * (d - 15) + 0.01 * np.random.randn(len(p_prod))
        p_sim = np.clip(p_prod + shift, 0, 1)
        kl_per_day.append(kl_divergence_bins(p_train, p_sim))
    is_drift, kl_now = detect_drift(p_train, p_prod, config.DRIFT_KL_THRESHOLD)

    st.subheader("KL divergence over time (simulated 30 days)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(days)), y=kl_per_day, mode="lines+markers", name="KL(prod || train)"))
    fig.add_hline(y=config.DRIFT_KL_THRESHOLD, line_dash="dash", line_color="red", annotation_text="Alert threshold")
    fig.update_layout(xaxis_title="Day", yaxis_title="KL divergence", height=350)
    st.plotly_chart(fig, use_container_width=True)
    if is_drift:
        st.warning(f"Drift detected: KL = {kl_now:.4f} > {config.DRIFT_KL_THRESHOLD}. Consider retraining.")
    else:
        st.success(f"No drift: KL = {kl_now:.4f} ≤ {config.DRIFT_KL_THRESHOLD}.")

    st.subheader("Demographic parity (FP rate by age targeting)")
    indices = np.arange(len(texts))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    age_val = df["age_targeting"].iloc[val_idx].values if "age_targeting" in df.columns else np.array(["general"] * len(y_val))
    p3 = theory.predict_proba(X_val).ravel()
    pred3 = (p3 >= config.THRESHOLD_HUMAN_REVIEW_LOW).astype(int)
    benign = y_val == 0
    if benign.sum() > 0 and len(age_val) == len(pred3):
        fp_df = pd.DataFrame({"age": age_val[benign], "fp": pred3[benign]})
        fp_rate = fp_df.groupby("age")["fp"].mean()
        st.bar_chart(fp_rate)
    else:
        st.write("FP rate (benign only): ", round(pred3[benign].sum() / max(benign.sum(), 1), 4))

# ----- Main -----
def main():
    st.title("Weapons Detection Content Moderation")
    st.caption("Theory-constrained calibration and cost-sensitive moderation for high-stakes trust & safety.")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "The Three Approaches",
        "Live Detection Demo",
        "Calibration Analysis",
        "Business Impact Simulator",
        "Monitoring & Drift",
    ])
    with tab1:
        tab_three_approaches()
    with tab2:
        tab_live_demo()
    with tab3:
        tab_calibration()
    with tab4:
        tab_business_impact()
    with tab5:
        tab_monitoring()

if __name__ == "__main__":
    main()
