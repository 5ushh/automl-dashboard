import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

from src.preprocessing import full_preprocess, detect_task_type
from src.model_trainer import train_all_models, get_best_model, get_feature_importance
from src.visualizations import (
    plot_distributions, plot_correlation, plot_missing,
    plot_model_comparison, plot_training_time,
    plot_feature_importance, plot_radar
)
from src.utils import load_csv, generate_classification_demo, generate_regression_demo, results_to_dataframe

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AutoML Dashboard", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 800; color: #111111; margin-bottom: 0; }
    .sub-title  { font-size: 1rem; color: #666666; margin-top: 0; }
    .metric-card {
        background: #f5f5f5; border-radius: 8px;
        padding: 16px 20px; margin: 4px 0;
        border-left: 4px solid #333333;
    }
    .best-banner {
        background: #222222; color: #ffffff;
        border-radius: 8px; padding: 14px 20px;
        font-size: 1.1rem; font-weight: 700; margin: 12px 0;
    }
    .section-divider { border-top: 1px solid #dddddd; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🤖 AutoML Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Machine Learning for Everyone — No Code Required</p>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📂 Data Source")
    data_source = st.radio("Choose data source", ["Upload CSV", "Classification Demo", "Regression Demo"])

    df = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded:
            df = load_csv(uploaded)
    elif data_source == "Classification Demo":
        df = generate_classification_demo()
        st.success("Loaded employee promotion dataset (300 rows)")
    else:
        df = generate_regression_demo()
        st.success("Loaded house price dataset (300 rows)")

    if df is not None:
        target_col = st.selectbox("🎯 Target Column", df.columns.tolist(), index=len(df.columns) - 1)

    st.subheader("🔧 Preprocessing")
    impute_strategy = st.selectbox("Missing Value Strategy", ["auto", "mean", "drop"])
    scaler_type = st.selectbox("Feature Scaler", ["standard", "minmax", "none"])
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

    st.subheader("🚀 Training")
    run_training = st.button("▶ Train All Models", use_container_width=True, type="primary")

    if st.button("🗑 Reset Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────────────
if df is None:
    st.info("👈 Load a dataset from the sidebar to get started.")
    st.stop()

# ── EDA Tab ───────────────────────────────────────────────────────────────────
tab_eda, tab_results, tab_inference = st.tabs(["📊 Data Explorer", "🏆 Model Results", "🔮 Predict"])

with tab_eda:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("#### Feature Distributions")
    fig_dist = plot_distributions(df)
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_corr = plot_correlation(df)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
    with col2:
        fig_miss = plot_missing(df)
        if fig_miss:
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("✅ No missing values detected.")

# ── Training ──────────────────────────────────────────────────────────────────
if run_training:
    with st.spinner("Preprocessing data..."):
        scaler_arg = None if scaler_type == "none" else scaler_type
        prep = full_preprocess(df, target_col,
                               impute_strategy=impute_strategy,
                               scaler_type=scaler_arg,
                               test_size=test_size)
        st.session_state["prep"] = prep

    st.session_state["results"] = {}
    progress_bar = st.progress(0)
    status = st.empty()

    def update_progress(val):
        progress_bar.progress(val)
        status.text(f"Training models... {int(val * 100)}%")

    results = train_all_models(
        prep["X_train"], prep["X_test"],
        prep["y_train"], prep["y_test"],
        prep["task_type"],
        progress_callback=update_progress
    )
    progress_bar.progress(1.0)
    status.text("✅ Training complete!")

    best_name, best_info = get_best_model(results, prep["task_type"])
    importances = get_feature_importance(best_info["model"], prep["features"])

    st.session_state["results"] = results
    st.session_state["best_name"] = best_name
    st.session_state["importances"] = importances

# ── Results Tab ───────────────────────────────────────────────────────────────
with tab_results:
    if "results" not in st.session_state or not st.session_state["results"]:
        st.info("Train models first using the sidebar.")
    else:
        results = st.session_state["results"]
        best_name = st.session_state["best_name"]
        prep = st.session_state["prep"]
        importances = st.session_state["importances"]

        st.markdown(f'<div class="best-banner">🏆 Best Model: {best_name}</div>', unsafe_allow_html=True)

        if prep["dropped_cols"]:
            st.warning(f"High cardinality columns dropped: {', '.join(prep['dropped_cols'])}")

        # Metrics table
        st.subheader("All Model Results")
        df_results = results_to_dataframe(results)
        st.dataframe(df_results.style.highlight_max(axis=0, color="#d4d4d4"), use_container_width=True)

        # CSV export
        csv = df_results.to_csv(index=False).encode()
        st.download_button("⬇ Download Results CSV", csv, "automl_results.csv", "text/csv")

        # Charts
        st.plotly_chart(plot_model_comparison(results, prep["task_type"]), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_training_time(results), use_container_width=True)
        with col2:
            fig_radar = plot_radar(results, prep["task_type"])
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)

        if importances:
            st.plotly_chart(plot_feature_importance(importances), use_container_width=True)

        # Model download
        buf = io.BytesIO()
        joblib.dump({"model": results[best_name]["model"], "prep": prep}, buf)
        st.download_button("⬇ Download Best Model (.pkl)", buf.getvalue(),
                           f"{best_name.replace(' ', '_')}_model.pkl", "application/octet-stream")

# ── Inference Tab ─────────────────────────────────────────────────────────────
with tab_inference:
    if "results" not in st.session_state or not st.session_state["results"]:
        st.info("Train models first to use the prediction tab.")
    else:
        prep = st.session_state["prep"]
        best_name = st.session_state["best_name"]
        best_model = st.session_state["results"][best_name]["model"]

        st.subheader(f"Predict with {best_name}")
        st.caption("Enter values for each feature below.")

        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(prep["features"]):
            with cols[i % 3]:
                input_vals[feat] = st.number_input(feat, value=0.0, key=f"inf_{feat}")

        if st.button("🔮 Predict", type="primary"):
            input_df = pd.DataFrame([input_vals])
            prediction = best_model.predict(input_df)[0]

            if prep["task_type"] == "classification" and prep["target_encoder"]:
                label = prep["target_encoder"].inverse_transform([int(prediction)])[0]
            else:
                label = round(float(prediction), 4)

            st.success(f"**Prediction:** {label}")

            if prep["task_type"] == "classification" and hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(input_df)[0]
                classes = prep["target_encoder"].classes_ if prep["target_encoder"] else [str(i) for i in range(len(proba))]
                proba_df = pd.DataFrame({"Class": classes, "Probability": proba}).sort_values("Probability", ascending=False)
                st.bar_chart(proba_df.set_index("Class"))
