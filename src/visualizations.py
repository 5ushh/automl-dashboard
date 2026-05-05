import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_distributions(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return None
    cols = min(3, len(num_cols))
    rows = (len(num_cols) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=num_cols)
    for i, col in enumerate(num_cols):
        r, c = divmod(i, cols)
        fig.add_trace(go.Histogram(x=df[col], name=col, marker_color="#888888"), row=r + 1, col=c + 1)
    fig.update_layout(showlegend=False, title="Feature Distributions", height=300 * rows,
                      paper_bgcolor="white", plot_bgcolor="#f9f9f9", font=dict(color="#333333"))
    return fig


def plot_correlation(df):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation Heatmap", text_auto=".2f")
    fig.update_layout(paper_bgcolor="white", font=dict(color="#333333"))
    return fig


def plot_missing(df):
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None
    fig = px.bar(x=missing.index, y=missing.values,
                 labels={"x": "Column", "y": "Missing (%)"},
                 title="Missing Values per Column",
                 color=missing.values, color_continuous_scale="Greys")
    fig.update_layout(paper_bgcolor="white", font=dict(color="#333333"))
    return fig


def plot_model_comparison(results, task_type):
    rows = []
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"] if task_type == "classification" else ["R2 Score"]
    for name, info in results.items():
        if not info["metrics"]:
            continue
        row = {"Model": name}
        for m in metrics:
            row[m] = info["metrics"].get(m, 0) or 0
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(metrics[0], ascending=False)
    fig = px.bar(df, x="Model", y=metrics, barmode="group",
                 title="Model Performance Comparison",
                 color_discrete_sequence=["#333333", "#666666", "#999999", "#cccccc"])
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f9f9f9",
                      font=dict(color="#333333"), xaxis_tickangle=-30)
    return fig


def plot_training_time(results):
    names = [n for n, i in results.items() if i.get("training_time")]
    times = [results[n]["training_time"] for n in names]
    df = pd.DataFrame({"Model": names, "Time (s)": times}).sort_values("Time (s)")
    fig = px.bar(df, x="Time (s)", y="Model", orientation="h",
                 title="Training Time per Model",
                 color="Time (s)", color_continuous_scale="Greys")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f9f9f9", font=dict(color="#333333"))
    return fig


def plot_feature_importance(importances, top_n=15):
    if not importances:
        return None
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*items)
    fig = px.bar(x=list(values), y=list(features), orientation="h",
                 title=f"Top {top_n} Feature Importances",
                 color=list(values), color_continuous_scale="Greys")
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      paper_bgcolor="white", plot_bgcolor="#f9f9f9", font=dict(color="#333333"))
    return fig


def plot_radar(results, task_type):
    if task_type != "classification":
        return None
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    top5 = sorted(
        [(n, i) for n, i in results.items() if i["metrics"]],
        key=lambda x: x[1]["metrics"].get("Accuracy", 0),
        reverse=True
    )[:5]
    fig = go.Figure()
    colors = ["#222222", "#555555", "#888888", "#aaaaaa", "#cccccc"]
    for (name, info), color in zip(top5, colors):
        vals = [info["metrics"].get(m, 0) or 0 for m in metrics]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=metrics + [metrics[0]],
                                      fill="toself", name=name, line_color=color))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      title="Top 5 Models Radar Chart",
                      paper_bgcolor="white", font=dict(color="#333333"))
    return fig
