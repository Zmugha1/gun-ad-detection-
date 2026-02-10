"""
Streamlit-friendly components: cost matrix table, approach comparison.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

try:
    from data.cost_matrices import get_cost_matrix
except ImportError:
    from ..data.cost_matrices import get_cost_matrix


def cost_matrix_fig(audience: str = "general"):
    """Plotly heatmap of cost matrix."""
    cm = get_cost_matrix(audience)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Benign (actual)", "Weapons (actual)"],
        y=["Allow (pred)", "Ban (pred)"],
        text=cm.astype(int),
        texttemplate="$%{text}",
        colorscale="Reds",
    ))
    fig.update_layout(
        title=f"Cost matrix ({audience})",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=300,
    )
    return fig


def approach_comparison_table(
    metrics_list: list,
    names: list,
) -> pd.DataFrame:
    """Table of metrics (ece, precision, recall, cost, etc.) per approach."""
    return pd.DataFrame(metrics_list, index=names)
