"""
InversionAI - Convergence Plot Component
Reusable convergence/misfit plotting functions.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List, Optional


def render_convergence(misfit_history: List[float], algorithm_name: str):
    """Render a single convergence line plot.

    Args:
        misfit_history: List of misfit values per iteration.
        algorithm_name: Name of the algorithm for the title/legend.
    """
    if not misfit_history:
        st.info(f"No convergence data available for {algorithm_name}.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(misfit_history) + 1)),
        y=misfit_history,
        mode="lines+markers",
        name=algorithm_name,
        line=dict(width=2),
        marker=dict(size=4),
    ))

    fig.update_layout(
        title=f"Misfit Convergence — {algorithm_name}",
        xaxis_title="Iteration",
        yaxis_title="Misfit",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(type="log"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_convergence_comparison(histories: Dict[str, List[float]]):
    """Render overlaid convergence curves for multiple algorithms.

    Args:
        histories: Dictionary mapping algorithm name to misfit history list.
    """
    if not histories:
        st.info("No convergence data available.")
        return

    colors = {
        "Tomofast-x": "#1f77b4",
        "SimPEG": "#ff7f0e",
    }

    fig = go.Figure()
    for name, history in histories.items():
        if not history:
            continue
        color = colors.get(name, None)
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=history,
            mode="lines+markers",
            name=name,
            line=dict(width=2, color=color),
            marker=dict(size=3),
        ))

    fig.update_layout(
        title="Misfit Convergence Comparison",
        xaxis_title="Iteration",
        yaxis_title="Misfit",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(type="log"),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)
