"""
InversionAI - Data Preview Component
Reusable functions for displaying dataset previews and validation status.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Any


def render_data_preview(dataset: Any):
    """Render a data preview with stats, histogram, and map.

    Args:
        dataset: A GeoDataset object or pandas DataFrame with columns X, Y,
                 and optionally Z, Value.
    """
    # Handle None gracefully
    if dataset is None:
        st.info("📂 No dataset loaded. Upload data to see a preview.")
        return

    # Convert to DataFrame if needed
    if isinstance(dataset, pd.DataFrame):
        df = dataset
    elif hasattr(dataset, "to_dataframe"):
        df = dataset.to_dataframe()
    elif hasattr(dataset, "data"):
        df = dataset.data if isinstance(dataset.data, pd.DataFrame) else pd.DataFrame()
    else:
        st.warning("Unsupported dataset format for preview.")
        return

    if df.empty:
        st.info("Dataset is empty.")
        return

    # Statistics
    st.markdown("**📊 Dataset Statistics**")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Rows", f"{len(df):,}")
    with stat_col2:
        st.metric("Columns", f"{df.shape[1]}")
    with stat_col3:
        nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        st.metric("Missing %", f"{nan_pct:.1f}%")
    with stat_col4:
        if "Value" in df.columns:
            st.metric("Value Mean", f"{df['Value'].mean():.4f}")

    # Histogram of values
    if "Value" in df.columns:
        st.markdown("**📈 Value Distribution**")
        fig = px.histogram(
            df,
            x="Value",
            nbins=40,
            title="Observed Data Histogram",
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=30, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Map of station locations
    if "X" in df.columns and "Y" in df.columns:
        st.markdown("**🗺️ Station Locations**")
        color_col = "Value" if "Value" in df.columns else None
        if "Z" in df.columns:
            fig = px.scatter_3d(
                df, x="X", y="Y", z="Z", color=color_col,
                color_continuous_scale="Viridis",
                title="Station Locations",
            )
            fig.update_layout(height=400)
        else:
            fig = px.scatter(
                df, x="X", y="Y", color=color_col,
                color_continuous_scale="Viridis",
                title="Station Locations",
            )
            fig.update_layout(height=350, yaxis_scaleanchor="x")

        fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)


def render_validation_status(result: Any):
    """Render colored validation status indicators.

    Args:
        result: A ValidationResult object or dict with boolean fields:
                has_coordinates, has_values, no_nan_coords,
                sufficient_points, value_range_ok, messages (list).
    """
    if result is None:
        st.info("No validation performed yet.")
        return

    # Support both dict and object with attributes
    def _get(key, default=False):
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)

    checks = [
        ("Coordinates Found", _get("has_coordinates")),
        ("Values Present", _get("has_values")),
        ("No NaN in Coords", _get("no_nan_coords")),
        ("Sufficient Points", _get("sufficient_points")),
        ("Value Range OK", _get("value_range_ok")),
    ]

    cols = st.columns(len(checks))
    for col, (label, passed) in zip(cols, checks):
        with col:
            if passed:
                st.markdown(f"🟢 **{label}**")
            else:
                st.markdown(f"🔴 **{label}**")

    # Show messages
    messages = _get("messages", [])
    if messages:
        for msg in messages:
            if "❌" in msg:
                st.error(msg)
            elif "⚠️" in msg:
                st.warning(msg)
            else:
                st.info(msg)

    # Overall status
    all_passed = all(status for _, status in checks)
    if all_passed:
        st.success("✅ All validation checks passed!")
    else:
        failed = [label for label, status in checks if not status]
        st.warning(f"Some checks did not pass: {', '.join(failed)}")
