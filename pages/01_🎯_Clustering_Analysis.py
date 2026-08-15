"""
PatrolIQ - Clustering Analysis Results
Enhanced Geographic + Temporal Clustering Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import streamlit.components.v1 as components
import joblib

# =======================================================
# PATHS & PAGE CONFIG
# =======================================================
BASE_DIR = Path(__file__).resolve().parents[3]

# Summaries and metrics
GEO_METRICS = BASE_DIR / "reports" / "summaries" / "geo_clustering_metrics.json"
TEMP_METRICS = BASE_DIR / "reports" / "summaries" / "temporal_clustering_metrics.json"
TEMP_SUMMARY = BASE_DIR / "reports" / "summaries" / "temporal_cluster_summary.csv"
CENTERS_PATH = BASE_DIR / "reports" / "summaries" / "kmeans_geo_centers_k9.csv"

# New summaries added from geo_clustering.py
KMEANS_SUMMARY = BASE_DIR / "reports" / "summaries" / "kmeans_geo_crime_summary.csv"
DBSCAN_SUMMARY = BASE_DIR / "reports" / "summaries" / "dbscan_geo_crime_summary.csv"
HIER_SUMMARY = BASE_DIR / "reports" / "summaries" / "cluster_crime_summary.csv"

# Figures & Maps
MAP_KMEANS = BASE_DIR / "reports" / "figures" / "map_kmeans_geo_k9.html"
MAP_DBSCAN = BASE_DIR / "reports" / "figures" / "map_dbscan_geo.html"
MAP_RISK = BASE_DIR / "reports" / "figures" / "map_kmeans_risk_zones.html"
DENDROGRAM_PATH = BASE_DIR / "reports" / "figures" / "dendrogram_geo.png"

# -------------------------------------------------------
st.set_page_config(page_title="Clustering Results", page_icon="🎯", layout="wide")
st.title("🎯 Clustering Analysis Results")
st.markdown("Explore geographic and temporal crime patterns identified by clustering algorithms.")

# =======================================================
# LOADERS
# =======================================================
@st.cache_data
def load_metrics():
    metrics = {}
    if GEO_METRICS.exists():
        with open(GEO_METRICS) as f:
            metrics['geo'] = json.load(f)
    if TEMP_METRICS.exists():
        with open(TEMP_METRICS) as f:
            metrics['temp'] = json.load(f)
    return metrics

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None

metrics = load_metrics()
temp_summary = load_csv(TEMP_SUMMARY)
centers = load_csv(CENTERS_PATH)
kmeans_summary = load_csv(KMEANS_SUMMARY)
dbscan_summary = load_csv(DBSCAN_SUMMARY)
hier_summary = load_csv(HIER_SUMMARY)

# =======================================================
# TABS
# =======================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 K-Means Clustering",
    "🔍 DBSCAN Clustering",
    "🌳 Hierarchical Clustering",
    "🔥 Risk Heatmap",
    "📊 Performance Metrics"
])

# =======================================================
# TAB 1 — KMEANS
# =======================================================
with tab1:
    st.header("📍 K-Means Clustering (9 Hotspots)")

    if 'geo' in metrics:
        kmeans_metrics = metrics['geo']['kmeans']
        col1, col2 = st.columns(2)
        with col1:
            if kmeans_metrics['silhouette'] is not None:
                st.metric("Silhouette Score", f"{kmeans_metrics['silhouette']:.4f}")
        with col2:
            if kmeans_metrics['davies_bouldin'] is not None:
                st.metric("Davies-Bouldin Index", f"{kmeans_metrics['davies_bouldin']:.4f}")

    # Hotspot Map
    if MAP_KMEANS.exists():
        st.subheader("🗺️ K-Means Hotspot Map")
        with open(MAP_KMEANS, "r", encoding="utf-8") as f:
            components.html(f.read(), height=600, scrolling=False)
    else:
        st.warning("⚠️ K-Means map not found.")

    # Hotspot Centers
    if centers is not None:
        st.subheader("🎯 Hotspot Center Coordinates")
        centers.index = [f"Hotspot {i}" for i in range(len(centers))]
        st.dataframe(centers, use_container_width=True)

    # Cluster Crime Summary
    if kmeans_summary is not None:
        st.subheader("📊 Cluster Crime Summary (K-Means)")
        st.dataframe(kmeans_summary, use_container_width=True)

# =======================================================
# TAB 2 — DBSCAN
# =======================================================
with tab2:
    st.header("🔍 DBSCAN Clustering")

    if 'geo' in metrics:
        dbscan_metrics = metrics['geo']['dbscan']
        col1, col2 = st.columns(2)
        with col1:
            if dbscan_metrics['silhouette'] is not None:
                st.metric("Silhouette Score", f"{dbscan_metrics['silhouette']:.4f}")
        with col2:
            if dbscan_metrics['davies_bouldin'] is not None:
                st.metric("Davies-Bouldin Index", f"{dbscan_metrics['davies_bouldin']:.4f}")

    st.info("DBSCAN identifies high-density crime zones and filters isolated incidents.")

    if MAP_DBSCAN.exists():
        st.subheader("🗺️ DBSCAN Crime Density Map")
        with open(MAP_DBSCAN, "r", encoding="utf-8") as f:
            components.html(f.read(), height=600, scrolling=False)
    else:
        st.warning("⚠️ DBSCAN map not found.")

    if dbscan_summary is not None:
        st.subheader("📊 Cluster Crime Summary (DBSCAN)")
        st.dataframe(dbscan_summary, use_container_width=True)

# =======================================================
# TAB 3 — HIERARCHICAL
# =======================================================
with tab3:
    st.header("🌳 Hierarchical Clustering — Geographic Relationships")

    if 'geo' in metrics:
        hier_metrics = metrics['geo']['hierarchical']
        col1, col2 = st.columns(2)
        with col1:
            if hier_metrics['silhouette'] is not None:
                st.metric("Silhouette Score", f"{hier_metrics['silhouette']:.4f}")
        with col2:
            if hier_metrics['davies_bouldin'] is not None:
                st.metric("Davies-Bouldin Index", f"{hier_metrics['davies_bouldin']:.4f}")

    if DENDROGRAM_PATH.exists():
        st.image(str(DENDROGRAM_PATH), caption="Hierarchical Dendrogram — Geographic Crime Zones", use_container_width=True)
        st.info("📊 The dendrogram illustrates nested relationships between neighborhoods.")
    else:
        st.warning("⚠️ Dendrogram not found.")

    if hier_summary is not None:
        st.subheader("📊 Cluster Crime Summary (Hierarchical)")
        st.dataframe(hier_summary, use_container_width=True)

# =======================================================
# TAB 4 — RISK HEATMAP
# =======================================================
with tab4:
    st.header("🔥 Crime Risk-Level Heatmap")

    if MAP_RISK.exists():
        with open(MAP_RISK, "r", encoding="utf-8") as f:
            components.html(f.read(), height=600, scrolling=False)
        st.markdown("""
        **Color Legend:**
        - 🔴 **Red** — High Risk (Top 20% crime density)
        - 🟠 **Orange** — Medium Risk (10–20%)
        - 🟢 **Green** — Low Risk (<10%)
        """)
    else:
        st.warning("⚠️ Risk heatmap not found. Please run geo_clustering.py again.")

# =======================================================
# TAB 5 — PERFORMANCE COMPARISON
# =======================================================
with tab5:
    st.header("📊 Algorithm Performance Comparison")

    comparison_data = []
    if 'geo' in metrics:
        for algo, vals in metrics['geo'].items():
            if vals['silhouette'] is not None:
                comparison_data.append({
                    'Algorithm': f"Geo {algo.upper()}",
                    'Silhouette': vals['silhouette'],
                    'Davies-Bouldin': vals['davies_bouldin'] if vals['davies_bouldin'] else 0
                })
    if 'temp' in metrics and metrics['temp'].get('silhouette') is not None:
        comparison_data.append({
            'Algorithm': 'Temporal KMeans',
            'Silhouette': metrics['temp']['silhouette'],
            'Davies-Bouldin': metrics['temp']['davies_bouldin'] if metrics['temp']['davies_bouldin'] else 0
        })

    if comparison_data:
        df_comp = pd.DataFrame(comparison_data)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df_comp, x='Algorithm', y='Silhouette', color='Silhouette',
                         color_continuous_scale='Viridis', title="Silhouette Score (Higher = Better)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(df_comp, x='Algorithm', y='Davies-Bouldin', color='Davies-Bouldin',
                         color_continuous_scale='Reds_r', title="Davies-Bouldin Index (Lower = Better)")
            st.plotly_chart(fig, use_container_width=True)

        best_algo = df_comp.loc[df_comp['Silhouette'].idxmax(), 'Algorithm']
        best_score = df_comp['Silhouette'].max()
        st.success(f"🏆 **Best Performing Algorithm:** {best_algo} (Silhouette = {best_score:.3f})")
    else:
        st.warning("⚠️ No metrics available. Please rerun geo_clustering.py")

    st.markdown("---")
    st.subheader("📘 Metric Guide")
    st.markdown("""
    - **Silhouette Score**: Higher = better separation between clusters (ideal ≥ 0.5)  
    - **Davies-Bouldin Index**: Lower = better cluster distinctness (ideal ≤ 1.0)
    """)

# =======================================================
# INSIGHTS SECTION
# =======================================================
st.markdown("---")
st.subheader("💡 Key Insights")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Geographic Clustering**
    - ✅ 9 distinct patrol hotspots identified (KMeans)
    - ✅ DBSCAN revealed organic dense areas
    - ✅ Hierarchical clustering visualized inter-zone relationships
    """)
with col2:
    st.markdown("""
    **Temporal Clustering**
    - 🕓 Found 4 major time-based crime patterns
    - 🌃 Differentiated day vs night crime behaviors
    - 📆 Highlighted peak months and repeat intervals
    """)

