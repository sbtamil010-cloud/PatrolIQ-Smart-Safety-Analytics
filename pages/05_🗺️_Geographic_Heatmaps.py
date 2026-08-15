"""
PatrolIQ - Geographic Crime Heatmaps
Interactive geographic visualizations with cluster boundaries
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH_01 = BASE_DIR / "data" / "processed" / "sample_250000_rows_01.csv"
DATA_PATH_02 = BASE_DIR / "data" / "processed" / "sample_250000_rows_02.csv"
KMEANS_MODEL = BASE_DIR / "models" / "clustering" / "kmeans_geo_k9.pkl"
CENTERS_PATH = BASE_DIR / "reports" / "summaries" / "kmeans_geo_centers_k9.csv"

st.set_page_config(page_title="Geographic Heatmaps", page_icon="🗺️", layout="wide")

st.title("🗺️ Geographic Crime Heatmaps")
st.markdown("Explore crime density across Chicago with interactive maps and clustering analysis")

# Cache data loading
@st.cache_data
def load_data():
    df_01 = pd.read_csv(DATA_PATH_01, low_memory=False)
    df_02 = pd.read_csv(DATA_PATH_02, low_memory=False)
    df = pd.concat([df_01, df_02], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude', 'Date'])
    return df

@st.cache_data
def load_cluster_centers():
    if CENTERS_PATH.exists():
        return pd.read_csv(CENTERS_PATH)
    return None

# Load data
with st.spinner("Loading crime data..."):
    df = load_data()
    centers = load_cluster_centers()

st.success(f"✅ Loaded {len(df):,} crime records")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Crime type filter
crime_types = ['All'] + sorted(df['Primary Type'].unique().tolist())
selected_crime = st.sidebar.selectbox("Crime Type", crime_types)

# Sample size for performance
sample_size = st.sidebar.slider("Sample Size (for performance)", 1000, 100000, 50000, 1000)

# Apply filters
filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['Date'].dt.date >= date_range[0]) &
        (filtered_df['Date'].dt.date <= date_range[1])
    ]

if selected_crime != 'All':
    filtered_df = filtered_df[filtered_df['Primary Type'] == selected_crime]

# Sample for visualization
viz_df = filtered_df.sample(n=min(sample_size, len(filtered_df)), random_state=42)

st.info(f"📊 Displaying {len(viz_df):,} out of {len(filtered_df):,} filtered records")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔥 Density Heatmap", "📍 Scatter Plot", "🎯 Cluster Centers"])

with tab1:
    st.subheader("Crime Density Heatmap")
    
    # Density heatmap
    fig = px.density_mapbox(
        viz_df,
        lat='Latitude',
        lon='Longitude',
        radius=10,
        zoom=10,
        center={"lat": 41.8781, "lon": -87.6298},
        mapbox_style="open-street-map",
        title="Crime Density Across Chicago",
        hover_data=['Primary Type', 'Date']
    )
    
    fig.update_layout(height=700, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Crime Location Scatter Plot")
    
    # Color by crime type
    fig = px.scatter_mapbox(
        viz_df,
        lat='Latitude',
        lon='Longitude',
        color='Primary Type',
        zoom=10,
        center={"lat": 41.8781, "lon": -87.6298},
        mapbox_style="open-street-map",
        title="Crime Locations by Type",
        hover_data=['Primary Type', 'Date', 'Location Description']
    )
    
    fig.update_layout(height=700, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("K-Means Cluster Centers (9 Hotspots)")
    
    if centers is not None:
        # Create map with cluster centers
        fig = go.Figure()
        
        # Add crime points
        fig.add_trace(go.Scattermapbox(
            lat=viz_df['Latitude'],
            lon=viz_df['Longitude'],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.3),
            name='Crime Locations',
            hoverinfo='skip'
        ))
        
        # Add cluster centers
        fig.add_trace(go.Scattermapbox(
            lat=centers['Latitude'],
            lon=centers['Longitude'],
            mode='markers+text',
            marker=dict(size=20, color='red', symbol='star'),
            text=[f"Hotspot {i}" for i in range(len(centers))],
            textposition="top center",
            name='Cluster Centers',
            hovertemplate='<b>Hotspot %{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center={"lat": 41.8781, "lon": -87.6298},
                zoom=10
            ),
            height=700,
            margin={"r":0,"t":40,"l":0,"b":0},
            title="9 Geographic Crime Hotspots (K-Means)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display cluster coordinates
        st.subheader("📍 Hotspot Coordinates")
        centers_display = centers.copy()
        centers_display.index = [f"Hotspot {i}" for i in range(len(centers))]
        st.dataframe(centers_display, use_container_width=True)
    else:
        st.warning("⚠️ Cluster centers not found. Please run geo_clustering.py first.")

# Statistics
st.markdown("---")
st.subheader("📊 Geographic Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Crimes", f"{len(filtered_df):,}")

with col2:
    most_common = filtered_df['Primary Type'].mode()[0]
    st.metric("Most Common Crime", most_common)

with col3:
    if 'Community Area' in filtered_df.columns:
        top_area = filtered_df['Community Area'].mode()[0]
        st.metric("Top Community Area", int(top_area))

with col4:
    if 'Arrest' in filtered_df.columns:
        arrest_rate = (filtered_df['Arrest'].sum() / len(filtered_df)) * 100
        st.metric("Arrest Rate", f"{arrest_rate:.1f}%")