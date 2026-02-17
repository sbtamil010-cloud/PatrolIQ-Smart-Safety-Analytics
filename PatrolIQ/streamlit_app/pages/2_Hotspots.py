# streamlit_app/pages/2_Hotspots.py

import streamlit as st
import pandas as pd
import pydeck as pdk
from src.clustering import run_kmeans
from src.feature_engineering import scale_geographic_features

st.title("🔥 Geographic Crime Hotspots")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/processed_crime.csv")

df = load_data()

geo_scaled, scaler = scale_geographic_features(df)

n_clusters = st.slider("Select Number of Clusters", 3, 12, 8)

labels, model, score = run_kmeans(geo_scaled, n_clusters)

df['Cluster'] = labels

st.write("Silhouette Score:", round(score, 3))

layer = pdk.Layer(
    "ScatterplotLayer",
    df,
    get_position='[Longitude, Latitude]',
    get_fill_color='[200, 30, 0, 160]',
    get_radius=50,
)

view_state = pdk.ViewState(
    latitude=df["Latitude"].mean(),
    longitude=df["Longitude"].mean(),
    zoom=10,
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
