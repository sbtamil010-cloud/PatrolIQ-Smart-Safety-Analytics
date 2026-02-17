# streamlit_app/pages/4_Dimensionality.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.dimensionality import apply_scaling, run_pca, run_tsne

st.title("📉 Dimensionality Reduction")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/processed_crime.csv")

df = load_data()

features = df[['Latitude','Longitude','Hour','Month','Crime_Type_Encoded','Crime_Severity']]

scaled, scaler = apply_scaling(features)

# PCA
components, pca_model, variance = run_pca(scaled, 2)

st.subheader("PCA Explained Variance")
st.write(variance)

fig, ax = plt.subplots()
ax.scatter(components[:,0], components[:,1], alpha=0.3)
st.pyplot(fig)

# t-SNE
st.subheader("t-SNE Visualization")
tsne_result = run_tsne(scaled)

fig2, ax2 = plt.subplots()
ax2.scatter(tsne_result[:,0], tsne_result[:,1], alpha=0.3)
st.pyplot(fig2)
