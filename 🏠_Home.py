"""
PatrolIQ - Crime Analysis Dashboard
Main Streamlit Application Entry Point
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Page configuration
st.set_page_config(
    page_title="PatrolIQ - Crime Analysis Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main page
def main():
    st.markdown('<div class="main-header">🚔 PatrolIQ - Crime Analysis Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to PatrolIQ</h3>
    <p>A comprehensive crime analysis platform powered by advanced machine learning algorithms. 
    Navigate through different sections using the sidebar to explore crime patterns, hotspots, and predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>500K+</h2>
            <p>Crime Records Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>33</h2>
            <p>Crime Types Tracked</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>9</h2>
            <p>Geographic Hotspots</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>4</h2>
            <p>Temporal Patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features overview
    st.subheader("📊 Dashboard Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🗺️ Geographic Analysis
        - **Interactive crime heatmaps** with real-time filtering
        - **K-Means cluster visualization** (9 hotspots)
        - **DBSCAN density-based clustering** for anomaly detection
        - **Hierarchical clustering** dendrogram analysis
        
        ### ⏰ Temporal Patterns
        - **Hourly, daily, and seasonal trends**
        - **Peak crime time identification**
        - **Weekday vs weekend analysis**
        - **Multi-year trend comparison**
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Advanced Analytics
        - **PCA & t-SNE visualizations** for pattern discovery
        - **UMAP embeddings** for cluster analysis
        - **Feature importance ranking**
        - **Dimensionality reduction insights**
        
        ### 📈 Model Performance
        - **MLflow experiment tracking**
        - **Real-time model metrics**
        - **Silhouette & Davies-Bouldin scores**
        - **Cluster quality assessment**
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    1. **Geographic Heatmaps** → Explore crime density across Chicago
    2. **Temporal Analysis** → Understand when crimes occur most frequently
    3. **Clustering Results** → View geographic and temporal crime patterns
    4. **Dimensionality Reduction** → Discover hidden patterns in data
    5. **MLflow Monitoring** → Track model performance metrics
    
    👈 **Use the sidebar** to navigate between different analysis modules.
    """)
    
    st.info("💡 **Tip**: All visualizations are interactive. Hover, zoom, and filter to explore the data in detail!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>PatrolIQ v1.0 | Powered by Streamlit, Scikit-learn, MLflow | Chicago Crime Data (2010-2025)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()