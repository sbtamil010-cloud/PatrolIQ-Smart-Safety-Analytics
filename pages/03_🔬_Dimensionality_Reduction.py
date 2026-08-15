"""
PatrolIQ - Dimensionality Reduction Visualizations
Interactive PCA, t-SNE, and UMAP visualizations with crime type and time period analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from PIL import Image

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
PCA_2D = BASE_DIR / "reports" / "summaries" / "pca2_embeddings.csv"
PCA_3D = BASE_DIR / "reports" / "summaries" / "pca3_embeddings.csv"
TSNE_2D = BASE_DIR / "reports" / "summaries" / "tsne_embeddings.csv"
UMAP_2D = BASE_DIR / "reports" / "summaries" / "umap_embeddings.csv"
PCA_IMPORTANCE = BASE_DIR / "reports" / "summaries" / "pca_feature_importance.csv"
DIM_SUMMARY = BASE_DIR / "reports" / "summaries" / "dimensionality_summary.json"
SCREE_PLOT = BASE_DIR / "reports" / "figures" / "pca_scree_plot.png"
IMPORTANCE_BAR = BASE_DIR / "reports" / "figures" / "feature_importance_bar.png"

st.set_page_config(page_title="Dimensionality Reduction", page_icon="🔬", layout="wide")

st.title("🔬 Dimensionality Reduction & Pattern Discovery")
st.markdown("""
Explore hidden crime patterns using **PCA**, **t-SNE**, and **UMAP** to compress high-dimensional data 
into 2D/3D visualizations while preserving maximum information.
""")

# Load data
@st.cache_data
def load_embeddings():
    embeddings = {}
    if PCA_2D.exists():
        embeddings['pca'] = pd.read_csv(PCA_2D)
    if PCA_3D.exists():
        embeddings['pca_3d'] = pd.read_csv(PCA_3D)
    if TSNE_2D.exists():
        embeddings['tsne'] = pd.read_csv(TSNE_2D)
    if UMAP_2D.exists():
        embeddings['umap'] = pd.read_csv(UMAP_2D)
    return embeddings

@st.cache_data
def load_importance():
    if PCA_IMPORTANCE.exists():
        return pd.read_csv(PCA_IMPORTANCE, index_col=0)
    return None

@st.cache_data
def load_summary():
    if DIM_SUMMARY.exists():
        with open(DIM_SUMMARY) as f:
            return json.load(f)
    return None

embeddings = load_embeddings()
importance_df = load_importance()
summary = load_summary()

# Overview metrics
if summary:
    st.subheader("📊 Dimensionality Reduction Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_features = summary.get('total_features', 0)
        st.metric(
            "Original Features",
            total_features,
            help="Number of features before dimensionality reduction"
        )
    
    with col2:
        variance_2d = summary.get('pca_2_components', {}).get('explained_variance', 0)
        st.metric(
            "PCA 2D Variance",
            f"{variance_2d*100:.1f}%",
            help="Percentage of variance captured by 2 principal components"
        )
    
    with col3:
        variance_3d = summary.get('pca_3_components', {}).get('explained_variance', 0)
        st.metric(
            "PCA 3D Variance",
            f"{variance_3d*100:.1f}%",
            help="Percentage of variance captured by 3 principal components"
        )
    
    with col4:
        n_70 = summary.get('components_for_70_percent', 0)
        st.metric(
            "Components for 70%",
            n_70,
            help="Number of components needed to capture 70% variance"
        )

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 PCA Analysis",
    "🔍 Feature Importance",
    "🎨 t-SNE Visualization",
    "✨ UMAP Projection",
    "🔄 Method Comparison",
    "📊 3D Visualizations"
])

# ----------------------------- TAB 1: PCA ANALYSIS -----------------------------
with tab1:
    st.header("Principal Component Analysis (PCA)")
    
    st.markdown("""
    **PCA** is a linear dimensionality reduction technique that:
    - Compresses high-dimensional data to 2-3 components
    - Preserves maximum variance (information)
    - Reveals which features contribute most to patterns
    """)
    
    # Scree plot
    if SCREE_PLOT.exists():
        st.subheader("📊 Scree Plot: Variance Explained")
        st.image(str(SCREE_PLOT), use_container_width=True)
        
        if summary:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Key Findings:**
                - 2 components: {summary['pca_2_components']['explained_variance']*100:.1f}% variance
                - 3 components: {summary['pca_3_components']['explained_variance']*100:.1f}% variance
                - Need {summary['components_for_70_percent']} components for 70% variance
                """)
            with col2:
                st.success(f"""
                **Target Achievement:**
                - ✅ Successfully compressed {summary['total_features']} features
                - ✅ 3 components capture {summary['pca_3_components']['explained_variance']*100:.1f}% information
                - ✅ Reduced complexity while preserving patterns
                """)
    
    st.markdown("---")
    
    # PCA 2D scatter plots
    if 'pca' in embeddings:
        st.subheader("🔵 PCA 2D Visualizations")
        
        pca_df = embeddings['pca']
        
        # Choose visualization type
        viz_option = st.radio(
            "Color code by:",
            ["Crime Type", "Time Period (Day/Night)", "Geographic Cluster", "Temporal Cluster"],
            horizontal=True
        )
        
        # Determine color column
        color_col = None
        if viz_option == "Crime Type" and "Primary Type" in pca_df.columns:
            color_col = "Primary Type"
            title = "PCA 2D: Crime Patterns by Type"
            info_text = "💡 Similar crime types should cluster together. Look for natural groupings and outliers."
        elif viz_option == "Time Period (Day/Night)" and "Time_Period" in pca_df.columns:
            color_col = "Time_Period"
            title = "PCA 2D: Day vs Night Crime Patterns"
            info_text = "💡 Day crimes (gold) vs Night crimes (blue). Notice if temporal patterns create distinct clusters."
        elif viz_option == "Geographic Cluster" and "KMeans_Geo" in pca_df.columns:
            color_col = "KMeans_Geo"
            title = "PCA 2D: Geographic Crime Clusters"
            info_text = "💡 Geographic clusters from spatial analysis. See how location patterns relate to other features."
        elif viz_option == "Temporal Cluster" and "TemporalCluster" in pca_df.columns:
            color_col = "TemporalCluster"
            title = "PCA 2D: Temporal Crime Patterns"
            info_text = "💡 Time-based clusters (late-night, rush-hour, etc.). Check how temporal patterns separate."
        
        if color_col:
            # Get variance percentages
            if summary:
                pc1_var = summary['pca_2_components']['component_1_variance'] * 100
                pc2_var = summary['pca_2_components']['component_2_variance'] * 100
            else:
                pc1_var, pc2_var = 0, 0
            
            # Sample for performance if dataset is large
            plot_df = pca_df
            if len(pca_df) > 10000:
                plot_df = pca_df.sample(n=10000, random_state=42)
                st.caption(f"📌 Displaying 10,000 sampled points (total: {len(pca_df):,})")
            
            fig = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color=color_col,
                title=title,
                labels={
                    'PC1': f'PC1 ({pc1_var:.1f}% variance)',
                    'PC2': f'PC2 ({pc2_var:.1f}% variance)'
                },
                opacity=0.6,
                hover_data=['Primary Type', 'Hour', 'Arrest'] if all(c in plot_df.columns for c in ['Primary Type', 'Hour', 'Arrest']) else None,
                height=600
            )
            
            fig.update_layout(
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.info(info_text)
        else:
            st.warning(f"⚠️ Data for '{viz_option}' visualization not available")
    else:
        st.warning("⚠️ PCA embeddings not found. Please run dimensionality_reduction.py")

# ----------------------------- TAB 2: FEATURE IMPORTANCE -----------------------------
with tab2:
    st.header("🔍 Feature Importance Analysis")
    
    st.markdown("""
    **Which features matter most?** PCA reveals which original features contribute most to the principal components.
    Higher importance = more influential in explaining crime patterns.
    """)
    
    if importance_df is not None:
        # Top features bar chart
        if IMPORTANCE_BAR.exists():
            st.subheader("🏆 Top 15 Most Important Features")
            st.image(str(IMPORTANCE_BAR), use_container_width=True)
        
        st.markdown("---")
        
        # Key insights
        if summary and 'top_5_features' in summary:
            st.subheader("💡 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 5 Most Important Features:**")
                for i, feat in enumerate(summary['top_5_features'], 1):
                    score = summary['feature_importance_scores'].get(feat, 0)
                    st.write(f"{i}. **{feat}** ({score:.4f})")
                
                st.success("✅ These features have the strongest influence on crime patterns")
            
            with col2:
                st.write("**What this means:**")
                st.markdown("""
                - **Location features** (Latitude, Longitude) likely rank high
                - **Time features** (Hour, Weekday, Month) capture temporal patterns
                - **Crime type indicators** distinguish different crime categories
                - High-importance features should be prioritized in patrol planning
                """)
        
        st.markdown("---")
        
        # Detailed loadings table
        st.subheader("📋 Feature Loadings on Principal Components")
        
        st.markdown("""
        **How to read this table:**
        - **PC1, PC2, PC3**: Loading values for each component
        - **Total_Importance**: Sum of absolute loadings across all components
        - Darker colors indicate stronger contributions
        """)
        
        # Display with conditional formatting
        styled_df = importance_df.style.background_gradient(
            cmap='RdYlGn',
            subset=['PC1', 'PC2', 'PC3'],
            vmin=-1,
            vmax=1
        ).background_gradient(
            cmap='Blues',
            subset=['Total_Importance']
        ).format("{:.4f}")
        
        st.dataframe(styled_df, use_container_width=True, height=500)
        
        # Download button
        csv = importance_df.to_csv()
        st.download_button(
            label="📥 Download Feature Importance CSV",
            data=csv,
            file_name="pca_feature_importance.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Feature importance data not found. Please run dimensionality_reduction.py")

# ----------------------------- TAB 3: t-SNE VISUALIZATION -----------------------------
with tab3:
    st.header("🎨 t-SNE Visualization")
    
    st.markdown("""
    **t-SNE** (t-Distributed Stochastic Neighbor Embedding) creates beautiful visualizations where:
    - Similar crimes cluster together naturally
    - Non-linear patterns become visible
    - Local structure is preserved better than PCA
    """)
    
    if 'tsne' in embeddings:
        tsne_df = embeddings['tsne']
        
        st.subheader("🎯 Interactive t-SNE Plots")
        
        # Choose color scheme
        available_cols = []
        col_labels = {}
        
        if 'Primary Type' in tsne_df.columns:
            available_cols.append('Primary Type')
            col_labels['Primary Type'] = "Crime Type"
        if 'Time_Period' in tsne_df.columns:
            available_cols.append('Time_Period')
            col_labels['Time_Period'] = "Day/Night Period"
        for col in ['KMeans_Geo', 'HDBSCAN_Geo', 'DBSCAN_Geo']:
            if col in tsne_df.columns:
                available_cols.append(col)
                col_labels[col] = col.replace('_', ' ')
        if 'TemporalCluster' in tsne_df.columns:
            available_cols.append('TemporalCluster')
            col_labels['TemporalCluster'] = "Temporal Pattern"
        
        if available_cols:
            color_choice = st.selectbox(
                "Color clusters by:",
                available_cols,
                format_func=lambda x: col_labels.get(x, x),
                key="tsne_color"
            )
            
            # Sample for performance
            plot_df = tsne_df
            if len(tsne_df) > 10000:
                plot_df = tsne_df.sample(n=10000, random_state=42)
                st.caption(f"📌 Displaying 10,000 sampled points (total: {len(tsne_df):,})")
            
            # Create plot
            hover_cols = ['Primary Type', 'Time_Period', 'Hour'] if all(c in plot_df.columns for c in ['Primary Type', 'Time_Period', 'Hour']) else None
            
            fig = px.scatter(
                plot_df,
                x='TSNE1',
                y='TSNE2',
                color=color_choice,
                title=f"t-SNE: Clustering by {col_labels.get(color_choice, color_choice)}",
                labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
                opacity=0.7,
                hover_data=hover_cols,
                height=600
            )
            
            fig.update_layout(
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation guide
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **What to look for:**
                - 🔵 **Tight clusters**: Similar crimes grouped together
                - 🔴 **Outliers**: Unusual or rare crime patterns
                - 🟢 **Separation**: Different crime types forming distinct groups
                - 🟡 **Overlap**: Crime types with similar characteristics
                """)
            
            with col2:
                st.markdown("""
                **Compare with PCA:**
                - t-SNE shows tighter, more separated clusters
                - Better at revealing subtle patterns
                - Non-linear relationships become visible
                - Good for identifying anomalies
                """)
        else:
            st.warning("⚠️ No color coding columns available in t-SNE data")
    else:
        st.warning("⚠️ t-SNE embeddings not found. Please run dimensionality_reduction.py")

# ----------------------------- TAB 4: UMAP PROJECTION -----------------------------
with tab4:
    st.header("✨ UMAP Projection")
    
    st.markdown("""
    **UMAP** (Uniform Manifold Approximation and Projection) provides:
    - Balance between local and global structure
    - Faster computation than t-SNE
    - Better preservation of data topology
    - Deterministic results (unlike t-SNE)
    """)
    
    if 'umap' in embeddings:
        umap_df = embeddings['umap']
        
        st.subheader("✨ UMAP 2D Projection")
        
        # Choose color scheme (same options as t-SNE)
        available_cols = []
        col_labels = {}
        
        if 'Primary Type' in umap_df.columns:
            available_cols.append('Primary Type')
            col_labels['Primary Type'] = "Crime Type"
        if 'Time_Period' in umap_df.columns:
            available_cols.append('Time_Period')
            col_labels['Time_Period'] = "Day/Night Period"
        for col in ['KMeans_Geo', 'HDBSCAN_Geo', 'DBSCAN_Geo']:
            if col in umap_df.columns:
                available_cols.append(col)
                col_labels[col] = col.replace('_', ' ')
        if 'TemporalCluster' in umap_df.columns:
            available_cols.append('TemporalCluster')
            col_labels['TemporalCluster'] = "Temporal Pattern"
        
        if available_cols:
            color_choice = st.selectbox(
                "Color clusters by:",
                available_cols,
                format_func=lambda x: col_labels.get(x, x),
                key="umap_color"
            )
            
            # Sample for performance
            plot_df = umap_df
            if len(umap_df) > 10000:
                plot_df = umap_df.sample(n=10000, random_state=42)
                st.caption(f"📌 Displaying 10,000 sampled points (total: {len(umap_df):,})")
            
            fig = px.scatter(
                plot_df,
                x='UMAP1',
                y='UMAP2',
                color=color_choice,
                title=f"UMAP: Crime Pattern Manifold by {col_labels.get(color_choice, color_choice)}",
                labels={'UMAP1': 'UMAP Component 1', 'UMAP2': 'UMAP Component 2'},
                opacity=0.6,
                height=600
            )
            
            fig.update_layout(
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            💡 **UMAP advantages**: 
            - Faster than t-SNE for large datasets
            - Preserves both local neighborhoods and global structure
            - Consistent results (same seed = same output)
            - Good for exploratory analysis and production pipelines
            """)
        else:
            st.warning("⚠️ No color coding columns available in UMAP data")
    else:
        st.info("⚠️ UMAP embeddings not found. Install umap-learn and run dimensionality_reduction.py")

# ----------------------------- TAB 5: METHOD COMPARISON -----------------------------
with tab5:
    st.header("🔄 Comparison: PCA vs t-SNE vs UMAP")
    
    st.markdown("""
    Each dimensionality reduction technique has strengths and weaknesses. 
    Compare them side-by-side to validate findings and choose the best method for your use case.
    """)
    
    # Side-by-side comparisons
    if 'pca' in embeddings and 'tsne' in embeddings:
        st.subheader("📊 Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧭 PCA (Linear)")
            pca_df = embeddings['pca']
            
            if len(pca_df) > 5000:
                pca_sample = pca_df.sample(n=5000, random_state=42)
            else:
                pca_sample = pca_df
            
            color_col = 'Primary Type' if 'Primary Type' in pca_sample.columns else None
            
            if color_col:
                fig_pca = px.scatter(
                    pca_sample,
                    x='PC1',
                    y='PC2',
                    color=color_col,
                    title="PCA: Linear Projection",
                    opacity=0.6,
                    height=400
                )
                fig_pca.update_layout(showlegend=False)
                st.plotly_chart(fig_pca, use_container_width=True)
        
        with col2:
            st.markdown("### 🎨 t-SNE (Non-linear)")
            tsne_df = embeddings['tsne']
            
            if len(tsne_df) > 5000:
                tsne_sample = tsne_df.sample(n=5000, random_state=42)
            else:
                tsne_sample = tsne_df
            
            color_col = 'Primary Type' if 'Primary Type' in tsne_sample.columns else None
            
            if color_col:
                fig_tsne = px.scatter(
                    tsne_sample,
                    x='TSNE1',
                    y='TSNE2',
                    color=color_col,
                    title="t-SNE: Non-linear Projection",
                    opacity=0.6,
                    height=400
                )
                fig_tsne.update_layout(showlegend=False)
                st.plotly_chart(fig_tsne, use_container_width=True)
    
    st.markdown("---")
    
    # Comparison table
    st.subheader("📋 Method Characteristics")
    
    comparison_data = {
        "Characteristic": [
            "Transformation Type",
            "Global Structure",
            "Local Structure",
            "Computation Speed",
            "Deterministic",
            "Interpretability",
            "Best Use Case"
        ],
        "PCA": [
            "Linear",
            "✅ Excellent",
            "⚠️ Moderate",
            "⚡ Very Fast",
            "✅ Yes",
            "🏆 High",
            "Initial exploration, variance analysis"
        ],
        "t-SNE": [
            "Non-linear",
            "❌ Poor",
            "✅ Excellent",
            "🐌 Slow",
            "❌ No",
            "⚠️ Low",
            "Cluster visualization, pattern discovery"
        ],
        "UMAP": [
            "Non-linear",
            "✅ Good",
            "✅ Excellent",
            "⚡ Fast",
            "✅ Yes",
            "⚠️ Moderate",
            "Production pipelines, balanced analysis"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **Use PCA when:**
        - Need to explain variance
        - Want interpretable results
        - Require fast computation
        - Building statistical models
        """)
    
    with col2:
        st.info("""
        **Use t-SNE when:**
        - Visualizing clusters
        - Exploring local patterns
        - Presenting to stakeholders
        - Don't need reproducibility
        """)
    
    with col3:
        st.warning("""
        **Use UMAP when:**
        - Building production systems
        - Need consistent results
        - Working with large datasets
        - Want balanced structure
        """)

# ----------------------------- TAB 6: 3D VISUALIZATIONS -----------------------------
with tab6:
    st.header("📊 3D Visualizations")
    
    st.markdown("""
    Explore crime patterns in 3D space using PCA's third component for additional perspective.
    """)
    
    if 'pca_3d' in embeddings:
        pca_3d_df = embeddings['pca_3d']
        
        # Sample for performance
        if len(pca_3d_df) > 5000:
            plot_df = pca_3d_df.sample(n=5000, random_state=42)
            st.caption(f"📌 Displaying 5,000 sampled points (total: {len(pca_3d_df):,})")
        else:
            plot_df = pca_3d_df
        
        # Choose color scheme
        color_options = []
        if 'Primary Type' in plot_df.columns:
            color_options.append('Primary Type')
        if 'Time_Period' in plot_df.columns:
            color_options.append('Time_Period')
        if 'KMeans_Geo' in plot_df.columns:
            color_options.append('KMeans_Geo')
        
        if color_options:
            color_col = st.selectbox("Color by:", color_options, key="3d_color")
            
            fig = px.scatter_3d(
                plot_df,
                x='PC1',
                y='PC2',
                z='PC3',
                color=color_col,
                title="PCA 3D: Crime Pattern Space",
                labels={
                    'PC1': 'Principal Component 1',
                    'PC2': 'Principal Component 2',
                    'PC3': 'Principal Component 3'
                },
                opacity=0.6,
                height=700
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Tip**: Click and drag to rotate the 3D plot. Use scroll to zoom in/out.")
        else:
            st.warning("⚠️ No color coding columns available")
    else:
        st.warning("⚠️ 3D PCA embeddings not found. Please run dimensionality_reduction.py")

# ----------------------------- EXPORT & SUMMARY -----------------------------
st.markdown("---")
st.header("📥 Export & Summary")

if summary:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Analysis Summary")
        summary_text = f"""
        **Dimensionality Reduction Results:**
        - Original Features: {summary['total_features']}
        - PCA 2D Variance: {summary['pca_2_components']['explained_variance']*100:.1f}%
        - PCA 3D Variance: {summary['pca_3_components']['explained_variance']*100:.1f}%
        - Components for 70%: {summary['components_for_70_percent']}
        - Components for 80%: {summary['components_for_80_percent']}
        
        **Top 3 Features:**
        1. {summary['top_5_features'][0]}
        2. {summary['top_5_features'][1]}
        3. {summary['top_5_features'][2]}
        """
        st.code(summary_text)
    
    with col2:
        st.subheader("✅ Analysis Complete")
        st.success(f"""
        **Visualizations Created:**
        - ✅ PCA 2D & 3D projections
        - ✅ t-SNE clustering
        - ✅ UMAP manifold
        - ✅ Feature importance rankings
        - ✅ Scree plot
        - ✅ Crime type & time period analysis
        
        **Total: {len(summary.get('visualizations_created', []))} visualizations**
        """)
        
        if st.button("📥 Download Summary JSON"):
            st.download_button(
                label="Click to Download",
                data=json.dumps(summary, indent=2),
                file_name="dimensionality_reduction_summary.json",
                mime="application/json"
            )