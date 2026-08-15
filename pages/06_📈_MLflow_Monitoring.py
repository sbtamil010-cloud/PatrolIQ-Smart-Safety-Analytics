"""
PatrolIQ - MLflow Model Monitoring
Track experiments, metrics, and model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
GEO_METRICS = BASE_DIR / "reports" / "summaries" / "geo_clustering_metrics.json"
TEMP_METRICS = BASE_DIR / "reports" / "summaries" / "temporal_clustering_metrics.json"
DIM_SUMMARY = BASE_DIR / "reports" / "summaries" / "dimensionality_summary.json"

st.set_page_config(page_title="MLflow Monitoring", page_icon="📈", layout="wide")

st.title("📈 MLflow Model Monitoring & Performance")
st.markdown("Track experiment metrics and model performance across different algorithms")

# Load all metrics
@st.cache_data
def load_all_metrics():
    metrics = {}
    
    try:
        if GEO_METRICS.exists():
            with open(GEO_METRICS, 'r', encoding='utf-8') as f:
                metrics['geo'] = json.load(f)
        else:
            st.warning(f"⚠️ Geographic metrics not found at: {GEO_METRICS}")
    except Exception as e:
        st.error(f"❌ Error loading geo metrics: {e}")
    
    try:
        if TEMP_METRICS.exists():
            with open(TEMP_METRICS, 'r', encoding='utf-8') as f:
                metrics['temp'] = json.load(f)
        else:
            st.warning(f"⚠️ Temporal metrics not found at: {TEMP_METRICS}")
    except Exception as e:
        st.error(f"❌ Error loading temp metrics: {e}")
    
    try:
        if DIM_SUMMARY.exists():
            with open(DIM_SUMMARY, 'r', encoding='utf-8') as f:
                metrics['dim'] = json.load(f)
        else:
            st.warning(f"⚠️ Dimensionality metrics not found at: {DIM_SUMMARY}")
    except Exception as e:
        st.error(f"❌ Error loading dim metrics: {e}")
    
    return metrics

# Display loading status
with st.spinner("Loading experiment metrics..."):
    metrics = load_all_metrics()

# Show what was loaded
if metrics:
    st.success(f"✅ Loaded {len(metrics)} experiment result(s)")
    # Debug info (can be removed later)
    with st.expander("🔍 Debug: Loaded Data Structure"):
        st.json(metrics)
else:
    st.error("❌ No metrics found. Please run the following scripts first:")
    st.code("""
python src/models/geo_clustering.py
python src/models/temporal_clustering.py
python src/models/dimensionality_reduction.py
    """, language="bash")

# Overview
st.header("🎯 Experiment Overview")

# Check if we have data
if not metrics:
    st.warning("⚠️ No experiment data found. Please run the modeling scripts first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Experiments", len(metrics), help="Geographic, Temporal, and Dimensionality Reduction")

with col2:
    total_models = 0
    if 'geo' in metrics:
        total_models += len(metrics['geo'])
    if 'temp' in metrics:
        total_models += 1
    if 'dim' in metrics:
        total_models += 1
    st.metric("Total Models", str(total_models))

with col3:
    tracked_metrics = 0
    for exp in metrics.values():
        if isinstance(exp, dict):
            for key, value in exp.items():
                if isinstance(value, dict):
                    # Count non-null metrics in nested dicts
                    tracked_metrics += len([v for v in value.values() if v is not None])
                elif value is not None and key != 'top_features':
                    # Count top-level metrics
                    tracked_metrics += 1
    st.metric("Tracked Metrics", str(tracked_metrics))

with col4:
    st.metric("MLflow Server", "Ready", delta="Port 5000", delta_color="normal")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance",
    "🔬 Experiment Comparison",
    "📉 Metrics Dashboard",
    "🔗 MLflow UI"
])

with tab1:
    st.header("Model Performance Summary")
    
    # Check if we have any metrics
    has_data = False
    
    # Geographic clustering performance
    st.subheader("🗺️ Geographic Clustering Models")
    
    if 'geo' in metrics and metrics['geo']:
        has_data = True
        geo_data = []
        
        for model_name, model_metrics in metrics['geo'].items():
            if isinstance(model_metrics, dict):
                geo_data.append({
                    'Model': model_name.upper(),
                    'Silhouette Score': model_metrics.get('silhouette'),
                    'Davies-Bouldin Index': model_metrics.get('davies_bouldin'),
                    'Status': '✅ Good' if model_metrics.get('silhouette', 0) and model_metrics.get('silhouette', 0) > 0.3 else '⚠️ Fair'
                })
        
        if geo_data:
            geo_df = pd.DataFrame(geo_data)
            
            # Display table
            st.dataframe(
                geo_df.style.background_gradient(subset=['Silhouette Score'], cmap='Greens')
                      .background_gradient(subset=['Davies-Bouldin Index'], cmap='Reds_r'),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualize performance
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    geo_df,
                    x='Model',
                    y='Silhouette Score',
                    title="Silhouette Score by Model",
                    color='Silhouette Score',
                    color_continuous_scale='Viridis'
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                            annotation_text="Good threshold")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    geo_df,
                    x='Model',
                    y='Davies-Bouldin Index',
                    title="Davies-Bouldin Index by Model",
                    color='Davies-Bouldin Index',
                    color_continuous_scale='Reds_r'
                )
                fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                            annotation_text="Good threshold")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No geographic clustering models found in the data structure")
    else:
        st.warning("⚠️ Geographic clustering metrics not found. Run: `python src/models/geo_clustering.py`")
    
    st.markdown("---")
    
    # Temporal clustering performance
    st.subheader("⏰ Temporal Clustering Model")
    
    if 'temp' in metrics and metrics['temp']:
        has_data = True
        temp_metrics = metrics['temp']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Algorithm", "K-Means")
            st.metric("Number of Clusters", temp_metrics.get('k', 'N/A'))
        with col2:
            if temp_metrics.get('silhouette') is not None:
                st.metric("Silhouette Score", f"{temp_metrics['silhouette']:.4f}")
        with col3:
            if temp_metrics.get('davies_bouldin') is not None:
                st.metric("Davies-Bouldin Index", f"{temp_metrics['davies_bouldin']:.4f}")
        
        # Status indicator
        sil_score = temp_metrics.get('silhouette', 0)
        if sil_score and sil_score > 0.5:
            st.success("✅ Model performance: Excellent")
        elif sil_score and sil_score > 0.3:
            st.info("ℹ️ Model performance: Good")
        elif sil_score:
            st.warning("⚠️ Model performance: Fair - Consider parameter tuning")
        else:
            st.warning("⚠️ Silhouette score not available")
    else:
        st.warning("⚠️ Temporal clustering metrics not found. Run: `python src/models/temporal_clustering.py`")
    
    st.markdown("---")
    
    # Dimensionality reduction
    st.subheader("🔬 Dimensionality Reduction")
    
    if 'dim' in metrics and metrics['dim']:
        has_data = True
        dim_metrics = metrics['dim']
        
        col1, col2 = st.columns(2)
        
        with col1:
            variance = dim_metrics.get('explained_variance', 0)
            st.metric(
                "Explained Variance (3 PCs)",
                f"{variance*100:.1f}%",
                help="Percentage of variance captured by first 3 components"
            )
        
        with col2:
            if variance > 0.7:
                st.success("✅ Excellent dimensionality reduction")
            elif variance > 0.5:
                st.info("ℹ️ Good dimensionality reduction")
            else:
                st.warning("⚠️ Consider using more components")
        
        # Display top features if available
        if 'top_features' in dim_metrics:
            st.subheader("🏆 Top 5 Important Features")
            for i, feat in enumerate(dim_metrics['top_features'][:5], 1):
                st.write(f"{i}. **{feat}**")
    else:
        st.warning("⚠️ Dimensionality reduction metrics not found. Run: `python src/models/dimensionality_reduction.py`")
    
    if not has_data:
        st.error("❌ No model performance data available. Please run the modeling scripts first.")

with tab2:
    st.header("🔬 Cross-Experiment Comparison")
    
    st.subheader("Clustering Quality Comparison")
    
    # Gather all silhouette scores
    comparison_data = []
    
    if 'geo' in metrics and metrics['geo']:
        for model_name, model_metrics in metrics['geo'].items():
            if isinstance(model_metrics, dict) and model_metrics.get('silhouette') is not None:
                comparison_data.append({
                    'Experiment': 'Geographic',
                    'Model': model_name.upper(),
                    'Silhouette Score': model_metrics['silhouette'],
                    'Davies-Bouldin': model_metrics.get('davies_bouldin', 0)
                })
    
    if 'temp' in metrics and metrics['temp'] and metrics['temp'].get('silhouette') is not None:
        comparison_data.append({
            'Experiment': 'Temporal',
            'Model': 'K-Means',
            'Silhouette Score': metrics['temp']['silhouette'],
            'Davies-Bouldin': metrics['temp'].get('davies_bouldin', 0)
        })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Grouped bar chart
        fig = px.bar(
            comp_df,
            x='Model',
            y='Silhouette Score',
            color='Experiment',
            barmode='group',
            title="Silhouette Score Comparison Across Experiments",
            labels={'Silhouette Score': 'Silhouette Score'},
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                     annotation_text="Good threshold (0.5)")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Best performing model
        st.subheader("🏆 Best Performing Models")
        
        best_model = max(comparison_data, key=lambda x: x['Silhouette Score'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Model", f"{best_model['Model']}")
        with col2:
            st.metric("Experiment", best_model['Experiment'])
        with col3:
            st.metric("Silhouette Score", f"{best_model['Silhouette Score']:.4f}")
    else:
        st.warning("⚠️ No comparison data available. Please run the clustering scripts first.")
        st.code("""
# Run these commands to generate metrics:
python src/models/geo_clustering.py
python src/models/temporal_clustering.py
        """, language="bash")

with tab3:
    st.header("📉 Comprehensive Metrics Dashboard")
    
    # Create metrics overview
    all_metrics_list = []
    
    if 'geo' in metrics and metrics['geo']:
        for model, vals in metrics['geo'].items():
            if isinstance(vals, dict):
                if vals.get('silhouette') is not None:
                    all_metrics_list.append({
                        'Category': 'Geographic Clustering',
                        'Model': model.upper(),
                        'Metric': 'Silhouette Score',
                        'Value': f"{vals['silhouette']:.4f}",
                        'Target': '> 0.5'
                    })
                if vals.get('davies_bouldin') is not None:
                    all_metrics_list.append({
                        'Category': 'Geographic Clustering',
                        'Model': model.upper(),
                        'Metric': 'Davies-Bouldin',
                        'Value': f"{vals['davies_bouldin']:.4f}",
                        'Target': '< 1.0'
                    })
    
    if 'temp' in metrics and metrics['temp']:
        if metrics['temp'].get('silhouette') is not None:
            all_metrics_list.append({
                'Category': 'Temporal Clustering',
                'Model': 'K-Means',
                'Metric': 'Silhouette Score',
                'Value': f"{metrics['temp']['silhouette']:.4f}",
                'Target': '> 0.5'
            })
        if metrics['temp'].get('davies_bouldin') is not None:
            all_metrics_list.append({
                'Category': 'Temporal Clustering',
                'Model': 'K-Means',
                'Metric': 'Davies-Bouldin',
                'Value': f"{metrics['temp']['davies_bouldin']:.4f}",
                'Target': '< 1.0'
            })
        if metrics['temp'].get('k') is not None:
            all_metrics_list.append({
                'Category': 'Temporal Clustering',
                'Model': 'K-Means',
                'Metric': 'Clusters',
                'Value': str(metrics['temp']['k']),
                'Target': '3-5'
            })
    
    if 'dim' in metrics and metrics['dim']:
        if metrics['dim'].get('explained_variance') is not None:
            all_metrics_list.append({
                'Category': 'Dimensionality Reduction',
                'Model': 'PCA',
                'Metric': 'Explained Variance',
                'Value': f"{metrics['dim']['explained_variance']*100:.1f}%",
                'Target': '> 70%'
            })
    
    if all_metrics_list:
        metrics_df = pd.DataFrame(all_metrics_list)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics as CSV",
            data=csv,
            file_name="patroliq_metrics.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📊 Metrics Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Metrics Tracked", len(all_metrics_list))
        
        with col2:
            categories = metrics_df['Category'].nunique()
            st.metric("Categories", categories)
        
        with col3:
            models = metrics_df['Model'].nunique()
            st.metric("Models Evaluated", models)
    else:
        st.warning("⚠️ No metrics available to display.")
        st.info("Please run the following scripts to generate metrics:")
        st.code("""
python src/models/geo_clustering.py
python src/models/temporal_clustering.py  
python src/models/dimensionality_reduction.py
        """, language="bash")

with tab4:
    st.header("🔗 MLflow Tracking UI")
    
    st.markdown("""
    ### Access MLflow Tracking Server
    
    PatrolIQ integrates with MLflow for comprehensive experiment tracking and model management.
    
    **MLflow UI URL:** `http://127.0.0.1:5000`
    
    #### Features Available in MLflow UI:
    
    1. **📊 Experiment Tracking**
       - View all experiments and runs
       - Compare metrics across runs
       - Track parameters and artifacts
    
    2. **📈 Metrics Visualization**
       - Interactive metric plots
       - Real-time performance monitoring
       - Historical trend analysis
    
    3. **🗂️ Model Registry**
       - Version control for models
       - Model staging and deployment
       - Model lineage tracking
    
    4. **📦 Artifact Storage**
       - Model files and weights
       - Plots and visualizations
       - Configuration files
    """)
    
    st.info("💡 **Tip**: Keep the MLflow server running to access real-time experiment data")
    
    # Connection status
    st.subheader("🔌 Connection Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("mlflow ui --host 127.0.0.1 --port 5000", language="bash")
    
    with col2:
        st.success("✅ MLflow server should be running on port 5000")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    st.markdown("""
    ```bash
    # Start MLflow server
    mlflow ui --host 127.0.0.1 --port 5000
    
    # View experiments
    mlflow experiments list
    
    # Search runs
    mlflow runs search --experiment-id 0
    ```
    """)

# Footer with experiment info
st.markdown("---")
st.subheader("📋 Experiment Tracking Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Tracked Experiments:**
    - Geographic Clustering
    - Temporal Clustering
    - Dimensionality Reduction
    """)

with col2:
    st.markdown("""
    **Logged Artifacts:**
    - Model files (.pkl)
    - Visualizations (.html, .png)
    - Metrics (.json, .csv)
    """)

with col3:
    st.markdown("""
    **Metrics Tracked:**
    - Silhouette Score
    - Davies-Bouldin Index
    - Explained Variance
    """)