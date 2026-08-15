"""
PatrolIQ - Exploratory Data Analysis (EDA) Insights
Display comprehensive EDA results and statistical summaries
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import base64

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH_01 = BASE_DIR / "data" / "processed" / "sample_250000_rows_01.csv"
DATA_PATH_02 = BASE_DIR / "data" / "processed" / "sample_250000_rows_02.csv"
FIG_DIR = BASE_DIR / "reports" / "figures"
SUM_DIR = BASE_DIR / "reports" / "summaries"

st.set_page_config(page_title="EDA Insights", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA) Insights")
st.markdown("Comprehensive statistical analysis and visual exploration of Chicago crime data")


# Helper functions
@st.cache_data
def load_summary_csv(filename):
    path = SUM_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_data():
    try:
        df_01 = pd.read_csv(DATA_PATH_01, low_memory=False)
        df_02 = pd.read_csv(DATA_PATH_02, low_memory=False)
        df = pd.concat([df_01, df_02], ignore_index=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            st.error("❌ 'Date' column not found")

        return df

    except Exception as e:
        st.error("🔥 Error while loading data")
        st.exception(e)
        return None


def display_image(image_path):
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"⚠️ Image not found: {image_path.name}")

def display_html(html_path):
    if html_path.exists():
        try:
            # Try UTF-8 first
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                with open(html_path, 'r', encoding='latin-1') as f:
                    html_content = f.read()
            except Exception as e:
                st.error(f"❌ Error reading HTML file: {e}")
                return
        
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.warning(f"⚠️ HTML file not found: {html_path.name}")

# Load data
df = load_data()

if df is not None:
    st.success(f"✅ Loaded {len(df):,} crime records for analysis")
else:
    st.error("❌ Data not found. Please run eda_pipeline.py first.")
    st.stop()

# Tabs for different EDA sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔢 Crime Distribution",
    "🗺️ Geographic Patterns",
    "⏰ Temporal Trends",
    "👮 Arrest Analysis",
    "📈 Statistical Summary",
    "🏘️ Community Analysis"
])

# ========================================
# TAB 1: CRIME DISTRIBUTION
# ========================================
with tab1:
    st.header("Crime Type Distribution Analysis")
    
    # Load crime counts
    crime_counts = load_summary_csv("crime_counts.csv")
    
    if crime_counts is not None:
        crime_counts.columns = ['Primary Type', 'Count']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 All Crime Types Distribution")
            
            # Interactive plotly chart
            fig = px.bar(
                crime_counts.head(20),
                x='Primary Type',
                y='Count',
                title="Top 20 Crime Types",
                color='Count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🥇 Top 10 Crimes")
            top_10 = crime_counts.head(10)
            
            for i, row in top_10.iterrows():
                st.metric(
                    f"{i+1}. {row['Primary Type'][:20]}",
                    f"{row['Count']:,}",
                    delta=f"{(row['Count']/crime_counts['Count'].sum()*100):.1f}%"
                )
        
        # Show static matplotlib image
        st.subheader("📈 Static Chart View")
        display_image(FIG_DIR / "top20_crime_types.png")
        
        # Display full HTML chart
        st.subheader("🔍 Interactive Full Distribution")
        display_html(FIG_DIR / "crime_type_distribution.html")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 Crime Type Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Crime Types", len(crime_counts))
        
        with col2:
            most_common = crime_counts.iloc[0]
            st.metric("Most Common", most_common['Primary Type'][:15])
        
        with col3:
            top_3_pct = (crime_counts.head(3)['Count'].sum() / crime_counts['Count'].sum() * 100)
            st.metric("Top 3 Crimes %", f"{top_3_pct:.1f}%")
        
        with col4:
            least_common = crime_counts.iloc[-1]
            st.metric("Least Common", least_common['Primary Type'][:15])
        
        # Download option
        st.download_button(
            label="📥 Download Crime Counts CSV",
            data=crime_counts.to_csv(index=False),
            file_name="crime_counts.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Crime counts data not found")

# ========================================
# TAB 2: GEOGRAPHIC PATTERNS
# ========================================
with tab2:
    st.header("Geographic Crime Patterns")
    
    st.subheader("🌍 Crime Location Scatter Plot (50K Sample)")
    display_image(FIG_DIR / "geo_scatter_50k.png")
    
    st.markdown("---")
    st.subheader("🔥 Interactive Crime Heatmap")
    display_html(FIG_DIR / "crime_heatmap.html")
    
    st.info("🗺️ The heatmap shows crime density across Chicago using 50,000 sampled points")
    
    # Geographic statistics
    if df is not None:
        st.markdown("---")
        st.subheader("📍 Geographic Statistics")
        
        latlon_data = df.dropna(subset=['Latitude', 'Longitude'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Records with Coordinates", f"{len(latlon_data):,}")
        
        with col2:
            lat_range = latlon_data['Latitude'].max() - latlon_data['Latitude'].min()
            st.metric("Latitude Range", f"{lat_range:.4f}°")
        
        with col3:
            lon_range = latlon_data['Longitude'].max() - latlon_data['Longitude'].min()
            st.metric("Longitude Range", f"{lon_range:.4f}°")
        
        with col4:
            center_lat = latlon_data['Latitude'].mean()
            st.metric("Center Latitude", f"{center_lat:.4f}°")
        
        # Coordinate distribution
        st.subheader("📊 Coordinate Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                latlon_data.sample(min(50000, len(latlon_data))),
                x='Latitude',
                nbins=50,
                title="Latitude Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                latlon_data.sample(min(50000, len(latlon_data))),
                x='Longitude',
                nbins=50,
                title="Longitude Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

# ========================================
# TAB 3: TEMPORAL TRENDS
# ========================================
with tab3:
    st.header("Temporal Crime Patterns")
    
    # Hourly patterns
    st.subheader("⏰ Hourly Crime Distribution")
    
    hourly_counts = load_summary_csv("hourly_counts.csv")
    if hourly_counts is not None:
        hourly_counts.columns = ['Hour', 'Count']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(
                hourly_counts,
                x='Hour',
                y='Count',
                markers=True,
                title="Crime Frequency by Hour of Day"
            )
            fig.update_traces(line_color='#1f77b4', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔝 Peak Hours")
            peak_hours = hourly_counts.nlargest(5, 'Count')
            for i, row in peak_hours.iterrows():
                st.metric(
                    f"Hour {row['Hour']}:00",
                    f"{row['Count']:,} crimes"
                )
    
    display_image(FIG_DIR / "crimes_by_hour.png")
    
    st.markdown("---")
    
    # Hour x Weekday heatmap
    st.subheader("🗓️ Weekday × Hour Heatmap")
    display_image(FIG_DIR / "weekday_hour_heatmap.png")
    
    hour_week = load_summary_csv("hour_week_counts.csv")
    if hour_week is not None:
        st.info("📊 This heatmap shows crime patterns across different hours and days of the week")
    
    st.markdown("---")
    
    # Monthly trends
    st.subheader("📅 Monthly Crime Trends by Year")
    display_html(FIG_DIR / "monthly_trend_by_year.html")
    
    monthly_trend = load_summary_csv("monthly_trend.csv")
    if monthly_trend is not None:
        monthly_trend = monthly_trend.loc[:, ~monthly_trend.columns.str.contains('^Unnamed')]
        st.dataframe(monthly_trend.head(20), use_container_width=True)
    
    st.markdown("---")
    
    # Seasonal patterns
    st.subheader("🌍 Seasonal Crime Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_image(FIG_DIR / "crimes_by_season.png")
    
    with col2:
        season_counts = load_summary_csv("season_counts.csv")
        if season_counts is not None:
            season_counts.columns = ['Season', 'Count']
            st.subheader("📊 Season Statistics")
            
            for i, row in season_counts.iterrows():
                emoji = {'Winter': '❄️', 'Spring': '🌸', 'Summer': '☀️', 'Fall': '🍂'}
                st.metric(
                    f"{emoji.get(row['Season'], '🌍')} {row['Season']}",
                    f"{row['Count']:,}"
                )

# ========================================
# TAB 4: ARREST ANALYSIS
# ========================================
with tab4:
    st.header("Arrest Rate & Domestic Incident Analysis")
    
    arrest_dom = load_summary_csv("arrest_domestic_by_type.csv")
    
    if arrest_dom is not None:
        st.subheader("📊 Arrest Rates by Crime Type")
        
        # Top 15 arrest rates chart
        display_image(FIG_DIR / "arrest_rate_by_type_top15.png")
        
        st.markdown("---")
        
        # Interactive analysis
        st.subheader("🔍 Detailed Arrest & Domestic Analysis")
        
        # Sort options
        sort_by = st.selectbox(
            "Sort by",
            ['Total Crimes', 'Arrest Rate', 'Domestic Rate'],
            key='arrest_sort'
        )
        
        sort_col_map = {
            'Total Crimes': 'total',
            'Arrest Rate': 'arrest_rate',
            'Domestic Rate': 'domestic_rate'
        }
        
        sorted_df = arrest_dom.sort_values(sort_col_map[sort_by], ascending=False).head(20)
        
        # Display as interactive chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                sorted_df,
                x='Primary Type',
                y='arrest_rate',
                title="Arrest Rate by Crime Type (Top 20)",
                color='arrest_rate',
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                sorted_df,
                x='Primary Type',
                y='domestic_rate',
                title="Domestic Incident Rate by Crime Type (Top 20)",
                color='domestic_rate',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Full Data Table")
        
        # Format percentages
        display_df = arrest_dom.copy()
        display_df['arrest_rate'] = (display_df['arrest_rate'] * 100).round(2)
        display_df['domestic_rate'] = (display_df['domestic_rate'] * 100).round(2)
        
        st.dataframe(
            display_df.style.background_gradient(subset=['arrest_rate'], cmap='Blues')
                           .background_gradient(subset=['domestic_rate'], cmap='Reds')
                           .format({'arrest_rate': '{:.2f}%', 'domestic_rate': '{:.2f}%'}),
            use_container_width=True,
            height=400
        )
        
        # Key insights
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            highest_arrest = arrest_dom.loc[arrest_dom['arrest_rate'].idxmax()]
            st.metric(
                "Highest Arrest Rate",
                highest_arrest['Primary Type'][:20],
                f"{highest_arrest['arrest_rate']*100:.1f}%"
            )
        
        with col2:
            lowest_arrest = arrest_dom.loc[arrest_dom['arrest_rate'].idxmin()]
            st.metric(
                "Lowest Arrest Rate",
                lowest_arrest['Primary Type'][:20],
                f"{lowest_arrest['arrest_rate']*100:.1f}%"
            )
        
        with col3:
            avg_arrest = arrest_dom['arrest_rate'].mean()
            st.metric(
                "Average Arrest Rate",
                f"{avg_arrest*100:.1f}%"
            )
        
        # Download option
        st.download_button(
            label="📥 Download Arrest Analysis CSV",
            data=arrest_dom.to_csv(index=False),
            file_name="arrest_domestic_analysis.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Arrest analysis data not found")

# ========================================
# TAB 5: STATISTICAL SUMMARY
# ========================================
with tab5:
    st.header("Statistical Summary & Insights")
    
    summary_stats = load_summary_csv("general_summary_stats.csv")
    
    if summary_stats is not None:
        st.subheader("📊 Descriptive Statistics")
        
        st.dataframe(
            summary_stats.style.background_gradient(cmap='YlOrRd', axis=1),
            use_container_width=True,
            height=600
        )
        
        st.markdown("---")
        
        # Key statistics
        st.subheader("🔢 Key Metrics")
        
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            
            with col2:
                unique_crimes = df['Primary Type'].nunique()
                st.metric("Unique Crime Types", unique_crimes)
            
            with col3:
                if 'District' in df.columns:
                    unique_districts = df['District'].nunique()
                    st.metric("Districts", unique_districts)
            
            with col4:
                if 'Community Area' in df.columns:
                    unique_areas = df['Community Area'].nunique()
                    st.metric("Community Areas", unique_areas)
            
            # Date range
            st.markdown("---")
            st.subheader("📅 Data Coverage")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_date = df['Date'].min()
                st.metric("Start Date", min_date.strftime('%Y-%m-%d'))
            
            with col2:
                max_date = df['Date'].max()
                st.metric("End Date", max_date.strftime('%Y-%m-%d'))
            
            with col3:
                date_range = (max_date - min_date).days
                st.metric("Date Range", f"{date_range:,} days")
            
            # Missing data analysis
            st.markdown("---")
            st.subheader("🔍 Data Quality Analysis")
            
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df) * 100).round(2)
            
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if not missing_df.empty:
                fig = px.bar(
                    missing_df.head(15),
                    x='Column',
                    y='Missing %',
                    title="Top 15 Columns with Missing Data",
                    color='Missing %',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing data found!")
        
        # Download option
        st.download_button(
            label="📥 Download Statistical Summary CSV",
            data=summary_stats.to_csv(),
            file_name="statistical_summary.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Statistical summary not found")

# ========================================
# TAB 6: COMMUNITY ANALYSIS
# ========================================
with tab6:
    st.header("Community Area Analysis")
    
    community_areas = load_summary_csv("top_community_areas.csv")
    
    if community_areas is not None:
        st.subheader("🏘️ Crime Distribution by Community Area")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top 20 community areas
            top_20 = community_areas.head(20)
            
            fig = px.bar(
                top_20,
                x='Community Area',
                y='counts',
                title="Top 20 Community Areas by Crime Count",
                color='counts',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🥇 Top 10 Areas")
            top_10 = community_areas.head(10)
            
            for i, row in top_10.iterrows():
                st.metric(
                    f"#{i+1} Area {int(row['Community Area'])}",
                    f"{row['counts']:,} crimes"
                )
        
        st.markdown("---")
        
        # Community area statistics
        st.subheader("📊 Community Area Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Areas", len(community_areas))
        
        with col2:
            top_area = community_areas.iloc[0]
            st.metric("Highest Crime Area", int(top_area['Community Area']))
        
        with col3:
            top_crime_count = community_areas.iloc[0]['counts']
            st.metric("Highest Crime Count", f"{top_crime_count:,}")
        
        with col4:
            avg_crimes = community_areas['counts'].mean()
            st.metric("Average per Area", f"{avg_crimes:,.0f}")
        
        # Full data table
        st.markdown("---")
        st.subheader("📋 All Community Areas Data")
        
        st.dataframe(
            community_areas.style.background_gradient(subset=['counts'], cmap='Reds'),
            use_container_width=True,
            height=400
        )
        
        # Distribution chart
        fig = px.histogram(
            community_areas,
            x='counts',
            nbins=30,
            title="Distribution of Crimes Across Community Areas",
            labels={'counts': 'Crime Count', 'count': 'Number of Areas'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download option
        st.download_button(
            label="📥 Download Community Area Data CSV",
            data=community_areas.to_csv(index=False),
            file_name="community_area_analysis.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Community area data not found")
    
    # Sample data for UI drilldown
    st.markdown("---")
    st.subheader("🔍 Sample Data Preview")
    
    sample_data = load_summary_csv("sample_for_ui.csv")
    if sample_data is not None:
        st.info("📊 Showing 1000 randomly sampled records for detailed exploration")
        st.dataframe(sample_data, use_container_width=True, height=400)
        
        st.download_button(
            label="📥 Download Sample Data CSV",
            data=sample_data.to_csv(index=False),
            file_name="sample_data_1000.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.subheader("📝 EDA Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Crime Distribution:**
    - 33 unique crime types
    - Top 3 crimes account for majority
    - Significant variation in frequency
    """)

with col2:
    st.markdown("""
    **Geographic Patterns:**
    - Crimes concentrated in specific areas
    - Clear hotspot zones identified
    - Coordinate-based clustering visible
    """)

with col3:
    st.markdown("""
    **Temporal Patterns:**
    - Clear hourly peaks
    - Weekday variations evident
    - Seasonal fluctuations present
    """)

st.info("💡 **Next Steps**: Use this EDA analysis to inform clustering and modeling strategies")
