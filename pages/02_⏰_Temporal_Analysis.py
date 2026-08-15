"""
PatrolIQ - Temporal Pattern Analysis
Interactive temporal visualizations and trend analysis
Enhanced with crime-type-specific insights and comprehensive pattern analysis
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
DATA_PATH_01 = BASE_DIR / "data" / "processed" / "sample_250000_rows_01.csv"
DATA_PATH_02 = BASE_DIR / "data" / "processed" / "sample_250000_rows_02.csv"
TEMP_METRICS = BASE_DIR / "reports" / "summaries" / "temporal_clustering_metrics.json"
TEMP_SUMMARY = BASE_DIR / "reports" / "summaries" / "temporal_cluster_summary.csv"

st.set_page_config(page_title="Temporal Analysis", page_icon="⏰", layout="wide")

st.title("⏰ Temporal Pattern Analysis")
st.markdown("Analyze crime patterns across time: hourly, daily, monthly, and seasonal trends")

# Cache data loading
@st.cache_data
def load_data():
    df_01 = pd.read_csv(DATA_PATH_01, low_memory=False)
    df_02 = pd.read_csv(DATA_PATH_02, low_memory=False)
    df = pd.concat([df_01, df_02], ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Create temporal features if not present
    if 'Year' not in df.columns:
        df['Year'] = df['Date'].dt.year
    if 'Month' not in df.columns:
        df['Month'] = df['Date'].dt.month
    if 'Day' not in df.columns:
        df['Day'] = df['Date'].dt.day
    if 'Hour' not in df.columns:
        df['Hour'] = df['Date'].dt.hour
    if 'Weekday' not in df.columns:
        df['Weekday'] = df['Date'].dt.weekday
    if 'Season' not in df.columns:
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
    
    return df

@st.cache_data
def load_metrics():
    metrics = {}
    if TEMP_METRICS.exists():
        with open(TEMP_METRICS) as f:
            metrics['temp'] = json.load(f)
    return metrics

def load_temporal_summary():
    if TEMP_SUMMARY.exists():
        return pd.read_csv(TEMP_SUMMARY)
    return None

# Load data
with st.spinner("Loading crime data..."):
    df = load_data()
    temp_summary = load_temporal_summary()
    metrics = load_metrics()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Year filter
years = sorted(df['Year'].unique())
selected_years = st.sidebar.multiselect("Select Years", years, default=years[-3:] if len(years) >= 3 else years)

# Crime type filter
crime_types = ['All'] + sorted(df['Primary Type'].unique().tolist())
selected_crime = st.sidebar.selectbox("Crime Type", crime_types)

# Apply filters
filtered_df = df[df['Year'].isin(selected_years)] if selected_years else df
if selected_crime != 'All':
    filtered_df = filtered_df[filtered_df['Primary Type'] == selected_crime]

st.sidebar.info(f"📊 Analyzing **{len(filtered_df):,}** records")

# Weekday mapping
weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
               4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

month_names = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
               7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⏰ Temporal Patterns",
    "📅 Hourly Analysis", 
    "📆 Weekly Patterns", 
    "📊 Monthly/Seasonal", 
    "🔥 Heatmaps",
    "⚠️ Danger Times"
])

# ----------------------------- TAB 1: TEMPORAL PATTERNS -----------------------------
with tab1:
    st.header("🎯 Identified Temporal Crime Patterns")
    
    if 'temp' in metrics:
        temp_metrics = metrics['temp']
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pattern Groups", temp_metrics['k'])
        with col2:
            if temp_metrics.get('silhouette') is not None:
                st.metric("Silhouette Score", f"{temp_metrics['silhouette']:.4f}")
        with col3:
            if temp_metrics.get('davies_bouldin') is not None:
                st.metric("Davies-Bouldin Index", f"{temp_metrics['davies_bouldin']:.4f}")
        with col4:
            total_crimes = len(filtered_df)
            st.metric("Total Crimes Analyzed", f"{total_crimes:,}")
        
        st.markdown("---")
        
        # Pattern details
        if temp_summary is not None:
            st.subheader("📋 Pattern Characteristics")
            
            display_summary = temp_summary.copy()
            
            # Rename columns for better display
            col_mapping = {
                'TemporalCluster': 'Cluster',
                'Hour': 'Avg Hour',
                'Weekday': 'Avg Weekday',
                'Month': 'Avg Month',
                'Pattern_Name': 'Pattern Type',
                'Description': 'Description',
                'Count': 'Crime Count'
            }
            
            display_cols = [col for col in col_mapping.keys() if col in display_summary.columns]
            display_summary = display_summary[display_cols]
            display_summary = display_summary.rename(columns=col_mapping)
            
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Visualize patterns - Radar Chart
            st.subheader("🎯 Pattern Profile Comparison")
            
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, row in temp_summary.iterrows():
                if i >= len(colors):
                    break
                pattern_name = row.get('Pattern_Name', f'Pattern {i}')
                
                fig.add_trace(go.Scatterpolar(
                    r=[row['Hour'], row['Weekday'], row['Month']],
                    theta=['Hour of Day (0-23)', 'Day of Week (0-6)', 'Month (1-12)'],
                    fill='toself',
                    name=pattern_name,
                    line_color=colors[i]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 24])
                ),
                showlegend=True,
                title="Temporal Pattern Profiles (Normalized View)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Interpretation:** Each pattern represents a distinct crime behavior based on time, day, and month. Patterns closer to the edges occur at more extreme times.")
        
        # Peak times summary
        st.markdown("---")
        st.subheader("🔥 Peak Crime Times")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'peak_hours' in temp_metrics:
                st.write("**Top 5 Peak Hours:**")
                peak_data = []
                for i, hour in enumerate(temp_metrics['peak_hours'], 1):
                    count = temp_metrics.get('peak_hours_counts', [0]*5)[i-1]
                    peak_data.append({"Rank": i, "Hour": f"{hour}:00", "Crimes": f"{count:,}"})
                st.table(pd.DataFrame(peak_data))
        
        with col2:
            if 'peak_months' in temp_metrics:
                st.write("**Top 3 Peak Months:**")
                peak_data = []
                for i, month in enumerate(temp_metrics['peak_months'], 1):
                    count = temp_metrics.get('peak_months_counts', [0]*3)[i-1]
                    peak_data.append({"Rank": i, "Month": month_names.get(month, str(month)), "Crimes": f"{count:,}"})
                st.table(pd.DataFrame(peak_data))
    else:
        st.warning("⚠️ Temporal clustering metrics not found. Please run temporal_clustering.py")

# ----------------------------- TAB 2: HOURLY ANALYSIS -----------------------------
with tab2:
    st.header("📅 Hourly Crime Distribution")
    
    hourly = filtered_df['Hour'].value_counts().sort_index()
    
    # Line chart
    fig = px.line(
        x=hourly.index,
        y=hourly.values,
        markers=True,
        title="Crime Frequency by Hour of Day"
    )
    fig.update_traces(line_color='#1f77b4', line_width=3, marker=dict(size=8))
    fig.update_layout(
        height=500,
        xaxis=dict(title="Hour of Day", dtick=1, range=[-0.5, 23.5]),
        yaxis=dict(title="Number of Crimes"),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Peak hours metrics
    st.subheader("🏆 Top Crime Hours")
    col1, col2, col3 = st.columns(3)
    peak_hours = hourly.nlargest(3)
    
    with col1:
        st.metric("🥇 Peak Hour #1", f"{peak_hours.index[0]}:00", f"{peak_hours.values[0]:,} crimes")
    with col2:
        st.metric("🥈 Peak Hour #2", f"{peak_hours.index[1]}:00", f"{peak_hours.values[1]:,} crimes")
    with col3:
        st.metric("🥉 Peak Hour #3", f"{peak_hours.index[2]}:00", f"{peak_hours.values[2]:,} crimes")
    
    # Time period analysis
    st.markdown("---")
    st.subheader("🌅 Crime by Time Period")
    
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'Morning (6-11 AM)'
        elif 12 <= hour < 18:
            return 'Afternoon (12-5 PM)'
        elif 18 <= hour < 22:
            return 'Evening (6-9 PM)'
        else:
            return 'Night (10 PM-5 AM)'
    
    filtered_df['TimePeriod'] = filtered_df['Hour'].apply(get_time_period)
    period_counts = filtered_df['TimePeriod'].value_counts()
    
    fig = px.pie(
        values=period_counts.values,
        names=period_counts.index,
        title="Crime Distribution by Time Period",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- TAB 3: WEEKLY PATTERNS -----------------------------
with tab3:
    st.header("📆 Daily and Weekly Patterns")
    
    # Weekday distribution
    weekday_counts = filtered_df['Weekday'].map(weekday_map).value_counts()
    weekday_counts = weekday_counts.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                             'Friday', 'Saturday', 'Sunday'])
    
    fig = px.bar(
        x=weekday_counts.index,
        y=weekday_counts.values,
        title="Crime Frequency by Day of Week",
        color=weekday_counts.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis=dict(title="Day of Week"),
        yaxis=dict(title="Number of Crimes")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekday vs Weekend detailed comparison
    st.markdown("---")
    st.subheader("📊 Weekday vs Weekend Deep Dive")
    
    weekend = filtered_df[filtered_df['Weekday'].isin([5, 6])]
    weekday = filtered_df[~filtered_df['Weekday'].isin([5, 6])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Weekday Patterns (Mon-Fri)**")
        weekday_hourly = weekday.groupby('Hour').size()
        fig = px.line(
            x=weekday_hourly.index, 
            y=weekday_hourly.values, 
            title="Weekday Hourly Pattern",
            markers=True
        )
        fig.update_traces(line_color='steelblue', line_width=2)
        fig.update_layout(height=300, xaxis_title="Hour", yaxis_title="Crimes")
        st.plotly_chart(fig, use_container_width=True)
        
        weekday_avg = len(weekday) / 5
        st.metric("Avg Crimes per Weekday", f"{weekday_avg:,.0f}")
        st.metric("Peak Weekday Hour", f"{weekday['Hour'].mode()[0]}:00")
    
    with col2:
        st.write("**Weekend Patterns (Sat-Sun)**")
        weekend_hourly = weekend.groupby('Hour').size()
        fig = px.line(
            x=weekend_hourly.index, 
            y=weekend_hourly.values, 
            title="Weekend Hourly Pattern",
            markers=True
        )
        fig.update_traces(line_color='coral', line_width=2)
        fig.update_layout(height=300, xaxis_title="Hour", yaxis_title="Crimes")
        st.plotly_chart(fig, use_container_width=True)
        
        weekend_avg = len(weekend) / 2
        st.metric("Avg Crimes per Weekend Day", f"{weekend_avg:,.0f}")
        st.metric("Peak Weekend Hour", f"{weekend['Hour'].mode()[0]}:00")
    
    # Statistical comparison
    diff_pct = ((weekend_avg) - (weekday_avg)) / (weekday_avg) * 100
    if diff_pct > 0:
        st.success(f"📊 Weekend days have **{abs(diff_pct):.1f}% MORE** crimes on average than weekdays")
    else:
        st.info(f"📊 Weekend days have **{abs(diff_pct):.1f}% FEWER** crimes on average than weekdays")

# ----------------------------- TAB 4: MONTHLY/SEASONAL -----------------------------
with tab4:
    st.header("📊 Monthly and Seasonal Trends")
    
    # Monthly trend over time
    monthly = filtered_df.groupby(['Year', 'Month']).size().reset_index(name='Count')
    monthly['YearMonth'] = pd.to_datetime(monthly[['Year', 'Month']].assign(DAY=1))
    
    fig = px.line(
        monthly,
        x='YearMonth',
        y='Count',
        title="Monthly Crime Trend Over Time",
        markers=True
    )
    fig.update_traces(line_color='#2ca02c', line_width=2)
    fig.update_layout(
        height=400,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Number of Crimes")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal distribution
    st.markdown("---")
    st.subheader("🌍 Seasonal Distribution")
    
    season_counts = filtered_df['Season'].value_counts()
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_counts = season_counts.reindex(season_order)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            x=season_counts.index,
            y=season_counts.values,
            title="Crime Frequency by Season",
            color=season_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(title="Season"),
            yaxis=dict(title="Number of Crimes")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Seasonal Breakdown:**")
        for season in season_order:
            count = season_counts[season]
            pct = (count / season_counts.sum() * 100)
            st.metric(f"{season}", f"{count:,}", f"{pct:.1f}%")
    
    # Year-over-year seasonal comparison
    if len(selected_years) > 1:
        st.markdown("---")
        st.subheader("📈 Year-over-Year Seasonal Comparison")
        
        seasonal_yearly = filtered_df.groupby(['Year', 'Season']).size().reset_index(name='Count')
        
        fig = px.line(
            seasonal_yearly,
            x='Season',
            y='Count',
            color='Year',
            title="Seasonal Crime Trends Across Years",
            markers=True,
            category_orders={"Season": season_order}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------- TAB 5: HEATMAPS -----------------------------
with tab5:
    st.header("🔥 Temporal Heatmaps")
    
    # Hour x Weekday heatmap
    st.subheader("Day of Week vs Hour Heatmap")
    
    heatmap_data = filtered_df.groupby(['Weekday', 'Hour']).size().reset_index(name='Count')
    heatmap_pivot = heatmap_data.pivot(index='Weekday', columns='Hour', values='Count').fillna(0)
    
    # Map weekday numbers to names
    heatmap_pivot.index = heatmap_pivot.index.map(weekday_map)
    heatmap_pivot = heatmap_pivot.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                           'Friday', 'Saturday', 'Sunday'])
    
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Crime Count"),
        title="Crime Heatmap: Day of Week vs Hour",
        aspect="auto",
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("🔥 **Darker colors indicate higher crime frequency**. Use this to identify peak danger times by day and hour.")
    
    # Month x Hour heatmap
    st.markdown("---")
    st.subheader("Month vs Hour Heatmap")
    
    month_hour_data = filtered_df.groupby(['Month', 'Hour']).size().reset_index(name='Count')
    month_hour_pivot = month_hour_data.pivot(index='Month', columns='Hour', values='Count').fillna(0)
    month_hour_pivot.index = month_hour_pivot.index.map(month_names)
    
    fig = px.imshow(
        month_hour_pivot,
        labels=dict(x="Hour of Day", y="Month", color="Crime Count"),
        title="Crime Heatmap: Month vs Hour",
        aspect="auto",
        color_continuous_scale='YlOrRd'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- TAB 6: DANGER TIMES -----------------------------
with tab6:
    st.header("⚠️ Peak Danger Time Analysis")
    
    st.markdown("""
    This section identifies the most dangerous times based on **violent crime patterns**.
    Violent crimes include: Homicide, Assault, Battery, Robbery, and Criminal Sexual Assault.
    """)
    
    # Define violent crimes
    violent_crimes = ['HOMICIDE', 'ASSAULT', 'BATTERY', 'ROBBERY', 
                     'CRIMINAL SEXUAL ASSAULT', 'CRIM SEXUAL ASSAULT', 
                     'WEAPONS VIOLATION']
    
    violent_df = filtered_df[filtered_df['Primary Type'].isin(violent_crimes)]
    
    if len(violent_df) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Violent Crimes", f"{len(violent_df):,}")
        with col2:
            violent_pct = (len(violent_df) / len(filtered_df) * 100)
            st.metric("% of All Crimes", f"{violent_pct:.1f}%")
        with col3:
            peak_violent_hour = violent_df['Hour'].mode()[0]
            st.metric("Peak Violent Hour", f"{peak_violent_hour}:00")
        
        st.markdown("---")
        
        # Violent crime hourly distribution
        st.subheader("🔴 Violent Crime by Hour")
        
        violent_hourly = violent_df['Hour'].value_counts().sort_index()
        danger_hours = violent_hourly.nlargest(5)
        
        danger_start = danger_hours.index.min()
        danger_end = danger_hours.index.max()
        
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=violent_hourly.index,
            y=violent_hourly.values,
            marker_color='darkred',
            name='Violent Crimes',
            opacity=0.7
        ))
        
        # Add danger zone rectangle
        fig.add_vrect(
            x0=danger_start - 0.5,
            x1=danger_end + 0.5,
            fillcolor="red",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text=f"Peak Danger Zone: {danger_start}:00 - {danger_end}:00",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="Violent Crime Distribution with Peak Danger Zone",
            xaxis=dict(title="Hour of Day", dtick=1),
            yaxis=dict(title="Violent Crime Count"),
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.error(f"🚨 **PEAK DANGER WINDOW: {danger_start}:00 - {danger_end}:00**  \n*{violent_hourly[danger_hours.index].sum():,} violent crimes occur during these hours*")
        
        # Violent crimes by day
        st.markdown("---")
        st.subheader("📅 Violent Crime by Day of Week")
        
        violent_weekday = violent_df['Weekday'].map(weekday_map).value_counts()
        violent_weekday = violent_weekday.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                                   'Friday', 'Saturday', 'Sunday'])
        
        fig = px.bar(
            x=violent_weekday.index,
            y=violent_weekday.values,
            title="Violent Crimes by Day",
            color=violent_weekday.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Most dangerous day
        most_dangerous_day = violent_weekday.idxmax()
        st.warning(f"⚠️ **Most Dangerous Day:** {most_dangerous_day} ({violent_weekday[most_dangerous_day]:,} violent crimes)")
        
    else:
        st.info("No violent crimes found in the selected filters.")

# ----------------------------- SUMMARY STATISTICS -----------------------------
st.markdown("---")
st.header("📊 Temporal Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    busiest_hour = filtered_df['Hour'].mode()[0]
    st.metric("⏰ Peak Crime Hour", f"{busiest_hour}:00")

with col2:
    busiest_day = filtered_df['Weekday'].mode()[0]
    st.metric("📅 Peak Crime Day", weekday_map[busiest_day])

with col3:
    busiest_month = filtered_df['Month'].mode()[0]
    st.metric("📆 Peak Crime Month", month_names[busiest_month])

with col4:
    busiest_season = filtered_df['Season'].mode()[0]
    st.metric("🌍 Peak Crime Season", busiest_season)

# Export option
st.markdown("---")
if st.button("📥 Export Temporal Analysis Summary"):
    summary_data = {
        "Analysis Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Total Records": len(filtered_df),
        "Date Range": f"{filtered_df['Date'].min()} to {filtered_df['Date'].max()}",
        "Peak Hour": f"{busiest_hour}:00",
        "Peak Day": weekday_map[busiest_day],
        "Peak Month": month_names[busiest_month],
        "Peak Season": busiest_season
    }
    
    st.json(summary_data)
    st.success("✅ Summary generated! Copy the JSON above or take a screenshot.")