# src/analysis/eda_pipeline.py
"""
Exploratory Data Analysis pipeline for PatrolIQ
Saves plots and summary CSVs to reports/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings("ignore")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH_01 = BASE_DIR / "data" / "processed" / "sample_250000_rows_01.csv"
DATA_PATH_02 = BASE_DIR / "data" / "processed" / "sample_250000_rows_02.csv"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = REPORT_DIR / "figures"
SUM_DIR = REPORT_DIR / "summaries"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
print("Loading data:", DATA_PATH_01)
df_01 = pd.read_csv(DATA_PATH_01, low_memory=False)
print("Loading data:", DATA_PATH_01)
df_02 = pd.read_csv(DATA_PATH_02, low_memory=False)

df = pd.concat([df_01, df_02], ignore_index=True)
print("Total_rows:", len(df))
# Ensure datetime and temporal features exist
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])          # drop rows with bad dates
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday   # 0=Mon
df['Hour'] = df['Date'].dt.hour
df['Season'] = df['Month'].map({12:'Winter',1:'Winter',2:'Winter',
                                3:'Spring',4:'Spring',5:'Spring',
                                6:'Summer',7:'Summer',8:'Summer',
                                9:'Fall',10:'Fall',11:'Fall'})

# 1) Crime distribution across 33 crime types
crime_counts = df['Primary Type'].value_counts().sort_values(ascending=False)
crime_counts.to_csv(SUM_DIR / "crime_counts.csv")

# Plot: top 20 crime types (matplotlib)
plt.figure(figsize=(10,6))
crime_counts.head(20).plot(kind='bar')
plt.title("Top 20 Crime Types")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "top20_crime_types.png", dpi=150)
plt.close()

# Plot: full distribution interactive (plotly)
crime_counts = df["Primary Type"].value_counts()

crime_df = crime_counts.reset_index()
crime_df.columns = ["Primary Type", "Count"]

fig = px.bar(
    crime_df,
    x="Primary Type",
    y="Count",
    title="Crime Type Distribution (all)"
)
fig.update_layout(
    xaxis={'categoryorder': 'total descending'},
    height=600
)

fig.write_html(FIG_DIR / "crime_type_distribution.html")
print("✅ Crime type distribution plot saved!")

fig.update_layout(xaxis={'categoryorder':'total descending'}, height=600)
fig.write_html(FIG_DIR / "crime_type_distribution.html")

# 2) Geographic patterns using lat/lon
# Basic scatter
latlon = df.dropna(subset=['Latitude','Longitude'])
plt.figure(figsize=(8,8))
plt.scatter(latlon['Longitude'].sample(50000, random_state=42),
            latlon['Latitude'].sample(50000, random_state=42),
            s=1, alpha=0.3)
plt.title("Crime locations (sample 50k points)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(FIG_DIR / "geo_scatter_50k.png", dpi=150)
plt.close()

# Heatmap with Folium (saves as HTML)
# Use a sample to reduce size
heat_data = list(zip(latlon['Latitude'].sample(50000, random_state=42),
                     latlon['Longitude'].sample(50000, random_state=42)))
m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)  # Chicago center
HeatMap(heat_data, radius=8, blur=12, min_opacity=0.4).add_to(m)
m.save(str(FIG_DIR / "crime_heatmap.html"))

# 3) Temporal trends (hourly, daily, monthly, seasonal)
# Hourly heatmap (hour x weekday)
hour_week = df.groupby(['Weekday','Hour']).size().unstack(fill_value=0)
hour_week.to_csv(SUM_DIR / "hour_week_counts.csv")

plt.figure(figsize=(12,6))
plt.imshow(hour_week, aspect='auto', origin='lower')
plt.colorbar(label='Count')
plt.title('Heatmap: Weekday (y) x Hour (x)')
plt.xlabel('Hour'); plt.ylabel('Weekday (0=Mon)')
plt.xticks(np.arange(0,24,1))
plt.tight_layout()
plt.savefig(FIG_DIR / "weekday_hour_heatmap.png", dpi=150)
plt.close()

# Hourly distribution line
hourly = df['Hour'].value_counts().sort_index()
hourly.to_csv(SUM_DIR / "hourly_counts.csv")
plt.figure(figsize=(10,4))
hourly.plot(kind='line', marker='o')
plt.title("Crimes by Hour of Day")
plt.xlabel("Hour"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "crimes_by_hour.png", dpi=150)
plt.close()

# Monthly trend
monthly = df.groupby(['Year','Month']).size().rename('count').reset_index()
monthly.to_csv(SUM_DIR / "monthly_trend.csv")
fig = px.line(monthly, x='Month', y='count', color='Year', title='Monthly crime counts by Year')
fig.write_html(FIG_DIR / "monthly_trend_by_year.html")

# Seasonal summary
season_counts = df['Season'].value_counts()
season_counts.to_csv(SUM_DIR / "season_counts.csv")
plt.figure(figsize=(6,4))
season_counts.plot(kind='bar')
plt.title("Crimes by Season")
plt.tight_layout()
plt.savefig(FIG_DIR / "crimes_by_season.png", dpi=150)
plt.close()

# 4) Arrest rates and domestic incident correlations
arrest_dom = df.groupby(['Primary Type']).agg(
    total=('ID','count') if 'ID' in df.columns else ('Date','count'),
    arrests=('Arrest', 'sum'),
    domestic=('Domestic','sum')
).reset_index()
arrest_dom['arrest_rate'] = arrest_dom['arrests'] / arrest_dom['total']
arrest_dom['domestic_rate'] = arrest_dom['domestic'] / arrest_dom['total']
arrest_dom = arrest_dom.sort_values('total', ascending=False)
arrest_dom.to_csv(SUM_DIR / "arrest_domestic_by_type.csv", index=False)

# Plot arrest rate top 15 types
plt.figure(figsize=(10,6))
arrest_dom.head(15).set_index('Primary Type')['arrest_rate'].plot(kind='bar')
plt.title("Arrest Rate by Crime Type (top 15)")
plt.ylabel("Arrest Rate")
plt.tight_layout()
plt.savefig(FIG_DIR / "arrest_rate_by_type_top15.png", dpi=150)
plt.close()

# 5) Statistical summaries and insights
# general stats
summary_stats = df.describe(include='all').transpose()
summary_stats.to_csv(SUM_DIR / "general_summary_stats.csv")

# Top N neighborhoods / blocks
if 'Community Area' in df.columns:
    block_top = df['Community Area'].value_counts().rename_axis('Community Area').reset_index(name='counts')
    block_top.to_csv(SUM_DIR / "top_community_areas.csv", index=False)

# Save a small sample for UI drilldown
df.sample(1000, random_state=42).to_csv(SUM_DIR / "sample_for_ui.csv", index=False)

print("EDA completed. Figures & summaries saved to:", FIG_DIR, SUM_DIR)
print(crime_df.head())
print(crime_df.columns)
