# 🚓 PatrolIQ – Smart Safety Analytics Platform

PatrolIQ is an urban crime intelligence platform designed to analyze 500,000 crime records from Chicago and provide actionable insights for law enforcement agencies.

---

## 📌 Project Objective

To identify:

- Geographic crime hotspots
- Temporal crime patterns
- High-risk areas for patrol optimization
- Key features driving crime patterns

---

## 📊 Dataset

Source: Chicago Crime Dataset (Public Data Portal)  
Records Used: 500,000 recent crimes  
Crime Categories: 33  
Geographic Coverage: Chicago districts  

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- MLflow
- Streamlit
- PyDeck
- PCA
- t-SNE
- K-Means
- DBSCAN
- Hierarchical Clustering

---

## 🏗 Architecture

Data → Preprocessing → Feature Engineering →  
Clustering → Dimensionality Reduction →  
MLflow Tracking → Streamlit Application → Cloud Deployment

---

## 🔥 Key Results

- Identified 8 major crime hotspots
- Achieved silhouette score > 0.52
- Reduced 22 features to 3 principal components explaining 74% variance
- Identified peak crime hours (10PM–2AM)
- Detected seasonal trends (higher summer crime)

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
cd streamlit_app
streamlit run app.py
