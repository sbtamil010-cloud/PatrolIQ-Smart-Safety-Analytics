# src/clustering.py

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def run_kmeans(data, n_clusters=8):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, model, score


def run_dbscan(data, eps=0.03, min_samples=20):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)

    # DBSCAN may fail silhouette if noise dominates
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels)
    else:
        score = -1

    return labels, model, score


def run_hierarchical(data, n_clusters=6):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(data)
    score = silhouette_score(data, labels)
    return labels, model, score
