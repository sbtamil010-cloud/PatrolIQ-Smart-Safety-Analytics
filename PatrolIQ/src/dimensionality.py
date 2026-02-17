# src/dimensionality.py

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def apply_scaling(features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def run_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return components, pca, explained_variance


def run_tsne(data, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(data)
    return tsne_result
