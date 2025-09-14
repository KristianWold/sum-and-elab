import unicodedata
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm


def cosine_similarity(embed_a, embed_b, normalize=True):
    """
    Compute the cosine similarity between two vectors.
    """
    if normalize:
        embed_a = embed_a/np.linalg.norm(embed_a, axis=1, keepdims=True)
        embed_b = embed_b/np.linalg.norm(embed_b, axis=1, keepdims=True)
    dot_product = embed_a@embed_b.T


    return dot_product


def cluster(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    inertia = kmeans.inertia_
    labels = kmeans.labels_
    clusters = kmeans.cluster_centers_

    return inertia, labels, clusters


class EmbeddingClustering:
    def __init__(self, tokenizer, n_clusters=10):
        
        self.tokenizer = tokenizer
        self.n_clusters = n_clusters

    def fit(self, word_embed):
        self.word_embed = word_embed.detach().cpu().numpy()

        inertia, labels, clusters = cluster(self.word_embed, self.n_clusters)
        
        self.inertia = inertia
        self.labels = labels
        self.clusters = np.array(clusters)

        cos_sim = cosine_similarity(self.clusters, self.word_embed)
        idx =  np.argsort(cos_sim, axis=-1)[:,::-1]

        return idx
