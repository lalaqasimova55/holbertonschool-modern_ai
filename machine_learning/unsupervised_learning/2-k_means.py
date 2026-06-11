#!/usr/bin/env python3
"""
Module for K-Means clustering using Scikit-learn.
"""

from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """
    Create and fit a K-Means clustering model.

    Args:
        X (numpy.ndarray): Tabular data of shape
            (n_samples, n_features).
        n_clusters (int): Number of clusters to find.
        random_state (int): Random seed for reproducibility.

    Returns:
        sklearn.cluster.KMeans: Fitted K-Means instance.
    """
    model = cluster.KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )

    model.fit(X)

    return model
