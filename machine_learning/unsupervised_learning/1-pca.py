#!/usr/bin/env python3
"""
Module for applying Principal Component Analysis (PCA)
to tabular data using Scikit-learn.
"""

from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """
    Perform PCA on a dataset.

    Args:
        X (numpy.ndarray): Input data of shape
            (n_samples, n_features).
        n_components (int | float | None):
            int: Number of principal components to keep.
            float: Fraction of variance to preserve.
            None: Keep all components.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple:
            numpy.ndarray: Transformed data in PCA space.
            sklearn.decomposition.PCA: Fitted PCA instance.
    """
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )

    X_pca = pca.fit_transform(X)

    return X_pca, pca
