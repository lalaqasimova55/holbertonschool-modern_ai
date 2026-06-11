#!/usr/bin/env python3

from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes the dataset.

    Args:
        X (numpy.ndarray): Shape (n_samples, n_features)

    Returns:
        numpy.ndarray: Standardized data with the same shape as X
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
