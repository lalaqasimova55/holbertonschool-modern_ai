#!/usr/bin/env python3
"""
Module for standardizing tabular data using Scikit-learn.
"""

from sklearn import preprocessing


def Standardize(X):
    """
    Standardize a dataset so that each feature has
    mean 0 and standard deviation 1.

    Args:
        X (numpy.ndarray): Dataset of shape
            (n_samples, n_features)

    Returns:
        numpy.ndarray: Standardized dataset.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
