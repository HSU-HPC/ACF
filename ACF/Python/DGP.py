# -*- coding: utf-8 -*-
"""
Includes functions connected with the DGP developed for the study

@author: Yannis Schumann
"""
import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import itertools


def introduce_MVs(X, mtot):
    """
    Introduces a fixed percentage of missing values at random into each observation.
    
    Parameters
    ----------
    X : np.ndarray
        The n_instances x n_features sized matrix.
    mtot : float
        Percentage of missing values per sample.

    Returns
    -------
    X_new : np.ndarray
        The dataset with the randomly introduced missing values.

    """
    rng = np.random.default_rng()
    # allocate memory for a new copy of X
    X_new = X.copy()
    # number of features
    ids = np.arange(X.shape[1])
    # number of total missing values per observation
    num_missing = int(mtot*X.shape[1])
    
    # iterate over all instances
    for t in range(X.shape[0]):
        # randomly select the features to drop out for this sample
        ncfs = rng.choice(ids, size=num_missing, replace=False)
        # set to NaN
        X_new[np.ix_([t],ncfs)] = np.NaN
    return X_new


def createCorrelatedClassCenters(base_corr):
    """
    Generates points in 10000-dimensional space that have approximately
    the prescribed (pearson) correlation matrix.

    Parameters
    ----------
    base_corr : np.ndarray
        The specified correlation matrix.

    Returns
    -------
    correlated_centers : np.ndarray
        Matrix with shape n_classes x 10000. The rows will have the prescribed correlations.

    """
    
    # Cholesky decomposition
    L = np.linalg.cholesky(base_corr)
    # draw samples from standard distribution (as many as we have classes)
    rng = np.random.default_rng()
    uncorrelated_centers = rng.standard_normal((base_corr.shape[0], 10000))
    # Generate normally distributed centers with (approximately) the 
    correlated_centers = np.dot(L, uncorrelated_centers)
    
    return correlated_centers


def createDataWithBlockCorrelation(classdistribution, base_corr, mvs, cluster_std=2.0, return_corr=False, method="pearson", make_covariable=False, std_covariables = 0.05, centers_covariables = [0.15,0.2,0.25]):
    """
    Creates data with correlation matrix of approximately the prescribed structure.

    Parameters
    ----------
    classdistribution : array-like
        Array with number of observations per class.
    base_corr : np.ndarray
        Correlation matrix of the class centers.
    mvs : float
        Number of missing values.
    cluster_std : float, optional
        Standard deviation per feature. The default is 1.0.
    return_corr : bool, optional
        Whether to compute and return the corelation matrix of the data. The default is False.
    method : str or callable, optional
        Type of correlation. Is passed to pd.DataFrame.corr(). The default is 'pearson'.
    make_covariable : bool, optional
        Whether to create an additional covariable. The default is False.
    std_covariables : float, optional
        Standard deviation of the class-wise distributions for the covariable. This parameter only matters
        if make_covariable is True. The default is 0.05.
    centers_covariables : list-like, optional
        Centers of the class-wise distributions for the covariable. This parameter only matters
        if make_covariable is True. The default is [0.15,0.2,0.25] and works for 3 classes.
    

    Returns
    -------
    X : np.ndarray
        n_instances x 10000 matrix. Corresponds to the dataset
    y : np.ndarray
        corresponding class labels
    corr : np.ndarray (optional)
        Correlation matrix of shape n_instances x n_instances.
    X_c : np.ndarray (optional)
        Matrix of shape (n_instances, 1) with additional covariable.

    """
    # generate centers with the prescribed correlation    
    correlated_centers =  createCorrelatedClassCenters(base_corr)
    # generate points around the correlated centers
    X, y = make_blobs(n_samples=classdistribution, centers=correlated_centers, n_features=10000, cluster_std=cluster_std)
    # introduce the missing values
    X = introduce_MVs(X, mvs)
    
    # compute correlation matrix and make a covariable, if requested
    if return_corr and not make_covariable:
        corr = pd.DataFrame(X).transpose().corr(method=method).values
        return X,y,corr
    if return_corr and make_covariable:
        corr = pd.DataFrame(X).transpose().corr(method=method).values
        X_c = generateCovariable(y, std_covariables, centers_covariables)
        return X,y,corr, X_c
    if not return_corr and make_covariable:
        X_c = generateCovariable(y, std_covariables, centers_covariables)
        return X,y,X_c
    return X,y


def get_sigma_allblocks(corr, y):
    """
    Computes and returns the block-wise standard deviations of the correlation matrix

    Parameters
    ----------
    corr : np.ndarray
        correlation matrix.
    y : np.ndarray
        correspondingly ordered class labels.

    Returns
    -------
    float
        average block-wise standard deviation.

    """
    # copy matrix, because we need to filter out the (biased) self-correlations
    corr_new = corr.copy()
    # filtering
    np.fill_diagonal(corr_new, np.NaN)
    # unique classes
    classes = np.unique(y)
    # array for block-wise sigmas
    sigma_res = []
    #iterate over all classes
    for c1, c2 in itertools.product(classes,classes):
        # c1 indices
        c1_indices = np.where(y==c1)[0]
        # c2 indices
        c2_indices = np.where(y==c2)[0]
        sigma_res.append(np.nanstd(corr_new[np.ix_(c1_indices, c2_indices)]))
    return np.mean(sigma_res)


def diff_mu_12_mu_11(corr, y):
    """
    Compute difference between mean correlations on block 11 (class 1 with class 1) and block 12 (class 1 with class 2)

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix.
    y : np.ndarray
        correspondingly ordered class labels.

    Returns
    -------
    float
        Absolute difference between the mean correlations between the blocks.

    """
    # copy and filter the (biased) self-correlations
    corr_new = corr.copy()
    np.fill_diagonal(corr_new, np.NaN)
    # find indices for 1st class (0)
    idx_1 = np.where(y==0)[0]
    # find indices for 2nd class (1)
    idx_2 = np.where(y==1)[0]
    
    # compute absolute difference
    return np.abs(np.nanmean(corr_new[np.ix_(idx_1, idx_2)])- np.nanmean(corr_new[np.ix_(idx_1, idx_1)]))


def diff_mu_13_mu_23(corr, y):
    """
    Compute difference between mean correlations on block 13 (class 1 with class 3) and block 12 (class 2 with class 3)

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix.
    y : np.ndarray
        correspondingly ordered class labels.

    Returns
    -------
    float
        Absolute difference between the mean correlations between the blocks.

    """
    # copy and filter the (biased) self-correlations
    corr_new = corr.copy()
    np.fill_diagonal(corr_new, np.NaN)
    # find indices for 1st class (0)
    idx_1 = np.where(y==0)[0]
    # find indices for 2nd class (1)
    idx_2 = np.where(y==1)[0]
    # find indices for 3rd class (2)
    idx_3 = np.where(y==2)[0]
    
    # compute absolute difference
    return np.abs(np.nanmean(corr_new[np.ix_(idx_1, idx_3)])- np.nanmean(corr_new[np.ix_(idx_2, idx_3)]))



def generateCovariable(y, std_covariables = 0.05, centers_covariables = [0.15,0.2,0.25]):
    """
    Generates an associated covariable (1D). Centers and standard deviation may be specified
    and refer to gaussian distributions.

    Parameters
    ----------
    y : np.ndarray
        Aray of length n_instances. Specifies the respective class of the samples
    std_covariables : float, optional
        Standard deviation of the associated distributions. The default is 0.05.
    centers_covariables : list-like, optional
        List of length n_instances. Specifies the centers of the distribution per class. The default is [0.15,0.2,0.25].

    Returns
    -------
    X_c : np.ndarray
        Array of shape (n_instances, 1). Covariable generated for the respective y (also same order).

    """
    centers_covariables = np.array(centers_covariables).reshape(-1, 1)
    uniques, counts = np.unique(y, return_counts=True)
    X_temp, y_temp = make_blobs(counts, n_features=1, centers=centers_covariables, cluster_std=std_covariables)
    unique_temp = np.unique(y_temp)
    
    X_c = X_temp.copy()
    
    for unique_class, unique_temp_class in zip(uniques, unique_temp):
        indices = np.nonzero(y==unique_class)[0]
        indices_temp = np.nonzero(y_temp==unique_temp_class)[0]
        assert len(indices)==len(indices_temp)
        X_c[indices,:] = X_temp[indices_temp,:]
    
    return X_c
