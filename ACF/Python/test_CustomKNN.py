# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:51:38 2022

@author: Yannis Schumann
"""
import CustomKNN
import pandas as pd
import numpy as np
import pytest


def test_CustomKNN_Constructor():
    n_neighbors = 15
    weights = "distance"
    oversampling = True
    precomputed = True
    corr_type="spearman"
    knn = CustomKNN.CustomKNN(n_neighbors=n_neighbors,weights=weights, oversampling=oversampling, precomputed=precomputed, corr_type=corr_type)
    assert knn.n_neighbors == n_neighbors
    assert knn.weights == weights
    assert knn.oversampling == oversampling
    assert knn.precomputed == precomputed
    assert knn.corr_type == corr_type


def test_custom_oversampling():
    corr_mat = pd.DataFrame([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[11,12,13,14,15]]).transpose()
    y = [1,1,1,2,3,3]
    knn = CustomKNN.CustomKNN()
    oversampled_corr_mat, oversampled_y = knn.__custom_oversampling__(corr_mat, y)
    
    expected_oversampled_corr = pd.DataFrame([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[11,12,13,14,15],[6,7,8,9,10],[6,7,8,9,10],[11,12,13,14,15]]).transpose()
    expected_y = np.array([1,1,1,2,3,3,2,2,3])
    
    assert np.all(oversampled_y==expected_y)
    assert  np.all(expected_oversampled_corr.values==oversampled_corr_mat.values)


def test_rescale_corr_mat():
    knn = CustomKNN.CustomKNN()
    rng = np.random.default_rng()
    corr_mat = pd.DataFrame(rng.uniform(low=-1, high=1, size=(50,50)))
    corr_mat_rescaled = knn.__rescale_corr_mat__(corr_mat)
    assert np.all(np.greater_equal(corr_mat_rescaled, 0.0))
    assert np.all(np.less_equal(corr_mat_rescaled, 1.0))
    assert np.argmax(corr_mat.values.ravel())==np.argmin(corr_mat_rescaled.values.ravel())


def test_nearest_neighbor_vote():
    knn = CustomKNN.CustomKNN(n_neighbors=2, weights="uniform")
    corr_mat = pd.DataFrame([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[11,12,13,14,15],[6,7,8,9,10],[6,7,8,9,10],[11,12,13,14,15]]).transpose()
    y = np.array([1,1,1,2,3,3,2,2,3])
    assert np.all(knn.__nearest_neighbor_vote__(corr_mat, y)==np.ones(5))

def test_CustomKNN_total():
    
    knn = CustomKNN.CustomKNN(n_neighbors=1, weights="uniform")
    rng = np.random.default_rng()
    X_train = rng.normal(size=(50,200))
    y_train = rng.choice([1,2,3], p=[0.5,0.25,0.25], size=50)
    knn.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    X_test = X_train[test_indices,:]
    y_predicted = knn.predict(X_test)
    assert np.all(y_train[test_indices]==y_predicted)
    
    knn = CustomKNN.CustomKNN(n_neighbors=1, weights="uniform", precomputed=True, oversampling=True)
    rng = np.random.default_rng()
    X_train = pd.DataFrame(rng.normal(size=(50,200))).transpose().corr()
    y_train = rng.choice([1,2,3], p=[0.5,0.25,0.25], size=50)
    knn.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    X_test = X_train.loc[test_indices]
    y_predicted = knn.predict(X_test)
    assert np.all(y_train[test_indices]==y_predicted)

def test_CustomKNN_fit():
    knn = CustomKNN.CustomKNN(n_neighbors=10, precomputed=True)
    rng = np.random.default_rng()
    X = rng.normal(size=(20,20))
    y = rng.choice([1,2,3], size=20)
    
    knn.fit(X,y)
    assert np.all(knn.X_.values==X)
    assert np.all(knn.y_==y)