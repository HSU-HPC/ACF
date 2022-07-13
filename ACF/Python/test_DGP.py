# -*- coding: utf-8 -*-
"""
Tests for DGP

@author: Yannis Schumann
"""
import numpy as np
import pandas as pd
import DGP

def test_generateCovariable():
    y = [1,2,3,1,2,3,1,2,3]
    X_c = DGP.generateCovariable(y, std_covariables=0.0)
    assert X_c[0,0]==X_c[3,0]==X_c[6,0]
    assert X_c[1,0]==X_c[4,0]==X_c[7,0]
    assert X_c[2,0]==X_c[5,0]==X_c[8,0]
    assert X_c[0,0]!=X_c[1,0]!=X_c[2,0]

def test_introduce_MVs():
    rng = np.random.default_rng()
    X = rng.normal(size=(50,1000))
    mtot = 0.75 # 750 features missing
    X_new = DGP.introduce_MVs(X, mtot)
    assert np.all(np.sum(np.isnan(X_new), axis=1)==750)

def test_createCorrelatedClassCenters():
    basecorr = np.array([[1.0,0.8,0.6],[0.8,1.0,0.2],[0.6,0.2,1.0]])
    correlated_centers = DGP.createCorrelatedClassCenters(basecorr)
    corr = pd.DataFrame(correlated_centers).transpose().corr()
    assert np.all(np.isclose(corr, basecorr, atol=2.5e-2)) # run a few times, when this assert fails. This can happen by chance
    assert correlated_centers.shape[0]==3
    assert correlated_centers.shape[1]==10000

def test_createDataWithBlockCorrelation():
    classdistribution = [2,7,12]
    base_corr = np.array([[1.0,0.3,0.6],[0.3,1.0,0.7],[0.6,0.7,1.0]])
    mvs = 0.8
    cluster_std = 0.2
    X,y, corr = DGP.createDataWithBlockCorrelation(classdistribution, base_corr, mvs,  cluster_std, return_corr = True, method="pearson")
    assert len(y)==X.shape[0]==sum(classdistribution)
    assert X.shape[1]==10000
    labels, counts =  np.unique(y, return_counts=True)
    assert np.all(np.array_equal(np.sort(counts),np.sort(classdistribution)))
    assert corr.shape[0] == sum(classdistribution)

def test_get_sigma_allblocks():
    corr = np.array([[1,0.1,0.6,0.8],[0.1,1,0.8,0.6],[0.6,0.8,1,0.2],[0.8,0.6,0.2,1]])
    y = np.array([0,0,1,1])
    mean_blockwise_std = np.mean([0.0,0.1,0.1,0.0])
    assert np.isclose([DGP.get_sigma_allblocks(corr, y)],[mean_blockwise_std])
    
def test_diff_mu_12_mu_11():
    corr = np.array([[1,0.1,0.6,0.8],[0.1,1,0.8,0.6],[0.6,0.8,1,0.2],[0.8,0.6,0.2,1]])
    y = np.array([0,0,1,1])
    assert np.isclose([DGP.diff_mu_12_mu_11(corr,y)],[0.6])

def test_diff_mu_13_mu_23():
    corr = np.array([[1,0.1,0.4,0.1,0.8,0.2],[0.1,1,0.3,0.6,0.4,0.8],[0.4,0.3,1,0.4,0.7,0.3],[0.1,0.6,0.4,1,0.9,0.2],[0.8,0.4,0.7,0.9,1.0,0.2],[0.2,0.8,0.3,0.2,0.2,1.0]])
    y = np.array([0,0,1,1,2,2])
    assert np.isclose([DGP.diff_mu_13_mu_23(corr,y)],[np.abs(np.mean([0.8,0.4,0.2,0.8])-np.mean([0.7,0.9,0.3,0.2]))])