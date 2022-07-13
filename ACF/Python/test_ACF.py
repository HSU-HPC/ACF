# -*- coding: utf-8 -*-
"""
pytest for ACF

@author: Yannis Schumann
"""
import ACF
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import numpy as np
import pandas as pd

def test_ACF_Constructor():
    baseline_classifier = SVC()
    corr_type="pearson"
    strategy="mean"
    variant=None
    n_ref = 10
    precomputed = True
    scale_Corrmat = "Standard"
    normalize_AC = "l2"
    
    ACF_CL = ACF.ACFClassifier(baseline_classifier,corr_type=corr_type,strategy=strategy,variant=variant,n_ref = n_ref,precomputed = precomputed, scale_Corrmat = scale_Corrmat, normalize_AC = normalize_AC)
    assert ACF_CL.baseline_classifier == baseline_classifier
    assert ACF_CL.corr_type == corr_type
    assert ACF_CL.strategy == strategy
    assert ACF_CL.variant == variant
    assert ACF_CL.n_ref == n_ref
    assert ACF_CL.precomputed == precomputed
    assert ACF_CL.scale_Corrmat == scale_Corrmat
    assert ACF_CL.normalize_AC == normalize_AC


def test_normalize_average_correlations():
    rng = np.random.default_rng()
    X = rng.normal(size=(20,200))
    baseline_classifier = DummyClassifier()
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean", normalize_AC=None)    
    corr = pd.DataFrame(X).transpose().corr()
    # nothing should change
    norm_corr = ACF_CL.__normalize_average_correlations__(corr)
    assert np.all(norm_corr==corr.values)
    # should be normalized by l1 norm
    ACF_CL.normalize_AC = "l1"
    norm_corr = ACF_CL.__normalize_average_correlations__(corr)
    assert np.all(np.isclose(np.sum(np.abs(norm_corr), axis=1),np.ones(shape=(X.shape[0]),dtype=np.float64), atol=1e-12))
    # should be normalized by l2 norm
    ACF_CL.normalize_AC = "l2"
    norm_corr = ACF_CL.__normalize_average_correlations__(corr)
    assert np.all(np.isclose(np.sum(np.power(norm_corr,2), axis=1),np.ones(shape=(X.shape[0]),dtype=np.float64)))
    

def test_scale_correlations():
    rng = np.random.default_rng()
    X = rng.normal(size=(20,200))
    baseline_classifier = DummyClassifier()
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean", scale_Corrmat=None)    
    corr = pd.DataFrame(X).transpose().corr()
    # nothing should change
    norm_corr = ACF_CL.__scale_correlations__(corr)
    assert np.all(norm_corr==corr.values)
    # should be standardized per row
    corr.loc[1,3] = corr.loc[3,1] = corr.loc[11, 2] = corr.loc[2, 11] = np.NaN
    ACF_CL.scale_Corrmat = "Standard"
    norm_corr = ACF_CL.__scale_correlations__(corr)
    assert np.all(np.isclose(np.nanstd(norm_corr, axis=1), np.ones(shape=(X.shape[0]),dtype=np.float64), atol=1e-12))
    assert np.all(np.isclose(np.nanmean(norm_corr, axis=1), np.zeros(shape=(X.shape[0]),dtype=np.float64), atol=1e-12))
    # should be scaled between 0 and 1
    ACF_CL.scale_Corrmat = "MinMax"
    norm_corr = ACF_CL.__scale_correlations__(corr)
    assert np.all(np.isclose(np.nanmin(norm_corr, axis=1), np.zeros(shape=(X.shape[0]),dtype=np.float64), atol=1e-12))
    assert np.all(np.isclose(np.nanmax(norm_corr, axis=1), np.ones(shape=(X.shape[0]),dtype=np.float64), atol=1e-12))
    
    


def test_ACF_get_indices_per_class():
    baseline_classifier = SVC()
    
    
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None)
    ACF_CL.classes_ = np.array([1,2,3])
    ACF_CL.y_ = [1,2,1,1,2,3,1,3,3,3]
    indices_per_class = ACF_CL.__get_indices_per_class__()
    print(indices_per_class)
    assert len(indices_per_class)==3
    assert np.all(indices_per_class[0] == np.array([0,2,3,6]))
    assert np.all(indices_per_class[1] == np.array([1,4]))
    assert np.all(indices_per_class[2] == np.array([5,7,8,9]))
    
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "B-ACF")
    ACF_CL.classes_ = np.array([1,2,3])
    ACF_CL.y_ = np.array([1,2,1,1,2,3,1,3,3,3])
    indices_per_class = ACF_CL.__get_indices_per_class__()
    print(indices_per_class)
    assert len(indices_per_class)==3
    assert np.all(indices_per_class[0] == np.array([0,2,3,6]))
    assert np.all(indices_per_class[1] == np.array([1,4]))
    assert np.all(indices_per_class[2] == np.array([5,7,8,9]))
    
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "F-ACF", n_ref = 2)
    ACF_CL.classes_ = np.array([1,2,3])
    ACF_CL.y_ = np.array([1,2,1,1,2,3,1,3,3,3])
    indices_per_class = ACF_CL.__get_indices_per_class__()
    print(indices_per_class)
    assert len(indices_per_class)==3
    assert len(indices_per_class[0])==len((indices_per_class[1]))==len(indices_per_class[2])==2
    for k in indices_per_class[0]:
        assert k in np.array([0,2,3,6])
    for k in indices_per_class[1]:
        assert k in np.array([1,4])
    for k in indices_per_class[2]:
        assert k in np.array([5,7,8,9])    


def test_ACF_get_average_correlations():
    baseline_classifier = SVC()
    
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean")
    indices_per_class = [np.array([0,1]),np.array([2,3])]
    corr_mat = np.array([[np.NaN,0.8,0.2,0.3],[0.8,np.NaN,0.4,np.NaN],[0.2,0.4,np.NaN,0.7],[0.3,np.NaN,0.7,np.NaN]])
    ACF_CL.corr_mat = pd.DataFrame(corr_mat)
    ACF_CL.indices_per_class = indices_per_class
    ACF_CL.classes_ = np.array([0,1])
    average_corr = ACF_CL.__get_average_correlations__(ACF_CL.corr_mat)        
    expected_average_correlations = np.array([[0.8,0.25],[0.8,0.4],[0.3,0.7],[0.3,0.7]])    
    # numerical precision
    assert np.all(np.isclose(average_corr,expected_average_correlations, atol=1e-15))
    
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="median")
    indices_per_class = [np.array([0,1]),np.array([2,3])]
    corr_mat = np.array([[np.NaN,0.8,0.2,0.3],[0.8,np.NaN,0.4,np.NaN],[0.2,0.4,np.NaN,0.7],[0.3,np.NaN,0.7,np.NaN]])
    ACF_CL.corr_mat = pd.DataFrame(corr_mat)
    ACF_CL.indices_per_class = indices_per_class
    ACF_CL.classes_ = np.array([0,1])
    average_corr = ACF_CL.__get_average_correlations__(ACF_CL.corr_mat)
    assert np.all(np.isclose(average_corr,expected_average_correlations, atol=1e-15))
    

def test_ACF_fit():
    rng = np.random.default_rng()
    X = rng.normal(size=(50,200))
    y = rng.choice([1,2,3], size=50)
    baseline_classifier = DummyClassifier()
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean")
    corr = pd.DataFrame(X).transpose().corr()
    np.fill_diagonal(corr.values, np.NaN)
    ACF_CL.fit(X,y)
    corr = corr[ACF_CL.corr_mat.columns.values]
    np.testing.assert_allclose(ACF_CL.corr_mat.values, corr.values, atol=1e-10)

    biased_elements = []
    for t1,t2 in zip(rng.choice(np.arange(0,50), size=5),rng.choice(np.arange(0,50), size=5)):
        biased_elements.append((t1,t2))
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "B-ACF", strategy="mean")
    ACF_CL.fit(X,y, biased_elements = biased_elements)
    for el in biased_elements:
        assert np.isnan(ACF_CL.corr_mat.loc[el[0],el[1]])
        assert np.isnan(ACF_CL.corr_mat.loc[el[1],el[0]])
    
    X = rng.normal(size=(50,200))
    y = rng.choice([1,2,3], size=50)
    baseline_classifier = DummyClassifier()
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "F-ACF", strategy="mean", n_ref = 5)
    corr = pd.DataFrame(X).transpose().corr()
    np.fill_diagonal(corr.values, np.NaN)
    ACF_CL.fit(X,y)
    corr = corr[np.concatenate(ACF_CL.indices_per_class)]
    np.testing.assert_allclose(ACF_CL.corr_mat.values, corr.values, atol=1e-10)
    pass


def test_ACF_worksOnCopies():
    rng = np.random.default_rng()
    X_train = rng.normal(size=(50,200))
    X_train_copy = X_train.copy()
    X_test = rng.normal(size=(5,200))
    X_test_copy = X_test.copy()
    y_train = rng.choice([1,2,3], size=50)
    baseline_classifier = DummyClassifier()
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean")
    ACF_CL.fit(X_train, y_train)
    y_test = ACF_CL.predict(X_test)
    assert np.array_equal(X_train, X_train_copy)
    assert np.array_equal(X_test, X_test_copy)

    rng = np.random.default_rng()
    X_train = rng.normal(size=(50,200))
    X_train_copy = X_train.copy()
    X_test = rng.normal(size=(5,200))
    X_test_copy = X_test.copy()
    y_train = rng.choice([1,2,3], size=50)
    baseline_classifier = DummyClassifier()
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean", scale_Corrmat="MinMax")
    ACF_CL.fit(X_train, y_train)
    y_test = ACF_CL.predict(X_test)
    assert np.array_equal(X_train, X_train_copy)
    assert np.array_equal(X_test, X_test_copy)
    
    baseline_classifier = KNeighborsClassifier(n_neighbors=1)
    X_train = pd.DataFrame(rng.normal(size=(50,200))).transpose().corr()
    X_train_copy = X_train.copy()
    y_train = rng.choice([1,2,3], p=[1/3, 1/3, 1/3], size=50)
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "F-ACF", n_ref=5, strategy="mean", precomputed=True)
    ACF_CL.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    X_test = X_train.loc[test_indices]
    X_test_copy = X_test.copy()
    y_predicted = ACF_CL.predict(X_test)
    assert np.array_equal(X_train.values, X_train_copy.values)
    assert np.array_equal(X_test.values, X_test_copy.values)
    
    baseline_classifier = KNeighborsClassifier(n_neighbors=1)
    X_train = pd.DataFrame(rng.normal(size=(50,200))).transpose().corr()
    X_train_copy = X_train.copy()
    y_train = rng.choice([1,2,3], p=[1/3, 1/3, 1/3], size=50)
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "F-ACF", n_ref=5, strategy="mean", precomputed=True, scale_Corrmat="MinMax")
    ACF_CL.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    X_test = X_train.loc[test_indices]
    X_test_copy = X_test.copy()
    y_predicted = ACF_CL.predict(X_test)
    assert np.array_equal(X_train.values, X_train_copy.values)
    assert np.array_equal(X_test.values, X_test_copy.values)


def test_ACF_total():
    rng = np.random.default_rng()
    X_train = rng.normal(size=(50,200))
    X_test = rng.normal(size=(5,200))
    y_train = rng.choice([1,2,3], size=50)
    baseline_classifier = DummyClassifier()
    
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean")
    ACF_CL.fit(X_train, y_train)
    y_test = ACF_CL.predict(X_test)
    assert len(y_test)== X_test.shape[0]
    # same indices used and same order
    assert np.all(np.array(ACF_CL.test_corr_mat.columns.values)==np.array(ACF_CL.corr_mat.columns.values))
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "F-ACF", strategy="mean", n_ref=5)
    ACF_CL.fit(X_train, y_train)
    y_test = ACF_CL.predict(X_test)
    assert len(y_test)== X_test.shape[0]
    # same indices used and same order
    assert np.all(np.array(ACF_CL.test_corr_mat.columns.values)==np.array(ACF_CL.corr_mat.columns.values))

    baseline_classifier = KNeighborsClassifier(n_neighbors=1)
    X_train = pd.DataFrame(rng.normal(size=(50,200))).transpose().corr()
    y_train = rng.choice([1,2,3], p=[0.5,0.25,0.25], size=50)
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean", precomputed=True, scale_Corrmat="Standard")
    ACF_CL.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    X_test = X_train.loc[test_indices]
    y_predicted = ACF_CL.predict(X_test)
    assert np.all(y_train[test_indices]==y_predicted)
    
    baseline_classifier = KNeighborsClassifier(n_neighbors=1)
    X_train = pd.DataFrame(rng.normal(size=(50,200))).transpose().corr()
    y_train = rng.choice([1,2,3], p=[1/3, 1/3, 1/3], size=50)
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "F-ACF", n_ref=5, strategy="mean", precomputed=True)
    ACF_CL.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    X_test = X_train.loc[test_indices]
    y_predicted = ACF_CL.predict(X_test)
    assert np.all(y_train[test_indices]==y_predicted)
    
    baseline_classifier = KNeighborsClassifier(n_neighbors=1)
    X_train = pd.DataFrame(rng.normal(size=(50,200)))
    y_train = rng.choice([1,2,3], p=[0.5,0.25,0.25], size=50)
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "B-ACF", strategy="mean", precomputed=False)
    ACF_CL.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    biased_elements = [(idx, test_indices[idx]) for idx in range(len(test_indices))]
    X_test = X_train.loc[test_indices]
    y_predicted = ACF_CL.predict(X_test, biased_elements=biased_elements)
    assert np.all(y_train[test_indices]==y_predicted)

    baseline_classifier = KNeighborsClassifier(n_neighbors=1)
    X_train = pd.DataFrame(rng.normal(size=(50,200)))
    y_train = rng.choice([1,2,3], p=[0.5,0.25,0.25], size=50)
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = "B-ACF", strategy="mean", precomputed=False, scale_Corrmat="Standard")
    ACF_CL.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    biased_elements = [(idx, test_indices[idx]) for idx in range(len(test_indices))]
    X_test = X_train.loc[test_indices]
    y_predicted = ACF_CL.predict(X_test, biased_elements=biased_elements)
    assert np.all(y_train[test_indices]==y_predicted)
    
    baseline_classifier = KNeighborsClassifier(n_neighbors=1)
    X_train = rng.normal(size=(3,200))
    y_train = np.array([1,2,3])
    X_train = np.vstack([X_train for _ in range(20)])
    y_train = np.concatenate([y_train for _ in range(20)])
    ACF_CL = ACF.ACFClassifier(baseline_classifier, variant = None, strategy="mean", scale_Corrmat="Standard", precomputed=False)
    ACF_CL.fit(X_train, y_train)
    test_indices = rng.choice(np.arange(0,50, dtype=np.int64), size=25)
    X_test = X_train[test_indices,:]
    y_predicted = ACF_CL.predict(X_test)
    assert np.all(y_train[test_indices]==y_predicted)
    
    