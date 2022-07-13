# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:13:55 2022

@author: Yannis Schumann
"""

import numpy as np
import DBC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score


def test_evenlyDistributedCircles():
    """
    This function corresponds to Example 1 from the supplementary information of the original DBC article.
    Returns
    -------
    None.

    """
    n_1= 60 # counted from tudy
    n_2 = 30 # same
    phi_1 = np.linspace(0.0,2*np.pi, num=n_1+1)[:-1]
    phi_2 = np.linspace(0.0,2*np.pi, num=n_2+1)[:-1]
    r_1 = 2
    r_2 = 2
    x_1 = r_1*np.cos(phi_1)
    y_1 = r_1*np.sin(phi_1)
    x_2 = r_2*np.cos(phi_2)+2
    y_2 = r_2*np.sin(phi_2)
    
    X_1 = np.vstack([x_1, y_1]).transpose()
    Y_1 = np.array([0 for _ in range(n_1)])
    X_2 = np.vstack([x_2, y_2]).transpose()
    Y_2 = np.array([1 for _ in range(n_2)])
    
    X = np.vstack((X_1, X_2))
    Y = np.hstack((Y_1, Y_2))
    
    X_dist = (-1)*euclidean_distances(X)
    np.fill_diagonal(X_dist, np.NaN)
    
    loo = LeaveOneOut()
    y_predicted = []
    y_trues = []
    for train_index, test_index in loo.split(X_dist):
        X_train, X_test = X_dist[np.ix_(train_index, train_index)], X_dist[np.ix_(test_index, train_index)]
        y_train, y_test = Y[train_index], Y[test_index]
        dbc = DBC.DBCClassifier(precomputed=True, istest=True)
        dbc.fit(X_train, y_train)
        y_predicted.append(dbc.predict(X_test))
        y_trues.append(y_test)
    
    y_predicted = np.array(y_predicted).ravel()
    y_trues = np.array(y_trues).ravel()
    
    assert accuracy_score(y_trues, y_predicted)==1


def test_simulated_Data_DBC():
    rng = np.random.default_rng()
    J = 50
    samples_per_class = 50
    
    repetitions = 100
    
    accuracies = []
    
    for rep in range(repetitions):
        a_j = rng.uniform(high=10, low=-10, size=J)
        b_j = rng.uniform(high=2, low=-2, size=J)
        e_jk = rng.normal(loc=0, scale=2, size=(2*samples_per_class, J))
        
        classes = np.hstack(([0 for _ in range(samples_per_class)], [1 for _ in range(samples_per_class)]))
        
        y_jk = a_j+b_j*classes.reshape((100,1)) + e_jk
        
        skf = StratifiedKFold(n_splits=3)
        
        y_true = []
        y_pred = []
        
        for train_index, test_index in skf.split(y_jk, classes):
            X_train, X_test = y_jk[train_index], y_jk[test_index]
            y_train, y_test = classes[train_index], classes[test_index]
            y_true.extend(list(y_test))
            dbc = DBC.DBCClassifier()
            dbc.fit(X_train, y_train)
            y_pred.extend(list(dbc.predict(X_test)))
        accuracies.append(accuracy_score(y_true, y_pred)) # 0.9503999999999999 +- 0.02727343029396925
    assert np.mean(accuracies)+np.std(accuracies)>=0.9657 # value from DBC article