# -*- coding: utf-8 -*-
"""
Comparison of F-ACF and KNN (in terms of runtime)

@author: Yannis Schumann
"""
import DGP
import DBC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import ACF
import CustomKNN
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#------------------------------------------------------------------------------
#                   
#                               SETUP
#
#------------------------------------------------------------------------------

n_iter = 60 # number of hyper parameter optimization steps
n_splits = 20 # number of stratified shuffle splits in hyperparameter optimization
test_size=0.1 # size (in percent) of the stratified shuffle test splits in hpo

repetitions = 10 # number of repetitions per parameter configuration in the simulation studies

#------------------------------------------------------------------------------
#                   
#                           TIME COMPLEXITY
#
#------------------------------------------------------------------------------
# no logging, since pretty fast (runs in a few minutes on a desktop computer)

# Setup
classdistribution = [100,100,100]
base_corr = np.array([[1.0,0.9,0.6],[0.9,1.0,0.8],[0.6,0.8,1.0]])
num_points = 25
mv = 0.88

n_refs = [10,20,30]

# dataframe for results
performance = pd.DataFrame(0.0, index=[], columns=["Algorithm", "References", "Train Size", "Runtime", "Value", "sigma", "diff_mu_12_mu_11", "diff_mu_13_mu_23"])

for rep in range(repetitions):
    print(rep)
    # Generate dataset
    X,y,corr = DGP.createDataWithBlockCorrelation(classdistribution, base_corr, mv, cluster_std=2.0, return_corr=True, method="pearson")
    sigma = DGP.get_sigma_allblocks(corr, y)
    diff_mu_12_mu_11 = DGP.diff_mu_12_mu_11(corr, y)
    diff_mu_13_mu_23 = DGP.diff_mu_13_mu_23(corr, y)
    
    # put 60 samples aside for classification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=60, stratify=y)
    
    # Sizes of the training data set
    train_sizes = np.linspace(100, X_train.shape[0]-3, num=num_points, dtype=np.int64) # minus 3 (no. of classes) comes from train-test split in the measurement
    
    # Measurement
    for train_size in train_sizes:
        print(train_size)
        # Consider a subsampled training dataset
        SS_X_train, _, SS_y_train, _ = train_test_split(X_train, y_train, train_size=train_size, stratify=y_train)
        # build and train KNN classifier
        knn = CustomKNN.CustomKNN(n_neighbors=10, weights="uniform", oversampling=False)
        knn.fit(SS_X_train, SS_y_train)
        
        # time KNN
        begin_knn = time.perf_counter()
        y_t_knn = knn.predict(X_test)
        end_knn = time.perf_counter()
        # store result
        performance = performance.append({"Algorithm":"KNN", "References":"Naive (All)", "Train Size":SS_X_train.shape[0], "Runtime":(end_knn-begin_knn)/X_test.shape[0], "Value":f1_score(y_test, y_t_knn, average="macro"), "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23}, ignore_index=True)
        
        dbc = DBC.DBCClassifier()
        dbc.fit(SS_X_train, SS_y_train)
        begin_dbc = time.perf_counter()
        y_t_dbc = dbc.predict(X_test)
        end_dbc = time.perf_counter()
        performance = performance.append({"Algorithm":"DBC", "References":"Naive (All)", "Train Size":SS_X_train.shape[0], "Runtime":(end_dbc-begin_dbc)/X_test.shape[0], "Value":f1_score(y_test, y_t_dbc, average="macro"), "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23}, ignore_index=True)
        
        
        # consider multiple numbers of reference instances
        for n_ref in n_refs:
            print(n_ref)
            # build and train ACF Classifier
            baseline_classifier = SVC(kernel="rbf", C=100) # baseline for ACF
            acf = ACF.ACFClassifier(baseline_classifier, strategy="mean", variant="F-ACF", n_ref = n_ref)
            acf.fit(SS_X_train, SS_y_train)
            
            dbc = DBC.DBCClassifier(variant="F-DBC", n_ref = n_ref)
            dbc.fit(SS_X_train, SS_y_train)
            
            
            # time F-ACF
            begin_acf = time.perf_counter()
            y_t_acf = acf.predict(X_test)
            end_acf = time.perf_counter()
            performance = performance.append({"Algorithm":"F-ACF", "References":n_ref, "Train Size":SS_X_train.shape[0], "Runtime":(end_acf-begin_acf)/X_test.shape[0], "Value":f1_score(y_test, y_t_acf, average="macro"), "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23}, ignore_index=True)
            
            begin_dbc = time.perf_counter()
            y_t_fdbc = dbc.predict(X_test)
            end_dbc = time.perf_counter()
            performance = performance.append({"Algorithm":"F-DBC", "References":n_ref, "Train Size":SS_X_train.shape[0], "Runtime":(end_dbc-begin_dbc)/X_test.shape[0], "Value":f1_score(y_test, y_t_fdbc, average="macro"), "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23}, ignore_index=True)
            
    
    performance.to_csv("Results/raw/FACF_KNN_TimeComplexity.csv")