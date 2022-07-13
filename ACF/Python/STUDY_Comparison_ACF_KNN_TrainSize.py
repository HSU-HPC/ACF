# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:02:11 2022

@author: Yannis Schumann
"""

import DGP
import HPO_utils
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import f1_score

#------------------------------------------------------------------------------
#                   
#                               SETUP
#
#------------------------------------------------------------------------------

n_iter = 60 # number of hyper parameter optimization steps
n_splits = 10 # number of stratified shuffle splits in hyperparameter optimization
test_size=0.1 # size (in percent) of the stratified shuffle test splits in hpo

repetitions = 10 # number of repetitions per parameter configuration in the simulation studies

#------------------------------------------------------------------------------
#                   
#                        IMPACT OF DATA SET SIZE
#
#------------------------------------------------------------------------------

# Set up logger
logging.basicConfig(filename='logs/ACF_KNN_Comparison_Size.log', level=logging.INFO)
 
# relative class abundances
base_sizes = np.array([7/15,3/15,5/15])
sizes = np.linspace(50,400,num=20, dtype=int) # dataset sizes
base_corr = np.array([[1.0,0.9,0.6],[0.9,1.0,0.8],[0.6,0.8,1.0]])
mv = 0.81 # missing values

# dataframe for results
classifier_performance = pd.DataFrame(0.0, index=[], columns=["Metric", "Value", "Method", "Algorithm", "mv", "sigma", "diff_mu_12_mu_11", "diff_mu_13_mu_23", "Neighbors","size"])
# measurements
for rep in range(repetitions):
    logging.info("----------------REPETITION "+str(rep)+"-----------------")
    for size in sizes:
        
        # get according absolute number of observations per class
        classdistribution = base_sizes*size
        classdistribution = classdistribution.astype(int)
        
        # Generating datasets
        X,y,corr = DGP.createDataWithBlockCorrelation(classdistribution, base_corr, mv, cluster_std=2.0, return_corr=True, method="pearson")
        # Characterizing dataset
        sigma = DGP.get_sigma_allblocks(corr, y)
        diff_mu_12_mu_11 = DGP.diff_mu_12_mu_11(corr, y)
        diff_mu_13_mu_23 = DGP.diff_mu_13_mu_23(corr, y)
        
        # Testing ACF Classifiers
        logging.info("Testing ACF Classifiers @ size "+str(size))
        acv_svc = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False)
        acv_rf = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False)
        acv_ridge = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False)
        # KNN Classifiers
        logging.info("Testing KNN Classifiers @ size "+str(size))
        knn_normal, knn_normal_neighbors = HPO_utils.characterize_KNN(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=False, return_neighbors = True)
        knn_oversampling, knn_oversampling_neighbors = HPO_utils.characterize_KNN(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=True, return_individual=False, return_neighbors = True)
        # DBC Classifier
        logging.info("Testing DBC Classifiers  @ size "+str(size))
        dbc_classifier = HPO_utils.characterize_DBC(corr, y, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"))
        
        
        #Appending results to result dataframe
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_svc, "Method":"ACF", "Algorithm":"ACF+SVC", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "size": size}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_rf, "Method":"ACF", "Algorithm":"ACF+RF", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "size": size}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_ridge, "Method":"ACF", "Algorithm":"ACF+Ridge", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "size": size}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":knn_normal, "Method":"KNN", "Algorithm":"KNN", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":knn_normal_neighbors, "size": size}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":knn_oversampling, "Method":"KNN", "Algorithm":"KNN+ROs", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":knn_oversampling_neighbors, "size": size}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":dbc_classifier, "Method":"DBC", "Algorithm":"DBC", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":knn_oversampling_neighbors, "size": size}, ignore_index=True)
        
    # save after each repetition
    classifier_performance.to_csv("Results/raw/ACFKNNSizeComparison.csv")