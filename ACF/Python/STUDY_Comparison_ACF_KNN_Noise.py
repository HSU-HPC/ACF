# -*- coding: utf-8 -*-
"""
Comparison with KNN on simulated datasets

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
std_covariables=0.015 # the standard deviation of the additional covariable

repetitions = 10 # number of repetitions per parameter configuration in the simulation studies


#------------------------------------------------------------------------------
#                   
#                           IMPACT OF NOISE
#
#------------------------------------------------------------------------------

# Set up logger
logging.basicConfig(filename='logs/ACF_KNN_Comparison_Noise.log', level=logging.INFO)

# Configuration
classdistribution = [70,30,50]
base_corr = np.array([[1.0,0.9,0.6],[0.9,1.0,0.8],[0.6,0.8,1.0]])
missing_values = 0.915-np.logspace(np.log10(0.01), np.log10(0.9), num=100)

# Result dataframe
classifier_performance = pd.DataFrame(0.0, index=[], columns=["Metric", "Value", "Method", "Algorithm", "mv", "sigma", "diff_mu_12_mu_11", "diff_mu_13_mu_23", "Neighbors", "Repetition"])

for rep in range(repetitions):
    logging.info("----------------REPETITION "+str(rep)+"-----------------")
    for mv in missing_values:
        # Generating datasets
        X,y,corr,X_c = DGP.createDataWithBlockCorrelation(classdistribution, base_corr, mv, cluster_std=2.0, return_corr=True, method="pearson", make_covariable=True, std_covariables=std_covariables)
        # Characterizing dataset
        sigma = DGP.get_sigma_allblocks(corr, y)
        diff_mu_12_mu_11 = DGP.diff_mu_12_mu_11(corr, y)
        diff_mu_13_mu_23 = DGP.diff_mu_13_mu_23(corr, y)
        
        # Testing ACF Classifiers
        logging.info("Testing ACF Classifiers @"+str(mv)+" % missing values, without additional covariables")
        acv_svc = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False)
        acv_rf = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False)
        acv_ridge = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False)
        # Testing ACF Classifiers - with additional covariable
        logging.info("Testing ACF Classifiers @"+str(mv)+" % missing values, with additional covariables")
        acv_svc_c = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False, X_c=X_c)
        acv_rf_c = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False, X_c=X_c)
        acv_ridge_c = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False, X_c=X_c)
        # KNN Classifiers
        logging.info("Testing KNN Classifiers @"+str(mv)+" % missing values")
        knn_normal, knn_normal_neighbors = HPO_utils.characterize_KNN(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=False, return_neighbors = True)
        knn_oversampling, knn_oversampling_neighbors = HPO_utils.characterize_KNN(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=True, return_individual=False, return_neighbors = True)
        # DBC Classifier
        logging.info("Testing DBC Classifiers @"+str(mv)+" % missing values")
        dbc_classifier = HPO_utils.characterize_DBC(corr, y, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"))
        
        
        #Appending results to result dataframe
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_svc, "Method":"ACF, correlation-only", "Algorithm":"ACF+SVC", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_rf, "Method":"ACF, correlation-only", "Algorithm":"ACF+RF", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_ridge, "Method":"ACF, correlation-only", "Algorithm":"ACF+Ridge", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_svc_c, "Method":"ACF, with covariable", "Algorithm":"ACF+SVC", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_rf_c, "Method":"ACF, with covariable", "Algorithm":"ACF+RF", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":acv_ridge_c, "Method":"ACF, with covariable", "Algorithm":"ACF+Ridge", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":knn_normal, "Method":"KNN", "Algorithm":"KNN", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":knn_normal_neighbors, "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":knn_oversampling, "Method":"KNN", "Algorithm":"KNN+ROs", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":knn_oversampling_neighbors, "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":dbc_classifier, "Method":"DBC", "Algorithm":"DBC", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
        
        # listwise deletion, if not more than 100 features -- reason is computational performance
        num_features = np.sum(np.all(~np.isnan(X), axis=0))
        if num_features <=100:
            logging.info("Testing Listwise-Deletion Classifiers @"+str(mv)+" % missing values")
            listwise_svc = HPO_utils.characterize_RemainingGenes(X, y,n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=False, base_clf="SVC")
            listwise_rf = HPO_utils.characterize_RemainingGenes(X, y,n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=False, base_clf="RF")
            listwise_ridge = HPO_utils.characterize_RemainingGenes(X, y,n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=False, base_clf="Ridge")
            # and append
            classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":listwise_svc, "Method":"Listwise Deletion", "Algorithm":"SVC", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
            classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":listwise_rf, "Method":"Listwise Deletion", "Algorithm":"RF", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
            classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":listwise_ridge, "Method":"Listwise Deletion", "Algorithm":"Ridge", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
        
        
        # save
        classifier_performance.to_csv("Results/raw/ACFKNNNoiseComparison.csv")
        
    # once per repetition: measure classification performance for covariable alone
    logging.info("Testing Covariable Alone")
    covariable_svc = HPO_utils.characterize_RemainingGenes(X_c, y,n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=False, base_clf="SVC")
    covariable_rf = HPO_utils.characterize_RemainingGenes(X_c, y,n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=False, base_clf="RF")
    covariable_ridge = HPO_utils.characterize_RemainingGenes(X_c, y,n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=False, base_clf="Ridge")
    # and append
    classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":covariable_svc, "Method":"Covariable Alone", "Algorithm":"SVC", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
    classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":covariable_rf, "Method":"Covariable Alone", "Algorithm":"RF", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
    classifier_performance = classifier_performance.append({"Metric":"F1 Macro", "Value":covariable_ridge, "Method":"Covariable Alone", "Algorithm":"Ridge", "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "Neighbors":np.NaN, "Repetition":rep}, ignore_index=True)
