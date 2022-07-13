# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:13:29 2022

This is to demonstrate that adding potential covariables to ACF is
a) possible and
b) potentially beneficial.

@author: Yannis Schumann
"""

import logging
import pandas as pd
import dataset_utils
import HPO_utils
from sklearn.metrics import f1_score

#------------------------------------------------------------------------------
#                   
#                               SETUP
#
#------------------------------------------------------------------------------

n_iter = 60 # number of hyper parameter optimization steps
n_splits = 10 # number of stratified shuffle splits in hyperparameter optimization
test_size=0.1 # size (in percent) of the stratified shuffle test splits in hpo

repetitions = 40 # number of repetitions per parameter configuration in the simulation studies
recompute_corr = True # whether to recompute the correlation matrices once

#------------------------------------------------------------------------------
#                   
#                               COMPUTATIONS
#
#------------------------------------------------------------------------------

dataset_list = []
value_list = []
method_list = []
algorithm_list = []
repetition_list = []

# Set up logger
logging.basicConfig(filename='logs/petralia_covariable_comparison.log', level=logging.INFO)

for rep in range(repetitions):
    logging.info("----------------REPETITION "+str(rep)+"-----------------")
    # load petralia dataset with additional covariable
    X, y, corr, considered_cv = dataset_utils.get_IRS_Petralia_withCovariable(recompute_corr)
    logging.info("Testing ACF+SVC on Petralia with covariable")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True, X_c = considered_cv)
    logging.info("Testing ACF+RF on Petralia with covariable")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True, X_c = considered_cv)
    logging.info("Testing ACF+Ridge on Petralia with covariable")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True, X_c = considered_cv)
    for split_idx in range(len(individual_metric_acf_svc)):
        dataset_list.extend(["Petralia IRS with Covariable" for _ in range(3)])
        value_list.append(individual_metric_acf_svc[split_idx])
        value_list.append(individual_metric_acf_rf[split_idx])
        value_list.append(individual_metric_acf_ridge[split_idx])
        method_list.extend(["ACF, with covariable" for _ in range(3)])
        algorithm_list.extend(["ACF+SVC","ACF+RF","ACF+Ridge"])
        repetition_list.extend([rep, rep, rep])
    
    print("Repetition ",rep," Without Covariables")
    X, y, corr = dataset_utils.get_IRS_Petralia(recompute_corr)
    logging.info("Testing ACF+SVC on Petralia without covariable")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Testing ACF+RF on Petralia without covariable")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Testing ACF+Ridge on Petralia without covariable")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    for split_idx in range(len(individual_metric_acf_svc)):
        dataset_list.extend(["Petralia IRS" for _ in range(3)])
        value_list.append(individual_metric_acf_svc[split_idx])
        value_list.append(individual_metric_acf_rf[split_idx])
        value_list.append(individual_metric_acf_ridge[split_idx])
        method_list.extend(["ACF, correlation only" for _ in range(3)])
        algorithm_list.extend(["ACF+SVC","ACF+RF","ACF+Ridge"])
        repetition_list.extend([rep, rep, rep])
    
    recompute_corr = False

    results = pd.DataFrame({"Dataset":dataset_list, "Value":value_list, "Method":method_list, "Algorithm":algorithm_list, "Repetition":repetition_list})
    results.to_csv("Results/raw/petralia_covariable_comparison.csv")