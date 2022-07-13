# -*- coding: utf-8 -*-
"""
STUDY - Comparison of ACF, KNN and a 'standard' classifier

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

repetitions = 10 # number of repetitions per parameter configuration in the simulation studies
recompute_corr = True # whether to recompute the correlation matrices once

#------------------------------------------------------------------------------
#                   
#                               COMPUTATIONS
#
#------------------------------------------------------------------------------


# Set up logger
logging.basicConfig(filename='logs/scRNAseqComparison.log', level=logging.INFO)
# Result dataframe
classifier_performance = pd.DataFrame(0.0, index=[], columns=["Dataset", "Metric", "Value", "Method", "Algorithm", "Repetition"])


#iterations
for rep in range(repetitions):
    logging.info("----------------REPETITION "+str(rep)+"-----------------")
    
    logging.info("Loading Baron dataset")
    data, data_meta, corr = dataset_utils.get_raw_BaronPancreas(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-SVC on Baron dataset")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-RF on Baron dataset")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-Ridge on Baron dataset")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting KNN on Baron dataset")
    total_metric_knn, individual_metric_knn, n_train_knn, n_test_knn = HPO_utils.characterize_KNN(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=True, return_neighbors = False)
    logging.info("Starting DBC on Baron dataset")
    total_metric_dbc, individual_metric_dbc = HPO_utils.characterize_DBC(corr, data_meta, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True)
    logging.info("Starting SVC classifier on Baron dataset")
    total_metric_remainingGenes_svc, individual_metric_remainingGenes_svc, n_train_remainingGenes_svc, n_test_remainingGenes_svc = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="SVC")
    logging.info("Starting RF classifier on Baron dataset")
    total_metric_remainingGenes_rf, individual_metric_remainingGenes_rf, n_train_remainingGenes_rf, n_test_remainingGenes_rf = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="RF")
    logging.info("Starting Ridge classifier on Baron dataset")
    total_metric_remainingGenes_ridge, individual_metric_remainingGenes_ridge, n_train_remainingGenes_ridge, n_test_remainingGenes_ridge = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="Ridge")
    
    
   # store results 
    for split_idx in range(len(individual_metric_acf_svc)):
        classifier_performance = classifier_performance.append({"Dataset":"Baron", "Metric":"F1 Macro", "Value":individual_metric_acf_svc[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Baron", "Metric":"F1 Macro", "Value":individual_metric_acf_rf[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Baron", "Metric":"F1 Macro", "Value":individual_metric_acf_ridge[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+Ridge", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Baron", "Metric":"F1 Macro", "Value":individual_metric_knn[split_idx], "Method":"KNN", "Algorithm":"KNN", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Baron", "Metric":"F1 Macro", "Value":individual_metric_dbc[split_idx], "Method":"DBC", "Algorithm":"DBC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Baron", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_svc[split_idx], "Method":"Listwise Deletion", "Algorithm":"SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Baron", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_rf[split_idx], "Method":"Listwise Deletion", "Algorithm":"RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Baron", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_ridge[split_idx], "Method":"Listwise Deletion", "Algorithm":"Ridge", "Repetition":rep}, ignore_index=True)
        
    classifier_performance.to_csv("Results/raw/full-comp-scRNAseq.csv")
    
    
    logging.info("Loading 3kPBMC dataset")
    data, data_meta, corr = dataset_utils.get_raw_3kpbmc(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-SVC on 3kPBMC dataset")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-RF on 3kPBMC dataset")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-Ridge on 3kPBMC dataset")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting KNN on 3kPBMC dataset")
    total_metric_knn, individual_metric_knn, n_train_knn, n_test_knn = HPO_utils.characterize_KNN(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=True, return_neighbors = False)
    logging.info("Starting DBC on 3kPBMC dataset")
    total_metric_dbc, individual_metric_dbc = HPO_utils.characterize_DBC(corr, data_meta, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True)
    logging.info("Starting SVC classifier on 3kPBMC dataset")
    total_metric_remainingGenes_svc, individual_metric_remainingGenes_svc, n_train_remainingGenes_svc, n_test_remainingGenes_svc = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="SVC")
    logging.info("Starting RF classifier on 3kPBMC dataset")
    total_metric_remainingGenes_rf, individual_metric_remainingGenes_rf, n_train_remainingGenes_rf, n_test_remainingGenes_rf = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="RF")
    logging.info("Starting Ridge classifier on 3kPBMC dataset")
    total_metric_remainingGenes_ridge, individual_metric_remainingGenes_ridge, n_train_remainingGenes_ridge, n_test_remainingGenes_ridge = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="Ridge")
    
    
   # store results 
    for split_idx in range(len(individual_metric_acf_svc)):
        classifier_performance = classifier_performance.append({"Dataset":"3kPBMC", "Metric":"F1 Macro", "Value":individual_metric_acf_svc[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"3kPBMC", "Metric":"F1 Macro", "Value":individual_metric_acf_rf[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"3kPBMC", "Metric":"F1 Macro", "Value":individual_metric_acf_ridge[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+Ridge", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"3kPBMC", "Metric":"F1 Macro", "Value":individual_metric_knn[split_idx], "Method":"KNN", "Algorithm":"KNN", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"3kPBMC", "Metric":"F1 Macro", "Value":individual_metric_dbc[split_idx], "Method":"DBC", "Algorithm":"DBC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"3kPBMC", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_svc[split_idx], "Method":"Listwise Deletion", "Algorithm":"SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"3kPBMC", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_rf[split_idx], "Method":"Listwise Deletion", "Algorithm":"RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"3kPBMC", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_ridge[split_idx], "Method":"Listwise Deletion", "Algorithm":"Ridge", "Repetition":rep}, ignore_index=True)
        
    classifier_performance.to_csv("Results/raw/full-comp-scRNAseq.csv")
    
    
    logging.info("Loading Xin dataset")
    data, data_meta, corr = dataset_utils.get_raw_XinPancreas(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-SVC on Xin dataset")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-RF on Xin dataset")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-Ridge on Xin dataset")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting KNN on Xin dataset")
    total_metric_knn, individual_metric_knn, n_train_knn, n_test_knn = HPO_utils.characterize_KNN(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=True, return_neighbors = False)
    logging.info("Starting DBC on Xin dataset")
    total_metric_dbc, individual_metric_dbc = HPO_utils.characterize_DBC(corr, data_meta, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True)
    logging.info("Starting SVC classifier on Xin dataset")
    total_metric_remainingGenes_svc, individual_metric_remainingGenes_svc, n_train_remainingGenes_svc, n_test_remainingGenes_svc = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="SVC")
    logging.info("Starting RF classifier on Xin dataset")
    total_metric_remainingGenes_rf, individual_metric_remainingGenes_rf, n_train_remainingGenes_rf, n_test_remainingGenes_rf = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="RF")
    logging.info("Starting Ridge classifier on Xin dataset")
    total_metric_remainingGenes_ridge, individual_metric_remainingGenes_ridge, n_train_remainingGenes_ridge, n_test_remainingGenes_ridge = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="Ridge")
    
   # store results 
    for split_idx in range(len(individual_metric_acf_svc)):
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_acf_svc[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_acf_rf[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_acf_ridge[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+Ridge", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_knn[split_idx], "Method":"KNN", "Algorithm":"KNN", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_dbc[split_idx], "Method":"DBC", "Algorithm":"DBC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_svc[split_idx], "Method":"Listwise Deletion", "Algorithm":"SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_rf[split_idx], "Method":"Listwise Deletion", "Algorithm":"RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_ridge[split_idx], "Method":"Listwise Deletion", "Algorithm":"Ridge", "Repetition":rep}, ignore_index=True)
        
    classifier_performance.to_csv("Results/raw/full-comp-scRNAseq.csv")
    
    
    # recomputing the correlation matrix once is sufficient!
    recompute_corr = False
    