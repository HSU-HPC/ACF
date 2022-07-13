# -*- coding: utf-8 -*-
"""
STUDY - Comparison of ACF, KNN and a 'standard' classifier
on 2 proteomic datasets

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
logging.basicConfig(filename='logs/proteomicsComparison.log', level=logging.INFO)

classifier_performance = pd.DataFrame(0.0, index=[], columns=["Dataset", "Metric", "Value", "Method", "Algorithm", "Repetition"])

#iterations
for rep in range(repetitions):
    logging.info("----------------REPETITION "+str(rep)+"-----------------")
    logging.info("Loading raw Petralia dataset")
    data, data_meta, corr, masked_corr = dataset_utils.get_raw_Petralia(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-SVC on raw Petralia dataset")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(masked_corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-RF on raw Petralia dataset")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(masked_corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-Ridge on raw Petralia dataset")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(masked_corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting KNN on raw Petralia dataset")
    total_metric_knn, individual_metric_knn, n_train_knn, n_test_knn = HPO_utils.characterize_KNN(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=True, return_neighbors = False)
    logging.info("Starting DBC on raw Petralia dataset")
    total_metric_dbc, individual_metric_dbc = HPO_utils.characterize_DBC(masked_corr, data_meta, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True)
    logging.info("Starting SVC classifier on raw Petralia dataset")
    total_metric_remainingGenes_svc, individual_metric_remainingGenes_svc, n_train_remainingGenes_svc, n_test_remainingGenes_svc = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="SVC")
    logging.info("Starting RF classifier on raw Petralia dataset")
    total_metric_remainingGenes_rf, individual_metric_remainingGenes_rf, n_train_remainingGenes_rf, n_test_remainingGenes_rf = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="RF")
    logging.info("Starting Ridge classifier on raw Petralia dataset")
    total_metric_remainingGenes_ridge, individual_metric_remainingGenes_ridge, n_train_remainingGenes_ridge, n_test_remainingGenes_ridge = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="Ridge")
    
    # store results 
    for split_idx in range(len(individual_metric_acf_svc)):
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(raw)", "Metric":"F1 Macro", "Value":individual_metric_acf_svc[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(raw)", "Metric":"F1 Macro", "Value":individual_metric_acf_rf[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(raw)", "Metric":"F1 Macro", "Value":individual_metric_acf_ridge[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+Ridge", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(raw)", "Metric":"F1 Macro", "Value":individual_metric_knn[split_idx], "Method":"KNN", "Algorithm":"KNN", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(raw)", "Metric":"F1 Macro", "Value":individual_metric_dbc[split_idx], "Method":"DBC", "Algorithm":"DBC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(raw)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_svc[split_idx], "Method":"Listwise Deletion", "Algorithm":"SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(raw)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_rf[split_idx], "Method":"Listwise Deletion", "Algorithm":"RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(raw)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_ridge[split_idx], "Method":"Listwise Deletion", "Algorithm":"Ridge", "Repetition":rep}, ignore_index=True)
        
    classifier_performance.to_csv("Results/raw/full-comp-proteomics.csv")
    
    
    logging.info("Loading IRS-corrected Petralia dataset")
    data, data_meta, corr = dataset_utils.get_IRS_Petralia(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-SVC on IRS Petralia dataset")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-RF on IRS Petralia dataset")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-Ridge on IRS Petralia dataset")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting KNN on IRS Petralia dataset")
    total_metric_knn, individual_metric_knn, n_train_knn, n_test_knn = HPO_utils.characterize_KNN(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=True, return_neighbors = False)
    logging.info("Starting DBC on IRS Petralia dataset")
    total_metric_dbc, individual_metric_dbc = HPO_utils.characterize_DBC(corr, data_meta, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True)
    logging.info("Starting SVC classifier on IRS Petralia dataset")
    total_metric_remainingGenes_svc, individual_metric_remainingGenes_svc, n_train_remainingGenes_svc, n_test_remainingGenes_svc = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="SVC")
    logging.info("Starting RF classifier on IRS Petralia dataset")
    total_metric_remainingGenes_rf, individual_metric_remainingGenes_rf, n_train_remainingGenes_rf, n_test_remainingGenes_rf = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="RF")
    logging.info("Starting Ridge classifier on IRS Petralia dataset")
    total_metric_remainingGenes_ridge, individual_metric_remainingGenes_ridge, n_train_remainingGenes_ridge, n_test_remainingGenes_ridge = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="Ridge")
    
    # store results 
    for split_idx in range(len(individual_metric_acf_svc)):
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(IRS)", "Metric":"F1 Macro", "Value":individual_metric_acf_svc[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(IRS)", "Metric":"F1 Macro", "Value":individual_metric_acf_rf[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(IRS)", "Metric":"F1 Macro", "Value":individual_metric_acf_ridge[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+Ridge", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(IRS)", "Metric":"F1 Macro", "Value":individual_metric_knn[split_idx], "Method":"KNN", "Algorithm":"KNN", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(IRS)", "Metric":"F1 Macro", "Value":individual_metric_dbc[split_idx], "Method":"DBC", "Algorithm":"DBC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(IRS)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_svc[split_idx], "Method":"Listwise Deletion", "Algorithm":"SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(IRS)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_rf[split_idx], "Method":"Listwise Deletion", "Algorithm":"RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia(IRS)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_ridge[split_idx], "Method":"Listwise Deletion", "Algorithm":"Ridge", "Repetition":rep}, ignore_index=True)
        
    classifier_performance.to_csv("Results/raw/full-comp-proteomics.csv")
    
    
    
    logging.info("Loading raw BRCA dataset")
    data, data_meta, corr, masked_corr = dataset_utils.get_raw_brcapam50(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-SVC on raw BRCA dataset")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(masked_corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-RF on raw BRCA dataset")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(masked_corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-Ridge on raw BRCA dataset")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(masked_corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting KNN on raw BRCA dataset")
    total_metric_knn, individual_metric_knn, n_train_knn, n_test_knn = HPO_utils.characterize_KNN(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=True, return_neighbors = False)
    logging.info("Starting DBC on IRS Petralia dataset")
    total_metric_dbc, individual_metric_dbc = HPO_utils.characterize_DBC(masked_corr, data_meta, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True)
    logging.info("Starting SVC classifier on IRS Petralia dataset")
    total_metric_remainingGenes_svc, individual_metric_remainingGenes_svc, n_train_remainingGenes_svc, n_test_remainingGenes_svc = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="SVC")
    logging.info("Starting RF classifier on IRS Petralia dataset")
    total_metric_remainingGenes_rf, individual_metric_remainingGenes_rf, n_train_remainingGenes_rf, n_test_remainingGenes_rf = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="RF")
    logging.info("Starting Ridge classifier on IRS Petralia dataset")
    total_metric_remainingGenes_ridge, individual_metric_remainingGenes_ridge, n_train_remainingGenes_ridge, n_test_remainingGenes_ridge = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="Ridge")
    
    # store results 
    for split_idx in range(len(individual_metric_acf_svc)):
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(raw)", "Metric":"F1 Macro", "Value":individual_metric_acf_svc[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(raw)", "Metric":"F1 Macro", "Value":individual_metric_acf_rf[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(raw)", "Metric":"F1 Macro", "Value":individual_metric_acf_ridge[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+Ridge", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(raw)", "Metric":"F1 Macro", "Value":individual_metric_knn[split_idx], "Method":"KNN", "Algorithm":"KNN", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(raw)", "Metric":"F1 Macro", "Value":individual_metric_dbc[split_idx], "Method":"DBC", "Algorithm":"DBC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(raw)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_svc[split_idx], "Method":"Listwise Deletion", "Algorithm":"SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(raw)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_rf[split_idx], "Method":"Listwise Deletion", "Algorithm":"RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(raw)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_ridge[split_idx], "Method":"Listwise Deletion", "Algorithm":"Ridge", "Repetition":rep}, ignore_index=True)
    
    classifier_performance.to_csv("Results/raw/full-comp-proteomics.csv")

    logging.info("Loading IRS-corrected BRCA dataset")
    data, data_meta, corr = dataset_utils.get_irs_brcapam50(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-SVC on IRS BRCA dataset")
    total_metric_acf_svc, individual_metric_acf_svc, n_train_acf_svc, n_test_acf_svc = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-RF on IRS BRCA dataset")
    total_metric_acf_rf, individual_metric_acf_rf, n_train_acf_rf, n_test_acf_rf = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="RF", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting ACF-Ridge on IRS BRCA dataset")
    total_metric_acf_ridge, individual_metric_acf_ridge, n_train_acf_ridge, n_test_acf_ridge = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="Ridge", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    logging.info("Starting KNN on IRS BRCA dataset")
    total_metric_knn, individual_metric_knn, n_train_knn, n_test_knn = HPO_utils.characterize_KNN(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=True, return_neighbors = False)
    logging.info("Starting DBC on IRS BRCA dataset")
    total_metric_dbc, individual_metric_dbc = HPO_utils.characterize_DBC(corr, data_meta, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True)
    logging.info("Starting SVC classifier on IRS BRCA dataset")
    total_metric_remainingGenes_svc, individual_metric_remainingGenes_svc, n_train_remainingGenes_svc, n_test_remainingGenes_svc = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="SVC")
    logging.info("Starting RF classifier on IRS BRCA dataset")
    total_metric_remainingGenes_rf, individual_metric_remainingGenes_rf, n_train_remainingGenes_rf, n_test_remainingGenes_rf = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="RF")
    logging.info("Starting Ridge classifier on IRS BRCA dataset")
    total_metric_remainingGenes_ridge, individual_metric_remainingGenes_ridge, n_train_remainingGenes_ridge, n_test_remainingGenes_ridge = HPO_utils.characterize_RemainingGenes(data, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual=True, base_clf="Ridge")
    
    for split_idx in range(len(individual_metric_acf_svc)):
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(IRS)", "Metric":"F1 Macro", "Value":individual_metric_acf_svc[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(IRS)", "Metric":"F1 Macro", "Value":individual_metric_acf_rf[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(IRS)", "Metric":"F1 Macro", "Value":individual_metric_acf_ridge[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+Ridge", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(IRS)", "Metric":"F1 Macro", "Value":individual_metric_knn[split_idx], "Method":"KNN", "Algorithm":"KNN", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(IRS)", "Metric":"F1 Macro", "Value":individual_metric_dbc[split_idx], "Method":"DBC", "Algorithm":"DBC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(IRS)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_svc[split_idx], "Method":"Listwise Deletion", "Algorithm":"SVC", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(IRS)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_rf[split_idx], "Method":"Listwise Deletion", "Algorithm":"RF", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"BRCA(IRS)", "Metric":"F1 Macro", "Value":individual_metric_remainingGenes_ridge[split_idx], "Method":"Listwise Deletion", "Algorithm":"Ridge", "Repetition":rep}, ignore_index=True)
        
    classifier_performance.to_csv("Results/raw/full-comp-proteomics.csv")
    
    # recomputing the correlation matrix once is sufficient!
    recompute_corr = False
    