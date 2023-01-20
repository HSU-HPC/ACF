# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:51:31 2023

@author: Yannis Schumann
"""

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
recompute_corr = False # whether to recompute the correlation matrices once

#------------------------------------------------------------------------------
#                   
#                               COMPUTATIONS
#
#------------------------------------------------------------------------------


# Set up logger
logging.basicConfig(filename='logs/deepLearning.log', level=logging.INFO)
# Result dataframe
classifier_performance = pd.DataFrame(0.0, index=[], columns=["Dataset", "Metric", "Value", "Method", "Algorithm", "Repetition"])


#iterations
for rep in range(repetitions):
    logging.info("----------------REPETITION "+str(rep)+"-----------------")
    
    logging.info("Loading Xin dataset")
    data, data_meta, corr = dataset_utils.get_raw_XinPancreas(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-MLP on Xin dataset")
    total_metric_acf_mlp, individual_metric_acf_mlp, n_train_acf_mlp, n_test_acf_mlp = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="MLP", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    
    logging.info("Loading IRS-corrected Petralia dataset")
    data, data_meta, corr = dataset_utils.get_IRS_Petralia(recompute_corr = recompute_corr)
    
    logging.info("Starting ACF-MLP on Petralia dataset")
    petralia_total_metric_acf_mlp, petralia_individual_metric_acf_mlp, petralia_n_train_acf_mlp, petralia_n_test_acf_mlp = HPO_utils.characterize_ACF(corr, data_meta, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="MLP", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=True)
    
   # store results 
    for split_idx in range(len(individual_metric_acf_mlp)):
        classifier_performance = classifier_performance.append({"Dataset":"Xin", "Metric":"F1 Macro", "Value":individual_metric_acf_mlp[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+MLP", "Repetition":rep}, ignore_index=True)
        classifier_performance = classifier_performance.append({"Dataset":"Petralia", "Metric":"F1 Macro", "Value":petralia_individual_metric_acf_mlp[split_idx], "Method":"ACF, correlation-only", "Algorithm":"ACF+MLP", "Repetition":rep}, ignore_index=True)
        
    classifier_performance.to_csv("Results/raw/acf_mlp.csv")
    
    
    # recomputing the correlation matrix once is sufficient!
    recompute_corr = False
    