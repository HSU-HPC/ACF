# -*- coding: utf-8 -*-
"""
Comparison ACF/F-ACF

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
n_splits = 20 # number of stratified shuffle splits in hyperparameter optimization
test_size=0.1 # size (in percent) of the stratified shuffle test splits in hpo

repetitions = 10 # number of repetitions per parameter configuration in the simulation studies

#------------------------------------------------------------------------------
#                   
#                               COMPARISON
#
#------------------------------------------------------------------------------

# Set up logger
logging.basicConfig(filename='logs/ACF_F-ACF_Comparison.log', level=logging.INFO)

# Configuration
classdistribution = [80,80,80]
base_corr = np.array([[1.0,0.9,0.6],[0.9,1.0,0.8],[0.6,0.8,1.0]])
missing_values = 0.91-np.logspace(np.log10(0.01), np.log10(0.9), num=20)
pointsalg2 = np.arange(4,65,step=4, dtype=np.int64)

# Result dataframe
classifier_performance = pd.DataFrame(0.0, index=[], columns=["Algorithm", "score", "mv", "sigma", "diff_mu_12_mu_11", "diff_mu_13_mu_23", "points"])

for rep in range(repetitions):
    logging.info("----------------REPETITION "+str(rep)+"-----------------")
    for mv in missing_values:
        found_error = True
        while found_error:
            found_error = False
            # Generating datasets
            X,y,corr = DGP.createDataWithBlockCorrelation(classdistribution, base_corr, mv, cluster_std=2.0, return_corr=True, method="pearson")
            # Characterizing dataset
            sigma = DGP.get_sigma_allblocks(corr, y)
            diff_mu_12_mu_11 = DGP.diff_mu_12_mu_11(corr, y)
            diff_mu_13_mu_23 = DGP.diff_mu_13_mu_23(corr, y)
            
            logging.info("Testing ACF Classifier @"+str(mv)+" % missing values")
            acv_svc = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False, timeout=60*30)
            if np.isnan(acv_svc):
                found_error = True
                continue # generate new dataset
            else:
                # append results
                classifier_performance = classifier_performance.append({"Algorithm":"ACF-SVC", "score":acv_svc, "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "points":np.NaN}, ignore_index=True)
        
            for points in pointsalg2:
                logging.info("Testing F-ACF Classifier @"+str(mv)+" % missing values and "+str(points)+" points")
                facv_svc = HPO_utils.characterize_ACF(corr, y, n_iter=n_iter, n_splits=n_splits, test_size=test_size, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant="F-ACF", n_ref=points, return_individual=False, timeout=60*30)
                if np.isnan(acv_svc): # could happen with timeout settings
                    found_error = True
                    break # go to new iteration of while loop and generate new dataset
                else:
                    classifier_performance = classifier_performance.append({"Algorithm":"F-ACF-SVC", "score":facv_svc, "mv":mv, "sigma":sigma, "diff_mu_12_mu_11":diff_mu_12_mu_11, "diff_mu_13_mu_23":diff_mu_13_mu_23, "points":points}, ignore_index=True)
            
                classifier_performance.to_csv("Results/raw/ACFFACFComparison.csv")