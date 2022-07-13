# -*- coding: utf-8 -*-
"""
Utility functions for statistical tests

@author: Yannis Schumann
"""
import numpy as np
from scipy.stats import t
from math import factorial
import pandas as pd
import itertools


def corrected_t(differences, n_train, n_test):
    """
    Computes and corrects the corrected t statistic for the differences.

    Parameters
    ----------
    differences : list-like
        List-like of differences between the scores of two models
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the test set.

    Returns
    -------
    t : float
        The corrected t statistic.

    """
    corrected_std = np.sqrt((1/len(differences)+n_test/n_train)*np.var(differences, ddof=1))
    mean_diff = np.mean(differences)
    
    t = mean_diff/corrected_std
    
    return t


def compute_Bonferroni_Corrected_p(differences, n_comparison):
    """
    Computes the Bonferroni.corrected p-value for the right-tailed paired t test

    Parameters
    ----------
    differences : list-like
        The differences between the model.
    n_comparison : int
        The number of pairwise comparisons performed.

    Returns
    -------
    p_stat : float
        The corrected p-value.

    """
    # compute degrees of freedom
    deg = len(differences)-1 # number of test sets -1
    # compute t statistic --> correction for correlated splits is NOT required, since we draw them independently for each model.
    t_stat = np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / len(differences))
    p_stat = t.sf(np.abs(t_stat), deg) # right tailed t-test
    
    # apply Bonferroni correction
    p_stat = p_stat*n_comparison
    # p values must not be larger than 1
    if p_stat>1.0:
        p_stat = 1.0
    return p_stat


def compute_pairwise_tests(df, levelofsignificance=0.05):
    """
    Computes and returns a dataframe with p-values for a corrected, right-tailed, paired t-test

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the scores on the individual columns. No other column must exist.
    levelofsignificance : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    result : pd.DataFrame

    """
    # number of models that we are comparing
    model_names = df.columns.values
    n_models = len(model_names)
    # number of comparisons
    n_comparisons = factorial(n_models)/factorial(2)/factorial(n_models-2)
    
    result = pd.DataFrame(columns=["Model 1", "Model 2", "Corrected p-Value", "Significant"])
    
    for model1, model2 in itertools.combinations(model_names, 2): # no combinations
        differences = df[model1].values-df[model2].values
        p = compute_Bonferroni_Corrected_p(differences, n_comparisons)
        sig = p<levelofsignificance
        result = result.append({"Model 1":model1, "Model 2":model2, "Corrected p-Value":p, "Significant":sig}, ignore_index=True)
    return result