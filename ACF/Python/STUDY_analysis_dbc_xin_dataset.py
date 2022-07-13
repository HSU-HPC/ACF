# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:28:25 2022

@author: Yannis Schumann
"""
import DBC
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import dataset_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report


#------------------------------------------------------------------------------
#                   
#                       Setup
#
#------------------------------------------------------------------------------
n_splits=5

#------------------------------------------------------------------------------
#                   
#                   Utility Functions
#
#------------------------------------------------------------------------------


def show_distribution_correlations(corr, y):
    oc = OrdinalEncoder()
    y = oc.fit_transform(y.reshape(-1, 1)).ravel()
    dbc = DBC.DBCClassifier(precomputed=True)
    dbc.fit(corr, y)
    
    fig, axs = plt.subplots(nrows=len(dbc.classes_), ncols=1, sharex=True)
    one_black_label = False
    one_red_label = False
    for cl_idx in range(len(dbc.classes_)):
        for cl2_idx in range(len(dbc.classes_)):
            if cl_idx==cl2_idx:
                color="red"
                if one_red_label:
                    label=None
                else:
                    label="Intra-Class Distribution"
                    one_red_label=True
            else:
                color="black"
                if one_black_label:
                    label=None
                else:
                    label="Inter-Class Distribution"
                    one_black_label=True
            axs[cl_idx].plot(dbc.possible_values, dbc.distribution_matrix[cl_idx, cl2_idx]/np.sum(dbc.distribution_matrix[cl_idx, cl2_idx]), color=color, label=label)
        axs[cl_idx].set_title("Distribution of Correlations, Class " + str(oc.inverse_transform(np.array(dbc.classes_[cl_idx]).reshape(1,-1)).ravel()[0]))
        axs[cl_idx].set_ylabel("Density")
    axs[0].legend()
    axs[0].set_xlim(0,1)
    axs[-1].set_xlabel("Correlation")
    fig.suptitle("Kernel Density Estimate of Correlations" "\n" "Xin et al")
    fig.tight_layout()
    plt.savefig("Supplements2/plots/xin_correlation_distribution.pdf", bbox_inches = 'tight')
    plt.show()
    

def get_classification_report_DBC(corr, y, n_splits=5):
    oc = OrdinalEncoder()
    y = oc.fit_transform(y.reshape(-1, 1)).ravel()
    
    true_values = []
    predicted_values = []
    
    skf = StratifiedKFold(n_splits=n_splits)
    
    dbc = DBC.DBCClassifier(precomputed=True)
    
    for train_index, test_index in skf.split(corr, y):
        corr_train = corr[np.ix_(train_index, train_index)]
        corr_test = corr[np.ix_(test_index, train_index)]
        y_train, y_test = y[train_index], y[test_index]
        
        dbc.fit(corr_train, y_train)
        predicted_values.extend(list(dbc.predict(corr_test)))
        true_values.extend(list(y_test))
    
    print(classification_report(true_values, predicted_values))
    
    return true_values, predicted_values
    

#------------------------------------------------------------------------------
#                   
#                   Analysis
#
#------------------------------------------------------------------------------

data, data_meta, corr = dataset_utils.get_raw_XinPancreas(recompute_corr = False)
show_distribution_correlations(corr, data_meta)
t, p = get_classification_report_DBC(corr, data_meta, n_splits=n_splits)