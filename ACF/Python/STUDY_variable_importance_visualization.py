# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:29:21 2022

Measure relative variable importance per class on the datasets

@author: Yannis Schumann
"""

import dataset_utils
from sklearn.svm import SVC
import numpy as np
import ACF
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle



#------------------------------------------------------------------------------
#                   
#                       Setup
#
#------------------------------------------------------------------------------
baseline_classifier = SVC(C=1e2, kernel="rbf", class_weight="balanced")
n_splits=5
repeats = 2

compute = True
plot = False

#------------------------------------------------------------------------------
#                   
#                   Utility Functions
#
#------------------------------------------------------------------------------

def precision_recall_f1score_variable_n_class_m(model, corr, y, variable_n, class_m, n_splits=5):
    # if variable_n==None, we report the metrics for acf, if all variables are included
    
    # we use the ACF-class only to compute the correlations in the correct way
    acf = ACF.ACFClassifier(baseline_classifier=DummyClassifier(), precomputed = True)
    # stratified k-fold for evaluation
    skf = StratifiedKFold(n_splits=n_splits)
    precision = []
    recall = []
    f1scores = []
    for train_index, test_index in skf.split(corr, y):
        corr_train = corr[np.ix_(train_index, train_index)]
        corr_test = corr[np.ix_(test_index, train_index)]
        y_train, y_test = y[train_index], y[test_index]
        acf.fit(corr_train, y_train)
        acf.predict(corr_test)
        if variable_n!=None:
            class_index = np.nonzero(acf.classes_==variable_n)[0]
            model.fit(np.delete(acf.training_features, class_index, axis=1), y_train)
            predictions_test = model.predict(np.delete(acf.test_features, class_index, axis=1))
        else:
            model.fit(acf.training_features, y_train)
            predictions_test = model.predict(acf.test_features)
        precision.append(precision_score(y_test, predictions_test, labels=acf.classes_, average=None, zero_division=0)[np.nonzero(acf.classes_==class_m)[0]])
        recall.append(recall_score(y_test, predictions_test, labels=acf.classes_, average=None, zero_division=0)[np.nonzero(acf.classes_==class_m)[0]])
        f1scores.append(f1_score(y_test, predictions_test, labels=acf.classes_, average=None, zero_division=0)[np.nonzero(acf.classes_==class_m)[0]])
        
    return np.array(precision).ravel(), np.array(recall).ravel(), np.array(f1scores).ravel()


def variable_importance(model, corr, y, n_splits=5, repeats=2):
    classes = np.unique(y)
    class_list = []
    variable_list = []
    precision_list = []
    recall_list = []
    f1_list = []            
    for rep in range(repeats):
        for c, v in product(classes, np.hstack([classes, [None]])):
            print("Repetition {}/{},\tClass:{}\t\t Variable:{}".format(rep+1, repeats, c,v))
            p, r, f = precision_recall_f1score_variable_n_class_m(model, corr, y, variable_n=v, class_m=c, n_splits=n_splits)
            precision_list.extend(list(p))
            recall_list.extend(list(r))
            class_list.extend([c for _ in p])
            variable_list.extend([v for _ in p])
            f1_list.extend(list(f))
    df = pd.DataFrame({"Class":class_list, "Left-Out Correlation":variable_list, "Precision":precision_list, "Recall":recall_list, "F1 Score":f1_list})
    for cl in df["Class"].unique():
        df.loc[df[df["Class"]==cl].index.values, "Precision"] = np.mean(df[(df["Class"]==cl) & (~pd.notnull(df["Left-Out Correlation"]))]["Precision"].values) - df.loc[df[df["Class"]==cl].index.values, "Precision"]
        df.loc[df[df["Class"]==cl].index.values, "Recall"] = np.mean(df[(df["Class"]==cl) & (~pd.notnull(df["Left-Out Correlation"]))]["Recall"].values)- df.loc[df[df["Class"]==cl].index.values, "Recall"]
        df.loc[df[df["Class"]==cl].index.values, "F1 Score"] = np.mean(df[(df["Class"]==cl) & (~pd.notnull(df["Left-Out Correlation"]))]["F1 Score"].values)- df.loc[df[df["Class"]==cl].index.values, "F1 Score"]
    df = df[pd.notnull(df["Left-Out Correlation"])]
    
    return df


def plot_variable_importance(df, filename, n_highlight=4):
    classes = df["Class"].unique()
    fig, axs = plt.subplots(nrows=len(classes), ncols=1, sharex=True)
    for cl_idx in range(len(classes)):
        c = classes[cl_idx]
        sns.barplot(data=df[df["Class"]==c], x="Left-Out Correlation", y="F1 Score", ax=axs[cl_idx])
        axs[cl_idx].set_title("Feature Importance for Class "+str(c))
        axs[cl_idx].set_xlabel(None)
        axs[cl_idx].axhline(y=0, color="black", linestyle="--")
    axs[-1].set_xlabel("Left-Out Correlation")
    plt.tight_layout()
    plt.savefig("Supplements2/plots/"+filename+"_bar.pdf", bbox_inches = 'tight')
    plt.show()
    
    df = df.pivot_table(index="Class", columns="Left-Out Correlation", values="F1 Score", aggfunc=lambda x: round(np.mean(x),2))
    fig, axs = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(df, annot=True, ax=axs, fmt=".2g")
    # highlight the n highest off-diagonal elements
    df2 = df.copy()
    np.fill_diagonal(df2.values, -np.inf)
    for _ in range(n_highlight):
        ax0_idx, ax1_idx = np.unravel_index(df2.values.argmax(), df2.shape)
        df2.values[ax0_idx, ax1_idx] = -np.inf
        axs.add_patch(Rectangle((ax1_idx, ax0_idx), 1, 1, fill=False, edgecolor='blue', lw=3))
    
    plt.savefig("Supplements2/plots/"+filename+"_heatmap.pdf", bbox_inches = 'tight')
    plt.show()
    
    

#------------------------------------------------------------------------------
#                   
#                   Variable Importance
#
#------------------------------------------------------------------------------

# choosing dataset by Xin et al, because the number of classes is limited there


#show_distribution_correlations(corr, data_meta)

if compute:
    data, data_meta, corr = dataset_utils.get_raw_XinPancreas(recompute_corr = False)
    xin_importance = variable_importance(baseline_classifier, corr, data_meta, n_splits=n_splits, repeats=repeats)
    data, data_meta, corr = dataset_utils.get_raw_3kpbmc(recompute_corr = False)
    pbmc_importance = variable_importance(baseline_classifier, corr, data_meta, n_splits=n_splits, repeats=repeats)
    data, data_meta, corr = dataset_utils.get_raw_BaronPancreas(recompute_corr = False)
    baron_importance = variable_importance(baseline_classifier, corr, data_meta, n_splits=n_splits, repeats=repeats)
    data, data_meta, corr = dataset_utils.get_IRS_Petralia(recompute_corr=False)
    petralia_importance = variable_importance(baseline_classifier, corr, data_meta, n_splits=n_splits, repeats=repeats)
    data, data_meta, corr = dataset_utils.get_irs_brcapam50(recompute_corr=False)
    brca_importance = variable_importance(baseline_classifier, corr, data_meta, n_splits=n_splits, repeats=repeats)
    
    xin_importance.to_csv("Results/raw/xin_acf_feature_importance.csv")
    pbmc_importance.to_csv("Results/raw/3kpbmc_acf_feature_importance.csv")
    baron_importance.to_csv("Results/raw/baron_acf_feature_importance.csv")
    petralia_importance.to_csv("Results/raw/petralia_acf_feature_importance.csv")
    brca_importance.to_csv("Results/raw/brca_acf_feature_importance.csv")
if plot:
    xin_importance = pd.read_csv("Results/raw/xin_acf_feature_importance.csv", index_col=0)
    pbmc_importance = pd.read_csv("Results/raw/3kpbmc_acf_feature_importance.csv", index_col=0)
    baron_importance = pd.read_csv("Results/raw/baron_acf_feature_importance.csv", index_col=0)
    petralia_importance = pd.read_csv("Results/raw/petralia_acf_feature_importance.csv", index_col=0)
    brca_importance =pd.read_csv("Results/raw/brca_acf_feature_importance.csv", index_col=0)
    plot_variable_importance(xin_importance, "xin_var_importance")
    plot_variable_importance(pbmc_importance, "3kpbmc_var_importance")
    plot_variable_importance(baron_importance, "baron_var_importance")
    plot_variable_importance(petralia_importance, "petralia_var_importance")
    plot_variable_importance(brca_importance, "brca_var_importance")