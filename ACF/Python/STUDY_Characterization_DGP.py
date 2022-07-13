# -*- coding: utf-8 -*-
"""
Characterization of DGP

@author: Yannis Schumann
"""

import DGP
import numpy as np
import itertools
from scipy.stats import normaltest
import matplotlib.pyplot as plt

# for dissertation, comment out for publication
#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)


def get_Mu_vs_Sigm_vsPGaussian_perBlock(corr, y):
    # filter out biased self-correlations
    corr_new = corr.copy()
    np.fill_diagonal(corr_new, np.NaN)
    
    #unique classes
    classes = np.unique(y)
    # labels for the classes
    lab = ["A","B","C"]
    
    #lists for storing results
    mu_results = [] # mean correlations per block
    sigma_results = [] # standard deviation per block
    p_normal_results = [] # p-values from test per block
    names = [] # legend names for the corresponding plots
    for c1, c2 in itertools.combinations_with_replacement(classes,2):
        # c1 indices
        c1_indices = np.where(y==c1)[0]
        # c2 indices
        c2_indices = np.where(y==c2)[0]
        mu_results.append(np.nanmean(corr_new[np.ix_(c1_indices, c2_indices)]))
        sigma_results.append(np.nanstd(corr_new[np.ix_(c1_indices, c2_indices)]))
        p_normal_results.append(normaltest(corr_new[np.ix_(c1_indices, c2_indices)].ravel(), nan_policy="omit")[1])
        names.append(lab[c1] + ","+lab[c2])
    return mu_results, sigma_results, p_normal_results, names



#------------------------------------------------------------------------------
#                   
#                       IMPACT OF MISSING VALUES
#
#_-----------------------------------------------------------------------------

# Setup
classdistribution = [30,30,30] # three classes with 30 samples each
base_corr = np.array([[1.0,0.9,0.6],[0.9,1.0,0.8],[0.6,0.8,1.0]])
mvs = np.linspace(0.01,0.90,num=20) # range of missing values
cluster_std=2.0 # sigma per feature
repeats = 20 # number of repetitions

# Allocate Memory for results
sigmas = np.zeros((len(mvs),repeats,6))
mus = np.zeros((len(mvs),repeats,6))
ps = np.zeros((len(mvs),repeats,6))

for mv_idx in range(len(mvs)):
    for r_idx in range(repeats):
        X,y,corr = DGP.createDataWithBlockCorrelation(classdistribution, base_corr, mvs[mv_idx], cluster_std, return_corr=True)
        mu_res, sigma_res, p_normal, names = get_Mu_vs_Sigm_vsPGaussian_perBlock(corr, y)
        sigmas[mv_idx,r_idx,:] = sigma_res
        mus[mv_idx,r_idx,:] = mu_res
        ps[mv_idx,r_idx,:] = p_normal

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
axs[0].set_ylabel("Average $\sigma$\nper Block")
for block_idx in range(sigmas.shape[2]):
    axs[0].plot(100*mvs, np.mean(sigmas, axis=1)[:,block_idx], label=names[block_idx])
axs[0].legend(ncol=2)
axs[0].set_yscale("log")
axs[2].set_ylabel("Average p-Val.\n per Block")
for block_idx in range(ps.shape[2]):
    axs[2].plot(100*mvs, np.mean(ps, axis=1)[:,block_idx])
axs[2].hlines(y=0.05, xmin=min(100*mvs),xmax=max(100*mvs), color="black",ls="--", lw=2.0)
axs[2].text(x=20, y=0.08, s="Below: Significant Difference to Normal Distribution", fontsize=9)
axs[1].set_ylabel("Average $\mu$\nper Block")
for block_idx in range(mus.shape[2]):
    axs[1].plot(100*mvs, np.mean(mus, axis=1)[:,block_idx])
axs[2].set_xlabel("Percentage of Missing Values")
fig.suptitle("Impact of Missing Values")
plt.savefig("Results/dgp_missing_values.pdf",bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
#                   
#                       IMPACT OF SIGMA_CLUSTER...
#
#_-----------------------------------------------------------------------------
classdistribution = [30,30,30]
base_corr = np.array([[1.0,0.9,0.6],[0.9,1.0,0.8],[0.6,0.8,1.0]])
cluster_std = np.linspace(0.1,3.0,num=20)
mvs2 = 0.85
repeats = 20

sigmas2 = np.zeros((len(cluster_std),repeats,6))
mus2 = np.zeros((len(cluster_std),repeats,6))
ps2 = np.zeros((len(cluster_std),repeats,6))

for cs_idx in range(len(cluster_std)):
    for r_idx in range(repeats):
        X,y,corr = DGP.createDataWithBlockCorrelation(classdistribution, base_corr, mvs2, cluster_std[cs_idx], return_corr=True)
        mu_res, sigma_res, p_normal, names = get_Mu_vs_Sigm_vsPGaussian_perBlock(corr, y)
        sigmas2[cs_idx,r_idx,:] = sigma_res
        mus2[cs_idx,r_idx,:] = mu_res
        ps2[cs_idx,r_idx,:] = p_normal

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
axs[0].set_ylabel("Average $\sigma$\nper Block")
for block_idx in range(sigmas.shape[2]):
    axs[0].plot(cluster_std, np.mean(sigmas2, axis=1)[:,block_idx], label=names[block_idx])
axs[0].legend(ncol=2)
axs[0].set_yscale("log")
axs[2].set_ylabel("Average p-Val.\n per Block")
for block_idx in range(sigmas.shape[2]):
    axs[2].plot(cluster_std, np.mean(ps2, axis=1)[:,block_idx])
axs[2].hlines(y=0.05, xmin=min(cluster_std),xmax=max(cluster_std), color="black",ls="--", lw=2.0)
axs[2].text(x=1.05, y=0.08, s="Below: Significant Difference to Normal Distribution", fontsize=9)
axs[1].set_ylabel("Average $\mu$\nper Block")
for block_idx in range(mus2.shape[2]):
    axs[1].plot(cluster_std, np.mean(mus2, axis=1)[:,block_idx])
axs[2].set_xlabel("$\sigma_{Features}$")
#axs[2].set_yscale("log")
fig.suptitle(r"Impact of $\sigma_{Features}$")
plt.savefig("Results/dgp_sigmafeatures.pdf",bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
# 2-in-1 figure
fig, axs = plt.subplots(nrows=3, ncols=2, sharey="row", sharex="col")
axs[0,0].set_ylabel("Std. Dev. of \n Correlations")
axs[2,0].set_ylabel("p-Value\n Deviation Normality")
axs[1,0].set_ylabel("Average of \n Correlations")
for block_idx in range(sigmas.shape[2]):
    axs[0,0].plot(cluster_std, np.mean(sigmas2, axis=1)[:,block_idx], label=names[block_idx])
axs[0,0].set_yscale("log")
axs[1,0].set_yscale("log")
for block_idx in range(sigmas.shape[2]):
    axs[0,1].plot(100*mvs, np.mean(sigmas, axis=1)[:,block_idx], label=names[block_idx])
axs[0,1].legend(ncol=2)
for block_idx in range(mus.shape[2]):
    axs[1,0].plot(cluster_std, np.mean(mus2, axis=1)[:,block_idx])
for block_idx in range(mus2.shape[2]):
    axs[1,1].plot(100*mvs, np.mean(mus, axis=1)[:,block_idx])
for block_idx in range(ps.shape[2]):
    axs[2,0].plot(cluster_std, np.mean(ps2, axis=1)[:,block_idx])
axs[2,0].hlines(y=0.05, xmin=min(cluster_std),xmax=max(cluster_std), color="black",ls="--", lw=2.0)
axs[2,0].text(x=2.0, y=0.08, s="p=0.05", fontsize=9)
for block_idx in range(sigmas.shape[2]):
    axs[2,1].plot(100*mvs, np.mean(ps, axis=1)[:,block_idx])
axs[2,1].hlines(y=0.05, xmin=min(100*mvs),xmax=max(100*mvs), color="black",ls="--", lw=2.0)
axs[2,1].text(x=20, y=0.08, s="p=0.05", fontsize=9)
axs[2,1].set_xlabel("Percentage of Missing Values")
axs[2,0].set_xlabel("Standard Deviation per Feature")

#axs[0,0].set_title(r"Impact of $\sigma_{Features}$")
#axs[0,1].set_title(r"Impact of Missing Values")
fig.suptitle("Block-Wise Metrics for" "\n" "the Data Generating Process")

plt.savefig("Results/dgp_characterization.pdf", bbox_inches = 'tight')