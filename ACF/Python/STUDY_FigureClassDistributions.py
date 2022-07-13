# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:03:39 2022

@author: Yannis Schumann
"""

import dataset_utils
import numpy as np
import matplotlib.pyplot as plt

X_pbmc, y_pbmc, _ = dataset_utils.get_raw_3kpbmc(False)
l_pbmc, c_pbmc = np.unique(y_pbmc, return_counts=True)
X_xin, y_xin, _ = dataset_utils.get_raw_XinPancreas(False)
l_xin, c_xin = np.unique(y_xin, return_counts=True)
X_baron, y_baron, _ = dataset_utils.get_raw_BaronPancreas(False)
l_baron, c_baron = np.unique(y_baron, return_counts=True)

fig = plt.gcf()
gs = fig.add_gridspec(4, 4)
ax = fig.add_subplot(gs[0:2, 1:3])
ax.pie(c_pbmc, labels=["Class "+str(l_pbmc[i]) +" ("+str(round(100*c_pbmc[i]/np.sum(c_pbmc), 1))+ "%)" for i in range(len(l_pbmc))])
ax.set_title("10X Genomics")
ax = fig.add_subplot(gs[2:4, 0:2])
ax.pie(c_xin, labels=[str(l_xin[i]) +" ("+str(round(100*c_xin[i]/np.sum(c_xin), 1))+ "%)" for i in range(len(l_xin))])
ax.set_title("Xin et al")
ax = fig.add_subplot(gs[2:4, 2:4])
ax.pie(c_baron, labels=[str(l_baron[i]) +" ("+str(round(100*c_baron[i]/np.sum(c_baron), 1))+ "%)" for i in range(len(l_baron))])
ax.set_title("Baron et al")
fig.tight_layout()
fig.suptitle("Class Distribution of the Considered scRNA-seq Datasets")
#fig.tight_layout()
plt.savefig("Results/classdistribution_scrnaseq.pdf", bbox_inches = 'tight')

X_petralia, y_petralia, _ = dataset_utils.get_IRS_Petralia(False)
l_petralia, c_petralia = np.unique(y_petralia, return_counts=True)
X_brca, y_brca, _ = dataset_utils.get_irs_brcapam50(False)
l_brca, c_brca = np.unique(y_brca, return_counts=True)

fig, axs = plt.subplots(ncols=2)
axs[0].pie(c_petralia, labels=[str(l_petralia[i]) +" ("+str(round(100*c_petralia[i]/np.sum(c_petralia), 1))+ "%)" for i in range(len(l_petralia))])
axs[0].set_title("Petralia et al")
axs[1].pie(c_brca, labels=[str(l_brca[i]) +" ("+str(round(100*c_brca[i]/np.sum(c_brca), 1))+ "%)" for i in range(len(l_brca))])
axs[1].set_title("Krug et al")
#fig.tight_layout()
fig.suptitle("Class Distribution of the Considered Proteomic Datasets")
#fig.tight_layout()
plt.savefig("Results/classdistribution_proteomics.pdf", bbox_inches = 'tight')