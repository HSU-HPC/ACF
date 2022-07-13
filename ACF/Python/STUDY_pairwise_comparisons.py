# -*- coding: utf-8 -*-
"""
Perform the pairwise comparison

@author: Yannis Schumann
"""
import pandas as pd
import numpy as np
import stat_utils

#------------------------------------------------------------------------------
#                   
#                               Utilities
#
#------------------------------------------------------------------------------
def long_to_wide_df(df):
    algorithms = df["Algorithm"].unique()
    wide_df = pd.DataFrame({alg:df.loc[df["Algorithm"]==alg, "Value"].values for alg in algorithms})
    return wide_df

def rename_algorithms(df):
    df.loc[df["Algorithm"]=="Ridge", "Algorithm"] = "Listwise Deletion + Ridge"
    df.loc[df["Algorithm"]=="RF", "Algorithm"] = "Listwise Deletion + RF"
    df.loc[df["Algorithm"]=="SVC", "Algorithm"] = "Listwise Deletion + SVC"
    df.loc[df["Algorithm"]=="ACF+Ridge", "Algorithm"] = "ACF + Ridge"
    df.loc[df["Algorithm"]=="ACF+RF", "Algorithm"] = "ACF + RF"
    df.loc[df["Algorithm"]=="ACF+SVC", "Algorithm"] = "ACF + SVC"
    return df

#------------------------------------------------------------------------------
#                   
#                               scRNA-seq
#
#------------------------------------------------------------------------------

scRNA_data = pd.read_csv("Results/raw/full-comp-scRNAseq.csv", index_col=0)

# select respective datasets
scRNA_data_baron = scRNA_data[scRNA_data["Dataset"]=="Baron"]
scRNA_data_baron = rename_algorithms(scRNA_data_baron)
scRNA_data_baron = long_to_wide_df(scRNA_data_baron)

scRNA_data_xin = scRNA_data[scRNA_data["Dataset"]=="Xin"]
scRNA_data_xin = rename_algorithms(scRNA_data_xin)
scRNA_data_xin = long_to_wide_df(scRNA_data_xin)

scRNA_data_pbmc = scRNA_data[scRNA_data["Dataset"]=="3kPBMC"]
scRNA_data_pbmc = rename_algorithms(scRNA_data_pbmc)
scRNA_data_pbmc = long_to_wide_df(scRNA_data_pbmc)


# order by performance of classifier
scRNA_data_baron = scRNA_data_baron.reindex(scRNA_data_baron.mean().sort_values(ascending=False).index, axis=1)
scRNA_data_xin = scRNA_data_xin.reindex(scRNA_data_xin.mean().sort_values(ascending=False).index, axis=1)
scRNA_data_pbmc = scRNA_data_pbmc.reindex(scRNA_data_pbmc.mean().sort_values(ascending=False).index, axis=1)

# significance of classifiers

baron_table = stat_utils.compute_pairwise_tests(scRNA_data_baron, 0.05) # alpha of 0.05
xin_table = stat_utils.compute_pairwise_tests(scRNA_data_xin, 0.05) # alpha of 0.05
pbmc_table = stat_utils.compute_pairwise_tests(scRNA_data_pbmc, 0.05) # alpha of 0.05


#------------------------------------------------------------------------------
#                   
#                               Proteomics
#
#------------------------------------------------------------------------------

proteomic_data = pd.read_csv("Results/raw/full-comp-proteomics.csv", index_col=0)


# select respective datasets
petralia_raw_data = proteomic_data[proteomic_data["Dataset"]=="Petralia(raw)"]
petralia_raw_data = rename_algorithms(petralia_raw_data)
petralia_raw_data = long_to_wide_df(petralia_raw_data)

petralia_irs_data = proteomic_data[proteomic_data["Dataset"]=="Petralia(IRS)"]
petralia_irs_data = rename_algorithms(petralia_irs_data)
petralia_irs_data = long_to_wide_df(petralia_irs_data)

brca_raw_data = proteomic_data[proteomic_data["Dataset"]=="BRCA(raw)"]
brca_raw_data = rename_algorithms(brca_raw_data)
brca_raw_data = long_to_wide_df(brca_raw_data)

brca_irs_data = proteomic_data[proteomic_data["Dataset"]=="BRCA(IRS)"]
brca_irs_data = rename_algorithms(brca_irs_data)
brca_irs_data = long_to_wide_df(brca_irs_data)


# order by performance of classifier
petralia_raw_data = petralia_raw_data.reindex(petralia_raw_data.mean().sort_values(ascending=False).index, axis=1)
petralia_irs_data = petralia_irs_data.reindex(petralia_irs_data.mean().sort_values(ascending=False).index, axis=1)
brca_raw_data = brca_raw_data.reindex(brca_raw_data.mean().sort_values(ascending=False).index, axis=1)
brca_irs_data = brca_irs_data.reindex(brca_irs_data.mean().sort_values(ascending=False).index, axis=1)

# significance of classifiers

petraliaraw_table = stat_utils.compute_pairwise_tests(petralia_raw_data, 0.05) # alpha of 0.05
petraliairs_table = stat_utils.compute_pairwise_tests(petralia_irs_data, 0.05) # alpha of 0.05
brcaraw_table = stat_utils.compute_pairwise_tests(brca_raw_data, 0.05) # alpha of 0.05
brcairs_table = stat_utils.compute_pairwise_tests(brca_irs_data, 0.05) # alpha of 0.05