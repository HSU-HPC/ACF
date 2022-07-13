# -*- coding: utf-8 -*-
"""
Utility functions for loading the datasets

@author: Yannis Schumann
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import IRS


def get_raw_3kpbmc(recompute_corr=True):
    """
    Loads the 3kPBMC dataset and computes the pairwise spearman correlations.

    Parameters
    ----------
    recompute_corr : bool, optional
        Whether to recompute the correlation matrix. If false, it will be loaded from the file. The default is True.

    Returns
    -------
    data : pd.DataFrame
        The original data of size n_train x n_genes.
    data_meta : pd.DataFrame
        Corresponding class assignments (in the same order ! ).
    corr : pd.DataFrame
        The matrix of pairwise correlations.

    """
    data = pd.read_csv("Datasets/scRNAseq/SeuratPBMC/filtered_umi.csv",index_col=0)
    data = data.transpose()
    data_meta = pd.read_csv("Datasets/scRNAseq/SeuratPBMC/celltypes.csv", index_col=0)
    data_meta.rename(columns={"x":"Type"}, inplace=True)
    # some cells were removed from clustering due to bad quality metrics.
    data = data.loc[data_meta.index.values] #take only clustered (high-quality) cells
    if recompute_corr:
        # pandas computes correlations between columns, so we need to transpose
        corr = data.transpose().corr(method="spearman")
        corr.to_csv("Datasets/scRNAseq/SeuratPBMC/corrmatrix.csv", na_rep="NaN")
    else:
        corr = pd.read_csv("Datasets/scRNAseq/SeuratPBMC/corrmatrix.csv", index_col=0, na_values="NaN")
    return data.values, data_meta.values.ravel(), corr.values


def get_raw_XinPancreas(recompute_corr=True):
    """
    Load the dataset by Xin et al with pancreatic cells.

    Parameters
    ----------
    recompute_corr : bool, optional
        Whether to recompute the correlation matrix. If false, it will be loaded from the file. The default is True.

    Returns
    -------
    data : pd.DataFrame
        The original data of size n_train x n_genes.
    data_meta : pd.DataFrame
        Corresponding class assignments (in the same order ! ).
    corr : pd.DataFrame
        The matrix of pairwise correlations.

    """
    # load raw
    data = pd.read_csv("Datasets/scRNAseq/XinPancreas/xinpancreasrpkm.csv",index_col=0)
    data_meta = pd.read_csv("Datasets/scRNAseq/XinPancreas/metadata.csv", index_col=0)
    
    # need to be transposed --> instances in axis 0, genes on axis 1
    data = data.transpose()
    
    # exclude all contaminated cells
    data_meta = data_meta[data_meta["cell.type"].isin(["alpha","beta","PP", "delta"])]
    # reduce to neccessary information
    data_meta = data_meta["cell.type"]
    # same indices on raw data
    data = data.loc[data_meta.index.values]
    
    if recompute_corr:
        # pandas computes correlations between columns, so we need to transpose
        corr = data.transpose().corr(method="spearman")
        corr.to_csv("Datasets/scRNAseq/XinPancreas/corrmatrix.csv", na_rep="NaN")
    else:
        corr = pd.read_csv("Datasets/scRNAseq/XinPancreas/corrmatrix.csv", index_col=0, na_values="NaN")
    return data.values, data_meta.values.ravel(), corr.values


def get_raw_BaronPancreas(recompute_corr=True):
    """
    Load the dataset by Baron et al with pancreatic cells.

    Parameters
    ----------
    recompute_corr : bool, optional
        Whether to recompute the correlation matrix. If false, it will be loaded from the file. The default is True.

    Returns
    -------
    data : pd.DataFrame
        The original data of size n_train x n_genes.
    data_meta : pd.DataFrame
        Corresponding class assignments (in the same order ! ).
    corr : pd.DataFrame
        The matrix of pairwise correlations.

    """
    # load raw data and transpose for computing correlation matrix
    data = pd.read_csv("Datasets/scRNAseq/BaronPancreas/baronpancreascounts.csv",index_col=0)
    data = data.transpose()
    
    # load class assignments
    data_meta = pd.read_csv("Datasets/scRNAseq/BaronPancreas/metadata.csv", index_col=0)
    #reduce metadata to class column
    #data_meta = data_meta["label"]
    
    #exclude minority celltypes
    celltypes = data_meta["label"].unique() # all cell types
    new_celltypes = [c for c in celltypes if data_meta[data_meta["label"]==c].shape[0]/data_meta.shape[0]>=0.01] # only cells with at least 1% relative abundance
    data_meta = data_meta[data_meta["label"].isin(new_celltypes)] # reduce to the new celltypes
    
    # bring to same order and reduce to new cell types
    data = data.loc[data_meta.index.values]
    
    # reduce metadata to class labels
    data_meta = data_meta["label"]
    
    # for computational performance, select only a stratified subset of the dataset
    # select random state for reproducibility
    X, _, y, _ = train_test_split(data.values,data_meta.values,stratify=data_meta.values, train_size=0.4, random_state=0)
    
    # compute or load the correlations
    if recompute_corr:
        # pandas computes correlations between columns, so we need to transpose
        corr = pd.DataFrame(X).transpose().corr(method="spearman")
        corr.to_csv("Datasets/scRNAseq/BaronPancreas/corrmatrix.csv", na_rep="NaN")
    else:
        corr = pd.read_csv("Datasets/scRNAseq/BaronPancreas/corrmatrix.csv", index_col=0, na_values="NaN")
    
    return X, y, corr.values


def get_raw_Petralia(recompute_corr = True):
    """
    Loads the Petralia dataset (without batch correction) and computes the 
    correlation matrix. Correlations between samples from same batch are masked to np.NaN,
    because we expect them to be biased.

    Parameters
    ----------
    recompute_corr : bool, optional
        Whether to recompute the correlation matrix. If false, it will be loaded from the file. The default is True.

    Returns
    -------
    data : ndarray
        The original data of size n_train x n_genes.
    data_meta : ndarray
        Corresponding class assignments (in the same order ! ).
    corr : ndarray
        The matrix of pairwise correlations.

    """
    # read raw input
    petralia = pd.read_csv("Datasets/Proteomics/Petralia/PetraliaRawNicelyFormatted.csv", index_col=0, na_values=["0","0.0","NA","NaN"])
    #drop duplicates and unneeded columns
    petralia = petralia.drop(labels=['7316-121.1', '7316-121.2', '7316-121.3', '7316-121.4', 'GeneName'], axis=1)
    # transpose, since we want to have features in columns
    petralia = petralia.transpose()
    # read file with batch per sample
    datadescription = pd.read_csv("Datasets/Proteomics/Petralia/petraliabatchdescriptionyannis.csv", index_col=0, na_values=["0","0.0","NA"])
    # read clinical descriptions (proteom-subgroups)
    clinicaldescription = pd.read_csv("Datasets/Proteomics/Petralia/clinicaldescription_control.csv", index_col=0)
    clinicaldescription.drop(columns="diagnosis_type", inplace=True) # we don't disciminate between initial/recurrent CNS tumors
    # indices of bridges
    bridge_labels = [x for x in datadescription.index.values if "Bridge" in x]
    bridge_entries = pd.DataFrame([["Bridge"] for x in bridge_labels], columns=['proteomcluster'], index=bridge_labels) # metadata for the bridges
    metadata = clinicaldescription.append(bridge_entries, verify_integrity=True) # composed metadata
    metadata.rename({"proteomcluster":"Type"}, axis=1, inplace=True)
    # add batch column
    metadata = pd.concat([metadata, datadescription], axis=1, join="inner")
    # bring to same order
    petralia = petralia.loc[metadata.index.values]
    
    # we don't want the bridges
    data, data_meta = IRS.drop_BridgeSamples(petralia, metadata)
    
    if recompute_corr:
        corr = data.transpose().corr(method="spearman")
        corr.to_csv("Datasets/Proteomics/Petralia/corrmatrix_raw.csv", na_rep="NaN")
    else:
        corr = pd.read_csv("Datasets/Proteomics/Petralia/corrmatrix_raw.csv", index_col=0, na_values="NaN")
    
    
    # mask the biased correlations
    masked_corr = IRS.mask_biased_correlations(corr, data_meta)
    
    return data.values, data_meta["Type"].values.ravel(), corr.values, masked_corr.values
    


def get_IRS_Petralia_withCovariable(recompute_corr = True):
    """
    Loads the Petralia dataset, corrects for batch effects using the internal
    references and computes the 
    correlation matrix. Additionally, we also return the covariables

    Parameters
    ----------
    recompute_corr : bool, optional
        Whether to recompute the correlation matrix. If false, it will be loaded from the file. The default is True.

    Returns
    -------
    data : ndarray
        The original data of size n_train x n_genes.
    data_meta : ndarray
        Corresponding class assignments (in the same order ! ).
    corr : ndarray
        The matrix of pairwise correlations.

    """
    # read raw input
    petralia = pd.read_csv("Datasets/Proteomics/Petralia/PetraliaRawNicelyFormatted.csv", index_col=0, na_values=["0","0.0","NA","NaN"])
    #drop duplicates and unneeded columns
    petralia = petralia.drop(labels=['7316-121.1', '7316-121.2', '7316-121.3', '7316-121.4', 'GeneName'], axis=1)
    # transpose, since we want to have features in columns
    petralia = petralia.transpose()
    # read file with batch per sample
    datadescription = pd.read_csv("Datasets/Proteomics/Petralia/petraliabatchdescriptionyannis.csv", index_col=0, na_values=["0","0.0","NA"])
    # read clinical descriptions (proteom-subgroups)
    clinicaldescription = pd.read_csv("Datasets/Proteomics/Petralia/clinicaldescription_control.csv", index_col=0)
    clinicaldescription.drop(columns="diagnosis_type", inplace=True) # we don't disciminate between initial/recurrent CNS tumors
    
    # indices of bridges
    bridge_labels = [x for x in datadescription.index.values if "Bridge" in x]
    bridge_entries = pd.DataFrame([["Bridge"] for x in bridge_labels], columns=['proteomcluster'], index=bridge_labels) # metadata for the bridges
    metadata = clinicaldescription.append(bridge_entries, verify_integrity=True) # composed metadata
    metadata.rename({"proteomcluster":"Type"}, axis=1, inplace=True)
    # add batch column
    metadata = pd.concat([metadata, datadescription], axis=1, join="inner")
    # bring to same order
    petralia = petralia.loc[metadata.index.values]
    
    # correct for differential loading
    petralia = IRS.sample_loading_adjustment(petralia, metadata)
    # IRS
    petralia = IRS.IRS(petralia, metadata)
    # we don't want the bridges anymore
    data, data_meta = IRS.drop_BridgeSamples(petralia, metadata)
    # just to be sure that they are ordered
    data = data.loc[data_meta.index.values]
    
    # read table with covariables
    covariabledescription = pd.read_csv("Datasets/Proteomics/Petralia/SupplementaryCovariables.csv", index_col=2)
    covariabledescription = covariabledescription.loc[data_meta.index.values]
    
    considered_cv = covariabledescription[["age.class", "tumor_location_condensed", "gender"]]
    
    # exclude samples that have NA in any of the covariables
    considered_cv = considered_cv[~np.any(considered_cv.isna(), axis=1)]
    data_meta = data_meta.loc[considered_cv.index.values]
    data = data.loc[considered_cv.index.values]
    
    considered_cv = OrdinalEncoder().fit_transform(considered_cv)
    
    if recompute_corr:
        corr = data.transpose().corr(method="spearman")
        corr.to_csv("Datasets/Proteomics/Petralia/corrmatrix_irs_cv.csv", na_rep="NaN")
    else:
        corr = pd.read_csv("Datasets/Proteomics/Petralia/corrmatrix_irs_cv.csv", index_col=0, na_values="NaN")
    
    return data.values, data_meta["Type"].values.ravel(), corr.values, considered_cv



def get_IRS_Petralia(recompute_corr = True):
    """
    Loads the Petralia dataset, corrects for batch effects using the internal
    references and computes the 
    correlation matrix. 

    Parameters
    ----------
    recompute_corr : bool, optional
        Whether to recompute the correlation matrix. If false, it will be loaded from the file. The default is True.

    Returns
    -------
    data : ndarray
        The original data of size n_train x n_genes.
    data_meta : ndarray
        Corresponding class assignments (in the same order ! ).
    corr : ndarray
        The matrix of pairwise correlations.

    """
    # read raw input
    petralia = pd.read_csv("Datasets/Proteomics/Petralia/PetraliaRawNicelyFormatted.csv", index_col=0, na_values=["0","0.0","NA","NaN"])
    #drop duplicates and unneeded columns
    petralia = petralia.drop(labels=['7316-121.1', '7316-121.2', '7316-121.3', '7316-121.4', 'GeneName'], axis=1)
    # transpose, since we want to have features in columns
    petralia = petralia.transpose()
    # read file with batch per sample
    datadescription = pd.read_csv("Datasets/Proteomics/Petralia/petraliabatchdescriptionyannis.csv", index_col=0, na_values=["0","0.0","NA"])
    # read clinical descriptions (proteom-subgroups)
    clinicaldescription = pd.read_csv("Datasets/Proteomics/Petralia/clinicaldescription_control.csv", index_col=0)
    clinicaldescription.drop(columns="diagnosis_type", inplace=True) # we don't disciminate between initial/recurrent CNS tumors
    # indices of bridges
    bridge_labels = [x for x in datadescription.index.values if "Bridge" in x]
    bridge_entries = pd.DataFrame([["Bridge"] for x in bridge_labels], columns=['proteomcluster'], index=bridge_labels) # metadata for the bridges
    metadata = clinicaldescription.append(bridge_entries, verify_integrity=True) # composed metadata
    metadata.rename({"proteomcluster":"Type"}, axis=1, inplace=True)
    # add batch column
    metadata = pd.concat([metadata, datadescription], axis=1, join="inner")
    # bring to same order
    petralia = petralia.loc[metadata.index.values]
    
    # correct for differential loading
    petralia = IRS.sample_loading_adjustment(petralia, metadata)
    # IRS
    petralia = IRS.IRS(petralia, metadata)
    # we don't want the bridges anymore
    data, data_meta = IRS.drop_BridgeSamples(petralia, metadata)
    # just to be sure that they are ordered
    data = data.loc[data_meta.index.values]
    
    if recompute_corr:
        corr = data.transpose().corr(method="spearman")
        corr.to_csv("Datasets/Proteomics/Petralia/corrmatrix_irs.csv", na_rep="NaN")
    else:
        corr = pd.read_csv("Datasets/Proteomics/Petralia/corrmatrix_irs.csv", index_col=0, na_values="NaN")
    
    return data.values, data_meta["Type"].values.ravel(), corr.values


def get_raw_brcapam50(recompute_corr=True):
    """
    Loads the BRCA dataset (without batch correction) and computes the 
    correlation matrix. Correlations between samples from same batch are masked to np.NaN,
    because we expect them to be biased.

    Parameters
    ----------
    recompute_corr : bool, optional
        Whether to recompute the correlation matrix. If false, it will be loaded from the file. The default is True.

    Returns
    -------
    data : ndarray
        The original data of size n_train x n_genes.
    data_meta : ndarray
        Corresponding class assignments (in the same order ! ).
    corr : ndarray
        The matrix of pairwise correlations.

    """
    
    brca = pd.read_csv("Datasets/Proteomics/BRCA/joined_datasets_pam50.csv", index_col=0, na_values=[0,0.0,np.NaN,"NAN"])
    metadata = pd.read_csv("Datasets/Proteomics/BRCA/metadata_pam50.csv", index_col=0)
    
    # drop Bridges
    data, data_meta = IRS.drop_BridgeSamples(brca, metadata)
    
    # ensure that they are in the same order
    data = data.loc[data_meta.index.values]
    
    if recompute_corr:
        corr = data.transpose().corr(method="spearman")
        corr.to_csv("Datasets/Proteomics/BRCA/corrmatrix_raw.csv", na_rep="NaN")
    else:
        corr = pd.read_csv("Datasets/Proteomics/BRCA/corrmatrix_raw.csv", index_col=0, na_values="NaN")
    
    masked_corr = IRS.mask_biased_correlations(corr, data_meta)
    
    return data.values, data_meta["Type"].values.ravel(), corr.values, masked_corr.values


def get_irs_brcapam50(recompute_corr=True):
    """
    Loads the BRCA dataset, performs IRS correction and computes the 
    correlation matrix. Correlations between samples from same batch are masked to np.NaN,
    because we expect them to be biased.

    Parameters
    ----------
    recompute_corr : bool, optional
        Whether to recompute the correlation matrix. If false, it will be loaded from the file. The default is True.

    Returns
    -------
    data : ndarray
        The original data of size n_train x n_genes.
    data_meta : ndarray
        Corresponding class assignments (in the same order ! ).
    corr : ndarray
        The matrix of pairwise correlations.

    """
    brca = pd.read_csv("Datasets/Proteomics/BRCA/joined_datasets_pam50.csv", index_col=0, na_values=[0,0.0,np.NaN,"NAN"])
    metadata = pd.read_csv("Datasets/Proteomics/BRCA/metadata_pam50.csv", index_col=0)
    
        # correct for differential loading
    brca = IRS.sample_loading_adjustment(brca, metadata)
    # IRS
    brca = IRS.IRS(brca, metadata)
    # we don't want the bridges anymore
    data, data_meta = IRS.drop_BridgeSamples(brca, metadata)
    # just to be sure that they are ordered
    data = data.loc[data_meta.index.values]
    
    if recompute_corr:
        corr = data.transpose().corr(method="spearman")
        corr.to_csv("Datasets/Proteomics/BRCA/corrmatrix_irs.csv", na_rep="NaN")
    else:
        corr = pd.read_csv("Datasets/Proteomics/BRCA/corrmatrix_irs.csv", index_col=0, na_values="NaN")
    
    return data.values, data_meta["Type"].values.ravel(), corr.values



