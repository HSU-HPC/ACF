# -*- coding: utf-8 -*-
"""
IRS adjustment

@author: Yannis Schumann
"""
import numpy as np
import pandas  as pd
import warnings
from scipy.stats.mstats import gmean
import itertools


def mask_biased_correlations(corr, metadata):
    corr_new = corr.copy()
    
    batches = metadata["Batch"].unique()
    
    for b in batches:
        indices = metadata[metadata["Batch"]==b].index.values
        corr_new.loc[indices, indices] = np.NaN
    
    return corr_new

def sample_loading_adjustment(data, metadata):
    """
    Performs Sample Loading Adjustment - grand total of all intensities of all
    proteins are matched between the channels within each TMT/iTRAQ run.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of shape n_observations x n_features.
    metadata : pd.DataFrame
        DataFrame with same indices as data, but with a column 'Batch' containing
        the batch ID.

    Returns
    -------
    norm_array : pd.DataFrame
        The adjusted data.

    """
    
    norm_array = data.copy()
    
    # get samples from each batch
    batches = metadata["Batch"].unique()
    sample_indices_per_batch = [metadata[metadata["Batch"]==batch].index.values for batch in batches]

    for sample_idxs in sample_indices_per_batch:
        # find mean intensity for this batch
        target = np.mean(np.nansum(data.loc[sample_idxs].values, axis=1)) # mean total intensity in that batch
        for sample in sample_idxs:
            # and scale each sample accordingly
            norm_array.loc[sample] = data.loc[sample]*target/np.nansum(data.loc[sample].values) # bring to mean int. of batch
    return norm_array


def IRS(data, metadata):
    """
    Performs "internal reference sample"-normalization to adjust batches towards
    each other (see Plubell et al). Usually, this is performed after sample-loading-adjustment.
    Data contains protein intensities in columns and samples in rows.
    metadata with column "Batch" and column "Type". Bridge samples should be labelled "Bridge" in type.
    We allow for multiple bridges per batch.

    Returns a DataFrame with all (!, including bridge-samples) rows 
    adjusted according to IRS-normalization.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of shape n_observations x n_features.
    metadata : pd.DataFrame
        DataFrame with same indices as data, but with a column 'Batch' containing
        the batch ID and "Type" that labels the bridges as 'Bridge' (or at least contains a 'Bridge'...).

    Returns
    -------
    irs_data : pd.DataFrame
        The adjusted data.

    """
    # copy the input array for the results
    irs_data = data.copy()
    
    #calculate average bridge standard to account for replicate standards
    
    
    batches = metadata["Batch"].unique()
    # allocate dataframe for the means of the bridges from each batch
    average_bridges = pd.DataFrame(np.NaN, index=batches, columns=data.columns)
    # iterate over the batches
    for batch in batches:
        # select all bridges from that batch
        bridges = metadata[(metadata["Type"]=="Bridge") & (metadata["Batch"]==batch)].index.values
        # in case that only one bridge exists in this batch, numpy.nanmean will raise a RuntimeWarning "Mean of empty slice" and will return np.NaN (which is, what we want here). Suppressing the warning. 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            average_bridges.loc[batch] = np.nanmean(data.loc[bridges], axis=0) # and compute their average
    
    # calculate geometric mean of all average bridges (per feature)
    global_average = gmean(np.ma.masked_invalid(average_bridges.to_numpy()), axis=0)
    global_average = global_average.filled(np.NaN) # a minority of proteins might not be present in the bridges at all.
    global_average = pd.DataFrame([global_average for _ in batches], index=batches, columns=data.columns)
    # get scaling_factors
    scaling_factors = global_average.div(average_bridges)
    #scale data accordingly
    for batch_idx in range(len(batches)):
        irs_data.loc[metadata["Batch"]==batches[batch_idx]] = data.loc[metadata["Batch"]==batches[batch_idx]].mul(scaling_factors.loc[batches[batch_idx]], axis=1)
    
    return irs_data


def drop_BridgeSamples(data, metadata):
    """
    Drops all bridge samples (samples with type "Bridge") from data and metadata.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of shape n_observations x n_features.
    metadata : pd.DataFrame
        DataFrame with same indices as data, but with a column 'Batch' containing
        the batch ID and "Type" that labels the bridges as 'Bridge' (or at least contains a 'Bridge' in the type).

    Returns
    -------
    data_without : pd.DataFrame
        Data without the bridges.
    metadata_without : pd.DataFrame
        Data without the bridges.

    """
    # select bridges
    bridge_samples = [x for x in metadata.index.values if "Bridge" in metadata.loc[x, "Type"]]
    data_without = data.drop(bridge_samples, axis=0)
    metadata_without = metadata.drop(bridge_samples, axis=0)
    return data_without, metadata_without