# -*- coding: utf-8 -*-
"""
Tests for IRS

@author: Yannis Schumann
"""
import IRS
import pandas as pd
import numpy as np

def test_sample_loading_adjustment():
    data = pd.DataFrame([
        [1,2,np.NaN],
        [2, np.NaN, 4],
        [0,1,2],
        [2,3,np.NaN]])
    metadata = pd.DataFrame([[1],[1],[2],[2]], columns=["Batch"])
    adjusted_data = IRS.sample_loading_adjustment(data, metadata)
    expected = [[1.5, 3, np.NaN],[1.5, np.NaN, 3],[0,4/3,8/3],[8/5,3*4/5,np.NaN]]
    np.testing.assert_array_almost_equal(adjusted_data.values, expected)


def test_IRS():
    data = pd.DataFrame([
        [1,4,np.NaN, 9],
        [16,1,5,1],
        [2,3,4,5],
        [6,7,8,9]])
    metadata = pd.DataFrame([
        ["Bridge", 1],
        ["Bridge",2],
        ["A",1],
        ["A",2]], columns=["Type", "Batch"])
    expected = [[4,2,np.NaN,3],[4,2,5,3],[2*4,3/2,np.NaN,5/3],[6/4,7*2,8,9*3]]
    result = IRS.IRS(data, metadata)
    np.testing.assert_array_almost_equal(result.values, expected)

def test_drop_BridgeSamples():
    data = pd.DataFrame([
        [1,4,np.NaN, 9],
        [16,1,5,1],
        [2,3,4,5],
        [6,7,8,9]])
    metadata = pd.DataFrame([
        ["Bridge-1", 1],
        ["Bridge",2],
        ["A",1],
        ["A",2]], columns=["Type", "Batch"])
    expected_data = np.array([[2,3,4,5],[6,7,8,9]])
    expected_meta = np.array([["A",1],["A",2]], dtype="object")
    result_data, result_metadata, = IRS.drop_BridgeSamples(data, metadata)
    np.testing.assert_array_almost_equal(result_data.values, expected_data)
    np.testing.assert_array_equal(result_metadata.values, expected_meta)

def test_mask_biased_correlations():
    metadata = pd.DataFrame([[1],[2],[1],[1],[2]], columns=["Batch"])
    data = pd.DataFrame([
        [1,2,3,4,5],
        [2,3,4,5,6],
        [3,4,5,6,7],
        [4,5,6,7,8],
        [5,6,7,8,9]])
    result = IRS.mask_biased_correlations(data, metadata)
    expected_result = pd.DataFrame([
        [np.NaN,2,np.NaN,np.NaN,5],
        [2,np.NaN,4,5,np.NaN],
        [np.NaN,4,np.NaN,np.NaN,7],
        [np.NaN,5,np.NaN,np.NaN,8],
        [5,np.NaN,7,8,np.NaN]])
    np.testing.assert_array_equal(result.values, expected_result.values)