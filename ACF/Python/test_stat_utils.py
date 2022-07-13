# -*- coding: utf-8 -*-
"""
test the functions for the statistical tests

@author: Yannis Schumann
"""
import stat_utils
import numpy as np

def test_corrected_t():
    diffs = [0.0,0.2]
    n_test = 50
    n_train = 100
    assert np.isclose(stat_utils.corrected_t(diffs, n_train, n_test),0.1/np.sqrt(0.02))
