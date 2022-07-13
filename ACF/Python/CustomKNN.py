# -*- coding: utf-8 -*-
"""
Implementation of Custom KNN Classifier (in order to include random oversampling
                                         and to include unify methods AND to
                                         ensure comparability in e.g. computing
                                         correlation matrices)

@author: Yannis Schumann
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.extmath import weighted_mode
import numpy as np
import pandas as pd

class CustomKNN(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible, custom implementation of the brute KNN classifier that
    implements a hyperparameter for random oversampling.
    """
    
    def __init__(self, n_neighbors=10, weights = "uniform", corr_type="pearson", oversampling=False, precomputed=False):
        """
        Construct CustomKNN

        Parameters
        ----------
        n_neighbors : int, optional
            Number of nearest neighbors to consider. The default is 10.
        weights : {'uniform','distance'}, optional
            How to weight the votes from the nearest neigbors. The default is "uniform".
        corr_type : {‘pearson’, ‘kendall’, ‘spearman’} or callable, default='pearson'
            Pairwise correlation used.
        oversampling : bool, optional
            Whether to artificially balance the dataset by using random oversampling. The default is False.
        precomputed : bool
            Indicates, whether to use a precomputed correlation matrix. 
        Returns
        -------
        None.

        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.oversampling = oversampling
        self.precomputed = precomputed
        self.corr_type = corr_type
    
    
    def __rescale_corr_mat__(self, corr_mat):
        """
        Scales the correlation matrix, such that large correlations become small
        distances and vice versa

        Parameters
        ----------
        corr_mat : pd.DataFrame
            The DataFrame of pairwise correlations.

        Returns
        -------
        pd.DataFrame
            The DataFrame of pairwise distances.

        """
        # by definition, correlations are in [-1,1]
        return pd.DataFrame((1-corr_mat.values)/2)
        
        
    
    def __nearest_neighbor_vote__(self, corr_mat, y):
        """
        Finds the n nearest neighbors and performs a class vote among them.
        Optionally, the weights of their votes will be adjusted inversely to their distance

        Parameters
        ----------
        corr_mat : pd.DataFrame
            Correlation matrix of shape n_test x n_train(_oversampled).
        y : list-like
            The training labels.

        Returns
        -------
        np.ndarray
            Array with the predicted classes.

        """
        #  find indices of the smallest elements along axis 1 to the beginning
        smallest_indices = np.argpartition(corr_mat.values, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        
        # list for the predictions
        y_test = []
        
        for test_idx in range(corr_mat.shape[0]):
            distances = corr_mat.values[test_idx, smallest_indices[test_idx]]
            labels = y[smallest_indices[test_idx]]
            if self.weights == "uniform":
                y_test.append(weighted_mode(a=labels,w=np.ones(len(labels)))[0]) #one-dimensional, so flattening operation should not be neccessary
            if self.weights == "distance":
                y_test.append(weighted_mode(a=labels,w=1/distances)[0]) #same here
        return np.asarray(y_test).ravel() #ravel() is necessary, because weighted_mode returns an array of modal values
    
    
    def __custom_oversampling__(self, corr_mat, y):
        """
        Generates an oversampled DataFrame, such that all classes are balanced

        Parameters
        ----------
        corr_mat : pd.DataFrame
            DataFrame of shape n_test x n_train with all pairwise correlations.
        y : list-like
            List of the training labels.

        Returns
        -------
        oversampled_corr_mat : pd.DataFrame
            Oversampled DataFrame of shape n_test x n_train_oversampled.
        oversampled_y : np.ndarray
            Array of the class-labels of the oversampled set of training instances.

        """
        rng = np.random.default_rng()
        
        # check number of occurences
        labels, counts = np.unique(y, return_counts=True)
        # get count of majority class
        max_count = max(counts)
        # allocate array of corresponding shape
        no_add_corrs = sum([max_count - count for count in counts]) # number of required additional correlations
        additional_correlations = np.zeros((corr_mat.shape[0], no_add_corrs))
        additional_ys = []
        
        # iterate over all classes
        offset = 0
        for label, count in zip(labels, counts):
            # select samples to oversample (potentially multiple times)
            required_samples = max_count - count
            if required_samples==0:
                continue
            available_indices = np.where(y==label)[0]
            selected_indices = rng.choice(available_indices, size=required_samples)
            # copy it into the array
            additional_correlations[:,offset:offset+required_samples] = corr_mat.values[:,selected_indices].copy()
            #increment offset accordingly
            offset = offset+required_samples
            # append to additional_ys
            for i in range(required_samples):
                additional_ys.append(label)
        
        # stack the correlations
        oversampled_corr_mat = pd.DataFrame(np.hstack((corr_mat.values, additional_correlations)))
        # append the correct labels
        oversampled_y = list(y)
        oversampled_y.extend(additional_ys)
        oversampled_y = np.asarray(oversampled_y)
        
        return oversampled_corr_mat, oversampled_y
    
    
    def fit(self, X,y):
        """
        Fits the custom KNN classifier.

        Parameters
        ----------
        X : arraylike
            Either matrix of shape n_train x n_features if precomputed= False, or correlation
            matrix of shape n_train x n_train otherwise.
        y : array-like
            List of n_train training labels.

        Returns
        -------
        CustomKNN
            The trained classifier.

        """
        # check shape and no inf's, NaN is only allowed (when precomputed=False, we're going to use pairwise correlations anyway, when precomputed=True, X is going to be ignored)
        X,y = check_X_y(X, y, force_all_finite="allow-nan")
        
        # convert to DataFrame to make use of functions for pairwise correlations
        X = pd.DataFrame(np.asarray(X))
        # Store training observations
        self.X_ = X
        self.y_ = np.asarray(y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # indicate status
        self.is_fitted_ = True
        return self


    def predict(self, X):
        """
        Predicts the classes of X

        Parameters
        ----------
        X : array-like
            Either matrix of shape n_test x n_features (if precompute=False) or
            n_test x n_train (if precompute=True).

        Returns
        -------
        np.ndarray
            Array with the predicted class labels.

        """
        # Validate array
        X = check_array(X, force_all_finite="allow-nan")
        X = np.asarray(X)
        
        # make X DataFrame
        X = pd.DataFrame(X)
        
        if self.precomputed == False:
            # allocate array of appropriate shape
            test_corr_mat = np.zeros((X.shape[0], self.X_.shape[0]))
            # compute all pairwise correlations
            for reference_idx in range(self.X_.shape[0]):
                test_corr_mat[:,reference_idx] = X.corrwith(self.X_.loc[reference_idx], axis=1, method=self.corr_type).values
            # make it a dataframe
            self.test_corr_mat = pd.DataFrame(test_corr_mat)
        else:
            # if precomputed, X is the correlation matrix with all training samples
            self.test_corr_mat = X
        
        # required for prediction and may be overwritten by oversampling
        self.test_y = self.y_
        
        if self.oversampling:
            # overwrite test fields, if oversampling=True
            self.test_corr_mat, self.test_y = self.__custom_oversampling__(self.test_corr_mat, self.test_y)
        
        # convert correlations to distances
        self.test_corr_mat = self.__rescale_corr_mat__(self.test_corr_mat)
        
        return self.__nearest_neighbor_vote__(self.test_corr_mat, self.test_y)
        