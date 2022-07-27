# -*- coding: utf-8 -*-
"""
Implementation of ACF and its variants.

Test with 
>>> import pytest
>>> pytest.main()

@author: Yannis Schumann
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
import numpy as np
import pandas as pd


class ACFClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn  compatible implementation of the ACF algorithm.
    For more information refer to the paper.
    
    Parameters
    ----------
    baseline_classifier : scikit-learn compatible classifier
        Classifier that is used internally for predicting test instances.
    corr_type : {‘pearson’, ‘kendall’, ‘spearman’} or callable, default='pearson'
        Pairwise correlation used.
    strategy : {'mean', 'median', 'kld'}, default='mean'
        Whether to use mean or median correlations as features or Kullback-Leiber-Distances of the distributions. The median might be most versatile for severely skewed distributions.
    variant : {None, 'F-ACF', 'B-ACF'}, default=None
        Whether to use default ACF (default), the fast implementation based on reference samples ('F-ACF') or the version that allows to include biases ('B-ACF'). These options are ot compatible at the moment.
    n_ref : int, default=None
        Number of reference samples to use per class when selecting variant='F-ACF'. Ignored otherwise. Instances are selected at random.
    precomputed : bool, default=False
        If true, the argument X of fit() and predict() is expected to be the precomputed correlation matrix
        between training-/testing- and reference-instances. np.NaN is allowed (e.g. to mask biased correlations such as self-correlations).
        Reduces computational overhead in crossvalidation-procedures.
    scale_Corrmat : {'Standard','MinMax',None}, default=None
        Whether (and how) to scale the rows of the correlation matrix prior to averaging
    normalize_AC : {'l1','l2',None}, default=None
        Whether (and how) to normalize the rows of the average correlations
    
    Examples
    --------
    see Readme.md
    
    """
    
    
    def __init__(self, baseline_classifier, corr_type="pearson", strategy="mean", variant=None, n_ref = None, precomputed = False, scale_Corrmat = None, normalize_AC = None):
        self.baseline_classifier = baseline_classifier
        self.corr_type = corr_type
        self.strategy = strategy
        self.variant = variant
        self.n_ref = n_ref
        self.precomputed = precomputed
        self.scale_Corrmat = scale_Corrmat
        self.normalize_AC = normalize_AC
    
    
    def __scale_correlations__(self, corr_mat):
        """
        Normalizes the correlation matrix prior to averaging according to
        parameter self.scale_Corrmat

        Parameters
        ----------
        corr_mat : pandas.DataFrame
            The correlation matrix.

        Raises
        ------
        ValueError
            If self.scale_Corrmat does not take the correct parameters.

        Returns
        -------
        np.ndarray
            The normalized correlation matrix.

        """
        if self.scale_Corrmat == None:
            # do nothing
            return corr_mat.values
        elif self.scale_Corrmat == "Standard":
            # scale accordingly with NaN-compatibility
            return (corr_mat.values - np.nanmean(corr_mat.values, axis=1, keepdims=True))/np.nanstd(corr_mat.values, axis=1, keepdims=True)
        elif self.scale_Corrmat == "MinMax":
            # scale accordingly with NaN-compatibility
            return (corr_mat.values - np.nanmin(corr_mat.values, axis=1, keepdims=True))/(np.nanmax(corr_mat.values, axis=1, keepdims=True)-np.nanmin(corr_mat.values, axis=1, keepdims=True))
        else:
            raise ValueError("scale_Corrmat must be in {'Standard','MinMax',None}")
    

    def __normalize_average_correlations__(self, corr_mat):
        """
        Normalizes the matrix average correlations according to
        parameter self.normalize_AC

        Parameters
        ----------
        corr_mat : np.ndarray
            The matrix of average correlations.

        Raises
        ------
        ValueError
            If self.normalize_AC does not take the correct parameters.

        Returns
        -------
        np.ndarray
            The normalized matrix.

        """
        if self.normalize_AC == None:
            # do nothing
            return corr_mat
        elif self.normalize_AC == "l1":
            # normalize accordingly with NaN-compatibility
            return corr_mat/np.nansum(np.abs(corr_mat), axis=1, keepdims=True)
        elif self.normalize_AC == "l2":
            # normalize accordingly with NaN-compatibility
            return corr_mat/np.sqrt(np.nansum(np.power(corr_mat,2), axis=1, keepdims=True))
        else:
            raise ValueError("normalize_AC must be in {'l1','l2',None}")
        
    
    def __get_average_correlations__(self, corr_mat):
        """
        Computes the average correlations of each training observation with reference samples per class.
        
        Parameters
        ----------
        corr_mat : pd.DataFrame
        DataFrame with pairwise correlations of each instance with all reference instances.

        Raises
        ------
        ValueError
            If self.strategy is not in {'mean', 'median'}.

        Returns
        -------
        average_correlations : np.ndarray
            Array of shape n_train x n_classes with the respective average correlations per class

        """
        
        # allocate memory
        average_correlations = np.zeros((corr_mat.shape[0], len(self.classes_)))
        
        # all reference instances
        all_indices = np.concatenate(self.indices_per_class)
        
        # use selected method
        if self.strategy == "mean":
            method = np.nanmean
        elif self.strategy == "median":
            method = np.nanmedian
        else:
            raise ValueError("Strategy keyword can only take values \"mean\" or \"median\".", self.strategy, "was passed.")
        # compute and fill in the average correlations
        for class_counter in range(len(self.classes_)):
            for training_instance_counter in range(corr_mat.shape[0]):
                average_correlations[training_instance_counter, class_counter] = method(corr_mat.values[training_instance_counter, np.in1d(all_indices, self.indices_per_class[class_counter]).nonzero()[0]])
        
        return average_correlations


    def __get_kld_per_class_fit__(self, corr_mat):
        """
        Computes the Kullback-Leibler divergence (~distance) of each training
        observation with reference samples per class.
        
        Parameters
        ----------
        corr_mat : pd.DataFrame
        DataFrame with pairwise correlations of each instance with all reference instances.


        Returns
        -------
        klds : np.ndarray
            Array of shape n_train x n_classes with the respective kld per class
        

        """
        
        # allocate memory
        klds = np.zeros((corr_mat.shape[0], len(self.classes_)))
        
        # all reference instances
        all_indices = np.concatenate(self.indices_per_class)
        
        # compute and fill in the Kullback-Leibler-distances
        self.possible_values = np.linspace(-1,1,100)[:,np.newaxis] # values to sample pdf
        
        # compute kld for each observation
        for observation_idx in range(corr_mat.shape[0]):
            # and each class
            for class_k_idx in range(len(self.classes_)):
                class_indices = np.in1d(all_indices, self.indices_per_class[class_k_idx]).nonzero()[0] # reference indices from this class
                class_indices = class_indices[np.nonzero(~np.isnan(corr_mat.values[observation_idx, class_indices]))[0]] # all reference observations that are not the considered observation
                
                # distribution of correlations of observation with other samples --> may still contain self-correlation
                distribution_observation = corr_mat.values[observation_idx, class_indices]
                distribution_observation = distribution_observation.ravel()[:,np.newaxis]
                # distribution references
                distribution_references = corr_mat.values[np.ix_(class_indices, class_indices)]
                distribution_references = distribution_references[np.nonzero(~np.isnan(distribution_references))]
                distribution_references = distribution_references.ravel()[:,np.newaxis]
                
                
                kde_observation = KernelDensity(bandwidth=0.05, kernel="gaussian") # bw=0.05 is default, future versions may support this as hyperparameter
                kde_references = KernelDensity(bandwidth=0.05, kernel="gaussian")
                
                # fit KDEs
                kde_observation.fit(distribution_observation)
                kde_references.fit(distribution_references)
                
                # sample pdf
                p_observed = np.exp(kde_observation.score_samples(self.possible_values))
                p_reference = np.exp(kde_references.score_samples(self.possible_values))
                
                # compute and store KLD
                klds[observation_idx, class_k_idx] = entropy(p_observed, p_reference)
                
        return klds


    def __get_kld_per_class_predict__(self, test_corr_mat, corr_mat):
        """
        Computes the Kullback-Leibler divergence (~distance) of each training
        observation with reference samples per class.
        
        Parameters
        ----------
        test_corr_mat : pd.DataFrame
            DataFrame with pairwise correlations of all test observations with all reference instances.
        corr_mat : pd.DataFrame
            DataFrame with pairwise correlations of all reference instances.


        Returns
        -------
        klds : np.ndarray
            Array of shape n_test x n_classes with the respective kld per class
        

        """
        
        # allocate memory
        klds = np.zeros((test_corr_mat.shape[0], len(self.classes_)))
        
        # all reference instances
        all_indices = np.concatenate(self.indices_per_class)
        
        # compute and fill in the Kullback-Leibler-distances
        self.possible_values = np.linspace(-1,1,100)[:,np.newaxis] # values to sample pdf
        
        # compute kld for each observation
        for observation_idx in range(test_corr_mat.shape[0]):
            # and each class
            for class_k_idx in range(len(self.classes_)):
                class_indices = np.in1d(all_indices, self.indices_per_class[class_k_idx]).nonzero()[0] # reference indices from this class
                class_indices = class_indices[np.nonzero(~np.isnan(corr_mat.values[observation_idx, class_indices]))[0]] # all reference observations that are not the considered observation
                
                # distribution of correlations of observation with other samples --> may still contain self-correlation
                distribution_observation = test_corr_mat.values[observation_idx, class_indices]
                distribution_observation = distribution_observation.ravel()[:,np.newaxis]
                # distribution references
                distribution_references = corr_mat.values[np.ix_(class_indices, class_indices)]
                distribution_references = distribution_references[np.nonzero(~np.isnan(distribution_references))]
                distribution_references = distribution_references.ravel()[:,np.newaxis]
                
                
                kde_observation = KernelDensity(bandwidth=0.05, kernel="gaussian") # bw=0.05 is default, future versions may support this as hyperparameter
                kde_references = KernelDensity(bandwidth=0.05, kernel="gaussian")
                
                # fit KDEs
                kde_observation.fit(distribution_observation)
                kde_references.fit(distribution_references)
                
                # sample pdf
                p_observed = np.exp(kde_observation.score_samples(self.possible_values))
                p_reference = np.exp(kde_references.score_samples(self.possible_values))
                
                # compute and store KLD
                klds[observation_idx, class_k_idx] = entropy(p_observed, p_reference)
                
        return klds


    def __get_indices_per_class__(self):
        """
        Selects the reference samples.
        If self.variant = None or self.variant=B-ACF, all training observations per class are selected.
        If self.variant = F-ACF, it randomly selects self.n_ref reference samples per class.
        
        Returns
        -------
        indices_per_class : list of np.ndarray
            List of the indices of the reference observations selected per class.

        """
        indices_per_class = []
        if self.variant == None or self.variant == "B-ACF":
            for c in self.classes_:
                indices_per_class.append(np.where(self.y_==c)[0])
        elif (self.variant == "F-ACF" and not self.precomputed):
            rng = np.random.default_rng()
            for c in self.classes_:
                # select all possible training observations of the corresponding class
                available_indices = np.where(self.y_==c)[0]
                # randomly select n_ref of them as a subset of reference observations
                indices_per_class.append(rng.choice(available_indices, size=self.n_ref, replace=False))
        elif (self.variant=="F-ACF" and self.precomputed):
            total_refs = self.X_.shape[1]
            assert total_refs%self.n_ref==0 # need to have n_references per class
            start_index = 0
            for c in self.classes_:
                indices_per_class.append(np.arange(start_index, start_index+self.n_ref))
                start_index+=self.n_ref
        return indices_per_class


    def fit(self, X,y, biased_elements=None, additional_covariable=[]):
        """
        Fit the baseline classifier to the average correlations

        Parameters
        ----------
        X : array-like
            Array-like of shape shape n_train x n_features (or n_train x (n_classes*n_reference), if precomputed=True. In this case, the reference instances should be sorted by class). NaN is allowed, +- np.inf will raise an error.
        y : list-like
            List of training-labels with length n_train.
        biased_elements : list of tuples, optional
            Tuples of sample-indices with correlations that are known to be biased -- those will be filtered if precomputed=False.
            For example
            [(1,3), (4,2)] will set correlation_matrix[1,3]=correlation_matrix[3,1]=correlation_matrix[4,2]=correlation_matrix[2,4] = np.NaN
            The default is None.
        additional_covariable : np.ndarray, optional
            Optional covariables with shape n_instances x n_features. NaN is not permitted. The default is []. 

        Returns
        -------
        self
            Instance of ACFClassifier
        """
        # check shape and no inf's
        X,y = check_X_y(X, y, force_all_finite="allow-nan")
        if len(additional_covariable)!=0:
            X_c, y = check_X_y(additional_covariable, y)
            self.X_c = X_c # directly store as field
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
                
        # convert to DataFrame to make use of functions for pairwise correlations
        X = pd.DataFrame(np.asarray(X))
        # Store training observations
        self.X_ = X
        self.y_ = np.asarray(y)
        
        # select used indices per class
        self.indices_per_class = self.__get_indices_per_class__()
        all_indices = np.concatenate(self.indices_per_class)
        
        # check that correlations are supposed to be computed here
        if self.precomputed == False:
            # allocate matrix of appropriate shape for pairwise correlations
            
            corr_mat = np.zeros(shape=(self.X_.shape[0],len(all_indices)))
            for reference_idx in range(len(all_indices)):
                corr_mat[:,reference_idx] = X.corrwith(X.loc[all_indices[reference_idx]], axis=1, method=self.corr_type).values
                # self-correlations are biased -> mask them
                corr_mat[all_indices[reference_idx],reference_idx] = np.NaN
            self.corr_mat = pd.DataFrame(corr_mat, index=X.index.values, columns=all_indices)
            
            #set biased elements to NaN
            if biased_elements != None and self.variant=="B-ACF":
                for biased_pair in biased_elements:
                    self.corr_mat.loc[biased_pair[0],biased_pair[1]] = self.corr_mat.loc[biased_pair[1],biased_pair[0]] = np.NaN
        else:
            # if correlation matrix is precomputed, select the required (sub)set of reference correlations
            self.corr_mat = self.X_[all_indices]
            self.corr_mat.values[np.nonzero(self.corr_mat.values==1)] = np.NaN # exclude self-correlations
        
        # scale after(!) setting the biased elements to NaN
        self.corr_mat = pd.DataFrame(self.__scale_correlations__(self.corr_mat), index=X.index.values, columns=all_indices)
        # compute all pairwise correlations
        if self.strategy in ["mean", "median"]:
            self.average_correlations = self. __get_average_correlations__(self.corr_mat)
        else:
            self.average_correlations = self.__get_kld_per_class_fit__(self.corr_mat)
        # normalize average correlations
        self.average_correlations = self.__normalize_average_correlations__(self.average_correlations)
        
        # (optionally) combine features
        if len(additional_covariable)!=0:
            self.training_features = np.hstack((self.average_correlations, self.X_c))
        else:
            self.training_features = self.average_correlations
        
        # standardize features
        self.scaler_ = StandardScaler()
        self.training_features = self.scaler_.fit_transform(self.training_features)
        
        # train baseline classifier
        self.baseline_classifier.fit(self.training_features, y)
        
        # indicate status
        self.is_fitted_ = True
        return self


    def predict(self, X, biased_elements=None, additional_covariable=[]):
        """
        Predicts classes of test observations based on their average (mean/median)
        correlations to the training instances.

        Parameters
        ----------
        X : array-like
            Array-like object of shape n_test x n_features (or n_test x n_references, if precomputed = True). +- np.inf is prohibited.

        biased_elements : list of tuples, optional
            Tuples of sample-indices with correlations that are known to be biased -- those will be filtered if precomputed=False.
            For example
            [(1,3)] 
            will set the correlation of the first test instance with the third training instance to np.NaN
        additional_covariable : np.ndarray, optional
            Optional covariable with shape n_instances x n_features. NaN is not permitted. The default is []. 
            
        Returns
        -------
        y_test : np.ndarray
            Array with the predicted class labels.

        """
        # Validate array
        X = check_array(X, force_all_finite="allow-nan")
        if len(additional_covariable)!=0:
            X_c_test = check_array(additional_covariable)
            X_c_test = np.array(X_c_test)
        X = np.asarray(X)
        
        # make X DataFrame
        X = pd.DataFrame(X)
        
        all_indices = np.concatenate(self.indices_per_class)
        
        if self.precomputed == False:
            # allocate array of appropriate shape
            test_corr_mat = np.zeros((X.shape[0], len(all_indices)))
            for reference_idx in range(len(all_indices)):
                test_corr_mat[:,reference_idx] = X.corrwith(self.X_.loc[all_indices[reference_idx]], axis=1, method=self.corr_type).values
            self.test_corr_mat = pd.DataFrame(test_corr_mat)
            #set biased elements to NaN
            if biased_elements != None and self.variant=="B-ACF":
                for biased_pair in biased_elements:
                    self.test_corr_mat.loc[biased_pair[0],np.where(all_indices==biased_pair[1])[0][0]] = np.NaN
        else:
            self.test_corr_mat = X[all_indices]
        
        # scale according to hyperparametern
        self.test_corr_mat = pd.DataFrame(self.__scale_correlations__(self.test_corr_mat), index=X.index.values, columns=all_indices)
        # compute average correlations
        #self.average_test_correlations = self.__get_average_correlations__(self.test_corr_mat)
        if self.strategy in ["mean", "median"]:
            self.average_test_correlations = self. __get_average_correlations__(self.test_corr_mat)
        else:
            self.average_test_correlations = self.__get_kld_per_class_predict__(self.test_corr_mat, self.corr_mat)
        # normalize average correlations
        self.average_test_correlations = self.__normalize_average_correlations__(self.average_test_correlations)
        # concatenate (optionally)
        if len(additional_covariable)!=0:
            self.test_features = np.hstack((self.average_test_correlations, X_c_test))
        else:
            self.test_features = self.average_test_correlations
        # standardize features
        self.test_features = self.scaler_.transform(self.test_features)
        
        y_test = self.baseline_classifier.predict(self.test_features)
        
        return y_test
