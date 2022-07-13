# -*- coding: utf-8 -*-
"""
DBC-algorithm by Wei et al in "Exploring the within- and between-class
correlation distributions for tumor classification".

@author: Yannis Schumann
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
from scipy.stats import entropy
from itertools import combinations_with_replacement, product

class DBCClassifier(BaseEstimator, ClassifierMixin):

    
    def __init__(self, corr_type="pearson", precomputed=False, istest=False, variant=None, n_ref = None):
        """
        Create instance.

        Parameters
        ----------
        corr_type : one in ("pearson", "spearman", "kendall" or callable), optional
            The type of correlation to use. The default is "pearson".
        precomputed : bool, optional
            If true, the arguments X in fit and predict are supposed to be the precomputed correlation matrices
        istest : bool, optional. The default is false.
            Whether this instance is intended for a test --> the kernel density estimate will be adapted for other metrics than correlations, to ensure comparability with original publication
        variant : one of {None, "F-DBC"}
        n_ref : int or None
            Number of reference instances to select per class
        
        Returns
        -------
        None.

        """
        self.corr_type = corr_type
        self.precomputed = precomputed
        self.istest = istest
        self.variant = variant
        self.n_ref = n_ref


    def __get_indices_per_class__(self):
        """
        Selects the reference samples.
        If self.variant = None all training observations per class are selected.
        If self.variant = F-DBC, it randomly selects self.n_ref reference samples per class.
        
        Returns
        -------
        indices_per_class : list of np.ndarray
            List of the indices of the reference observations selected per class.

        """
        
        indices_per_class = []
        if self.variant == None:
            for c in self.classes_:
                indices_per_class.append(np.where(self.y_==c)[0])
        elif self.variant == "F-DBC":
            rng = np.random.default_rng()
            for c in self.classes_:
                # select all possible training observations of the corresponding class
                available_indices = np.where(self.y_==c)[0]
                # randomly select n_ref of them as a subset of reference observations
                indices_per_class.append(rng.choice(available_indices, size=self.n_ref, replace=False))
        return indices_per_class

    
    def fit(self, X,y):
        """
        Stores X and y and computes the probbility-distributions f_ij,
        so that they can be used for prediction in a lazy fashion.

        Parameters
        ----------
        X : ndarray
            Training data of shape (n_samples, n_features).
        y : ndarray
            Training classes of shape (n_samples, ).

        Returns
        -------
        DBCClassifier (fitted instance)

        """
        # check shape and no inf's
        X,y = check_X_y(X, y, force_all_finite="allow-nan")
        # make dataframe
        X = pd.DataFrame(np.asarray(X))
        self.X_ = X # store dataframe 
        self.y_ = np.asarray(y) # store corresponding training classes
        self.classes_ = unique_labels(y) # store the classes that were seen during training

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
            
        else:
            # if correlation matrix is precomputed, select the required (sub)set of reference correlations
            self.corr_mat = self.X_[all_indices]
            self.corr_mat.values[np.nonzero(self.corr_mat.values==1)] = np.NaN # exclude self-correlations, if any
        
        
        # interval for correlations
        # tested also higher number of datapoints, did not improve performance
        if self.istest:
            self.possible_values = np.linspace(np.nanmin(self.corr_mat),np.nanmax(self.corr_mat),100)[:, np.newaxis] # if we use euclidean distance for example, [-1,1] would not make sense --> need to modify this for comparisons with original publication
        else:
            self.possible_values = np.linspace(-1,1,100)[:, np.newaxis]
        # allocate memory for the block-wise probability correlations
        self.distribution_matrix = np.zeros((len(self.classes_), len(self.classes_), len(self.possible_values)))
        
        # iterate over every unique pair of classes
        for class_k_idx, class_j_idx in product(range(len(self.classes_)), range(len(self.classes_))):
            class_k = self.classes_[class_k_idx] # class 1
            class_j = self.classes_[class_j_idx] # class 2
            
            # get corresponding block in correlation matrix
            training_indices_k = np.nonzero(self.y_==class_k)[0] # axis 0 of corr_mat
            training_indices_j = np.in1d(all_indices, self.indices_per_class[class_j_idx]) # axis 1 of corr_mat
            
            block_corr = self.corr_mat.values[np.ix_(training_indices_k,training_indices_j)].ravel()
            
            # exclude NaNs, for possible external bias models and also for self-correlations
            block_corr = block_corr[np.nonzero(~np.isnan(block_corr))]
            # fit KernelDensityEstimate
            kde_kj = KernelDensity(bandwidth=0.05, kernel="gaussian")
            kde_kj.fit(block_corr[:, np.newaxis])
            
            # get PDF on self.possible_values and store to self.distribution_matrix
            self.distribution_matrix[class_k_idx, class_j_idx, :] = np.exp(kde_kj.score_samples(self.possible_values))
            
            
        self.is_fitted_ = True # was fitted. Everything is stored away.
        
        return self
        
        
    
    def predict(self, X):
        """
        Predicts class of all samples in X, using the DBC procedure

        Parameters
        ----------
        X : ndarray
            Array of test instances with shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Predicted instances as array of shape (n_samples,).

        """
        
        # verify input and make to dataframe, in order to exploit pandas .corrwith-function
        X = check_array(X, force_all_finite="allow-nan")
        X = np.asarray(X)
        X = pd.DataFrame(X)

        all_indices = np.concatenate(self.indices_per_class)
        
        if self.precomputed == False:
            # allocate array of appropriate shape
            test_corr_mat = np.zeros((X.shape[0], len(all_indices)))
            for reference_idx in range(len(all_indices)):
                test_corr_mat[:,reference_idx] = X.corrwith(self.X_.loc[all_indices[reference_idx]], axis=1, method=self.corr_type).values
            self.test_corr_mat = pd.DataFrame(test_corr_mat)
        else:
            self.test_corr_mat = X[all_indices]
        
        self.test_corr_mat = self.test_corr_mat.values
        
        
        # allocate memory 
        ytest = np.zeros(self.test_corr_mat.shape[0])
        
        # we need to perform estimation procedure for every test instance
        for test_instance in range(X.shape[0]):
            # allocate memory for class claims (eg. a 1 on 3rd and 5th position means that class 3 and 5 claim this sample)
            y_test_instance = np.zeros(len(self.classes_))
            # pdf matrix
            pdf_test_instance = np.zeros((len(self.classes_), len(self.possible_values)))
            
            # first step: parallel hypothesis tests
            for class_k_idx in range(len(self.classes_)):
                k_class = self.classes_[class_k_idx] # class 1
                
                training_indices_k = np.in1d(all_indices, self.indices_per_class[class_k_idx]) # training indices with class k
                
                f_k_star = self.test_corr_mat[test_instance, training_indices_k].ravel() # distribution of correlations between trainig instances from k and test instances
                # exclude NaNs, for possible external bias models
                f_k_star = f_k_star[np.nonzero(~np.isnan(f_k_star))]
                # fit KDE
                kde_k_star = KernelDensity(bandwidth=0.05, kernel="gaussian")
                kde_k_star.fit(f_k_star[:, np.newaxis])
                
                # predict PDF from self.possible_values and store away
                distribution_k_star = np.exp(kde_k_star.score_samples(self.possible_values))
                pdf_test_instance[class_k_idx, :] = distribution_k_star
                
                # KL distance
                d_fkstar_fkk= entropy(distribution_k_star, self.distribution_matrix[class_k_idx, class_k_idx, :])
                
                # check, whether other classes
                j_dists = []
                # iterate over all other classes
                for class_j_idx in range(len(self.classes_)):
                    j_class = self.classes_[class_j_idx] # class 2
                    if k_class==j_class:
                        continue
                    # compute distance from this distribution to cross-correlation-distribution
                    d_fkstar_fjk = entropy(distribution_k_star, self.distribution_matrix[class_k_idx, class_j_idx, :])
                    # append to distances
                    j_dists.append(d_fkstar_fjk)
                    
                j_dists = np.array(j_dists) # make numpy-array for elementwise comparisons
                if np.all(j_dists>d_fkstar_fkk):
                    y_test_instance[class_k_idx] = 1
            # if exactly one class claims the instance --> everything is fine
            if np.sum(y_test_instance)==1:
                ytest[test_instance]=self.classes_[np.nonzero(y_test_instance==1)]
            else: # now the weighted sum
                if np.all(y_test_instance==0): # if no classes claim the instance, all classes claim it
                    y_test_instance[:] = 1
                
                min_val = 1e9 # value of current minimal weighted KL distance. Initial value is 1e9
                min_class = -1 # current best class. Initial value is -1.
                # compute weighted KL distance and go for 
                for class_k_idx in range(len(self.classes_)):
                    summe = 0.0 # weighted KL distance for this class k
                    k_class = self.classes_[class_k_idx] # class k
                    training_indices_k = np.nonzero(self.y_==k_class)[0] # training indices from class k
                    
                    # compute sum over j=1...len(self.classes_)
                    for class_j_idx in range(len(self.classes_)):
                        wk = 1 if y_test_instance[class_j_idx]==1 else 0 # weight is 1 if classes contribute
                        
                        d_fkstar_fjk = entropy(pdf_test_instance[class_j_idx,:], self.distribution_matrix[class_k_idx, class_j_idx, :])
                        summe += wk*d_fkstar_fjk
                    if summe<min_val:
                        min_val=summe
                        min_class=k_class
                ytest[test_instance]=min_class
        
        return np.asarray(ytest)