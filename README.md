# ACF

This repository provides the source code for the ACF algorithm
(ACF = Average Correlations as Features), all source code associated with the
corresponding study, unit tests for the code, as well as the generated data.
 
## Purpose

A common pitfall associated with many modern omics-technologies is the strong
prevalence of missing values. Imputation methods have been found to generalize
badly across datasets and can even introduce false positives. Another strategy
is listwise deletion, which refers to the deletion of all features with at least one
missing value. This leads to severe loss of imputation and therefore hampers
doownstream analyses.

We therefore advocate the use of pairwise metrics and focus on pairwise
correlations. The ACF-algorithm uses average correlations to the training instances
per class to classify unseen data.

## Installation and Testing


We require the latest python version, as well as up-to-date versions of
`numpy`,`pandas`,`optuna`,`scikit-learn`,`pytest`,`matplotlib` and `seaborn`. Apart from that,
no specific installation procedure is required - the repository can simply be downloaded
to your local system.

The installation can be tested by executing python in the `ACF/Python` directory and
calling

    import pytest
    pytest.main(["-v"])


## Options and Parameters

All parameters of ACF are properly described in the source code. For completeness, we also report them here.

* `baseline_classifier`: A scikit-learn compatible classifier that is used internally for predicting test instances. This parameter MUST by set by the user.
* `corr_type`: One in {'pearson', 'kendall', 'spearman'} or callable. Determines the type of correlation used. The default is 'pearson'.
* `strategy`: One in {'mean', 'median'}. Determines, whether to use mean or median correlations. The default is 'mean'.
* `variant`: One in {None, 'F-ACF', 'B-ACF'}. Whether to use default ACF (default), the fast implementation based on reference samples ('F-ACF') or the version that allows to include biases ('B-ACF'). The default is None.
* `n_ref`: Number of reference samples to use per class when selecting variant='F-ACF'. Ignored otherwise. Instances are selected at random.
* `precomputed`: If true, the argument X of fit() and predict() is expected to be the precomputed correlation matrix between training-/testing- and reference-instances. np.NaN is allowed (e.g. to mask biased correlations such as self-correlations). This particularly useful to reduce the computational overhead in crossvalidation-procedures. The default is False.
* `scale_Corrmat`: One in {'Standard','MinMax',None}. Whether (and how) to scale the rows of the correlation matrix prior to averaging. The default is None.
* `normalize_AC`: One in {'l1','l2',None}. Whether (and how) to normalize the average correlations of each sample across all classes.

## Example

The code below creates a dataset with a noisy correlation matrix and splits it into
train- and test set. 

    import DGP # for creation of the dataset
    import ACF # ACF algorithm
    from sklearn.svm import SVC # baseline classifier
    from sklearn.model_selection import train_test_split # for train test split
    from sklearn.metrics import f1_score
    import numpy as np
    
    
    # create dataset with 60 instances from three classes and a typical correlation matrix
    X,y = DGP.createDataWithBlockCorrelation([60,60,60], np.array([[1.0,0.9,0.6],[0.9,1.0,0.8],[0.6,0.8,1.0]]), 0.8, 2.0)
    
    # split into train- and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # baseline for ACF
    baseline_classifier = SVC(kernel="rbf") 
    
    # create ACF instance
    acf = ACF.ACFClassifier(baseline_classifier, strategy="mean")
    
    # Fit
    acf.fit(X_train, y_train)
    
    # predict
    y_predicted = acf.predict(X_test)
    
    # f1-score
    print(f1_score(y_test, y_predicted, average="macro"))
    

## Repository Structure

All source code required to reproduce the results from our study can be found
in  `ACF/ACF/Python/`. The most relevant files are briefly highlighted below:
* ACF algorithm
    * ACF.py -- the ACF algorithm
    * test_ACF.py -- test cases for ACF
* KNN and DBC Implementation
    *  CustomKNN.py -- a custom implementation of the KNN classifier
    *  DBC.py -- a custom implementation of the DBC classifier
    *  test_CustomKNN.py -- test cases for KNN
    *  test_DBC.py -- test cases for DBC
* Data Generating Process
    * DGP.py -- the data generating process developed for the study
    * test_DGP.py -- test cases for the DGP
* Scripts for figures in the study
    * STUDY_Characterization_DGP.py -- Produces the characterization of the parameters in the data generating process
    * STUDY_Comparison_ACF_FACF.py -- Produces the comparison between ACF and F-ACF
    * STUDY_Comparison_ACF_KNN_Imbalance.py -- Produces the comparison between ACF, DBC and KNN with respect to class imbalance
    * STUDY_Comparison_ACF_KNN_Noise.py -- Produces the comparison between ACF, DBC and KNN with respect to noise on the correlation matrix
    * STUDY_Comparison_ACF_KNN_TrainSize.py -- Produces the comparison of ACF, DBC and KNN with respect to the number of training instances
    * STUDY_Comparison_FACF_KNN_TimeComplexity.py -- Produces the comparison of the prediction time complexity of F-ACF, F-DBC and KNN with respect to the number of training instances
    * STUDY_Comparison_proteomics.py -- Compares ACF/KNN/Listwise-deletion on three datasets from scRNA-seq
    * STUDY_Comparison_scRNAseq.py -- Compares ACF/KNN/Listwise-deletion on three datasets from proteomic datasets
    * STUDY_Comparison_Petralia_Covariable.py -- Demonstrates ACF with and without additional covariables
    * STUDY_FigureClassDistributions.py -- Produces the figures of the class distributions from the considered datasets.
    * STUDY_analysis_dbc_xin_dataset.py -- Produces figures to explain the concept of DBC on the dataset by Xin et al.
    * STUDY_variable_importance_visualization.py -- Measures the variable importance for each class on all datasets.
    
Executing any STUDY_....py file will directly reproduce the corresponding results from
the study. (Beware: Execution may take several days.)

The folder `ACF/ACF/Python/Results` contains all plots from the study, as well 
as the script used to create those plots. The corresponding folder `ACF/ACF/Python/Results/raw`
contains the raw data generated by our source code.
