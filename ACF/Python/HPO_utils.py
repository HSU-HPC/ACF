# -*- coding: utf-8 -*-
"""
Utility functions for Hyperparameter Optimization and Validation

@author: Yannis Schumann
"""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from joblib import Parallel, delayed
import numpy as np
import ACF
import CustomKNN
import optuna
import DBC

def objective_KNN(trial, X_train, y_train, n_splits, test_size, metric, oversampling):
    """
    Objective function for hyperparameter optimization of KNN on correlation matrix.

    Parameters
    ----------
    trial : optuna trial
        Trial for study.
    X_train : np.ndarray
        Array of shape n_train x n_train with the correlations of all training observations
    y_train : np.ndarray
        Corresponding training labels.
    n_splits : int
        Number of splits for the StratifiedShuffleSplit that is used to compute the metric.
    test_size : float
        Size of stratified test splits in percent of the training set.
    metric : callable
        The metric to optimize in HPO, eg. F1-score.
    oversampling : bool
        Whether to use random oversampling

    Returns
    -------
    perfmet : float
        f1-score with that hyperparameter combination.

    """
    # maximum number of nearest neighbors that can be considered in this split
    max_neighbors = int((1-test_size)*X_train.shape[0])-1
    
    # draw hyperparameters
    n_neighbors = trial.suggest_int("n_neighbors", 1,max_neighbors)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    
    # compute metric on stratified shuffle splits
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    
    pred = [] # for the predictions
    true = [] # for the true class labels
    # iterate over splits
    for train_index, test_index in ss.split(X_train, y_train):
        # NOTE: X_train is the correlation matrix! So we need to select suitable subsets of that
        x_t, x_v = X_train[np.ix_(train_index, train_index)], X_train[np.ix_(test_index, train_index)]
        y_t, y_v = y_train[train_index], y_train[test_index]
        # build knn classifier
        knn = CustomKNN.CustomKNN(n_neighbors=n_neighbors, weights = weights, oversampling=oversampling, precomputed=True)
        knn.fit(x_t, y_t)
        # extend lists of class labels
        pred.extend(knn.predict(x_v))
        true.extend(y_v)
    #with warnings.catch_warnings():
    #warnings.filterwarnings("ignore", category=RuntimeWarning)
    perfmet = metric(true, pred) # compute user-selected metric
    return perfmet


def objective_RemainingGenes(trial, X_train, y_train, n_splits, test_size, metric, base_clf=None):
    """
    Objective function for hyperparameter optimization on all remaining genes.

    Parameters
    ----------
    trial : optuna trial
        Trial for study.
    X_train : np.ndarray
        Array of shape n_train x n_train with the correlations of all training observations
    y_train : np.ndarray
        Corresponding training labels.
    n_splits : int
        Number of splits for the StratifiedShuffleSplit that is used to compute the metric.
    test_size : float
        Size of stratified test splits in percent of the training set.
    metric : callable
        The metric to optimize in HPO, eg. F1-score.
    base_clf : str or None. optional
        None indicates that the baseline classifier should be chosen from ["SVC", "RF", "Ridge"] as
        part of the hyperparameter optimization. Otherwise, one of the aforementioned algorithms
        are permitted. The default is None.

    Returns
    -------
    perfmet : float
        f1-score with that hyperparameter combination.

    """
    # scaler to scale the data
    scaler = StandardScaler()
    # scaling    
    X_train_scaled = scaler.fit_transform(X_train)
    
    # which classifier to use
    if base_clf==None:
        baseline_classifier = trial.suggest_categorical("baseline_classifier",["SVC","RF","Ridge"])
    else:
        baseline_classifier = base_clf
    
    
    # same hyperparameters as for ACF
    # if SVC
    if baseline_classifier == "SVC":
        kernel = trial.suggest_categorical("kernel", ["linear","rbf"]) # suggest kernel
        C = trial.suggest_loguniform("C",5e-3,5e2) # suggest C for regularization
        gamma = trial.suggest_categorical("gamma",["scale","auto"]) # suggest gamme
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None]) # and specify class weights
        model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight) # build model
    #if random forest
    elif baseline_classifier == "RF":
        n_estimators = trial.suggest_int("n_estimators", 80,300) # no. of estimators --> 
        max_depth = trial.suggest_int("max_depth",2,40) # max. depth of 40 should be sufficient
        max_features = trial.suggest_int("max_features", 1, X_train.shape[1]) # how many features to consider
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None]) # specify class weights
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features = max_features, class_weight=class_weight) # build models
    # if ridge classifier
    elif baseline_classifier == "Ridge":
        alpha = trial.suggest_loguniform("alpha", 1e-3,1e4) # regularization
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None]) # suggest class weights
        model = RidgeClassifier(alpha=alpha, class_weight=class_weight) # build model
    
    # compute metric on stratified shuffle splits
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    
    pred = [] # for the predictions
    true = [] # for the true class labels
    # iterate over the splits
    for train_index, test_index in ss.split(X_train, y_train):
        # NOTE: here, X_train_scaled consists of genes and is not a correlation matrix. Therefore, we everything (!) along axis 1
        x_t, x_v = X_train_scaled[train_index], X_train_scaled[test_index]
        y_t, y_v = y_train[train_index], y_train[test_index]
        model.fit(x_t,y_t)
        pred.extend(model.predict(x_v))
        true.extend(y_v)
        
    perfmet =  metric(true, pred)
    return perfmet


def objective_ACF(trial, X_train, y_train, n_splits, test_size, metric, baseline_classifier, variant, n_ref, strategy="mean", X_c_train = []):
    """
    Objective function for hyperparameter optimization of ACF on correlation matrix.

    Parameters
    ----------
    trial : optuna trial
        Trial for study.
    X_train : np.ndarray
        Array of shape n_train x n_train with the correlations of all training observations
    y_train : np.ndarray
        Corresponding training labels.
    n_splits : int
        Number of splits for the StratifiedShuffleSplit that is used to compute the metric.
    test_size : float
        Size of stratified test splits in percent of the training set.
    metric : callable
        The metric to optimize in HPO, eg. F1-score.
    baseline_classifier : str in {'SVC', 'RF', 'Ridge'}
        Baseline classifier selected here.
    variant : str in {None, 'F-ACF'}
        ACF-variant to use. B-ACF is prohibited. Biased elements should be masked by the user.
    n_ref : int
        Number of reference samples to use if F-ACF.
    strategy : str, optional
        Strategy for ACF. One in ("median", "mean", "kld"). The default is mean.
    X_c_train : np.ndarray, optional
        Array of shape n_observations x n_features. Additional covariable to consider for ACF. The default is []. (Then it does not contribute.)

    Returns
    -------
    perfmet : float
        f1-score with that hyperparameter combination.

    """
    
    # scaling of correlation matrix prior to averaging
    scale_Corrmat = trial.suggest_categorical("scale_Corrmat", [None, "MinMax", "Standard"])
    # normalization of average correlations
    normalize_AC = trial.suggest_categorical("normalize_AC", [None, "l1", "l2"])
    
    # further hyperparameters need to be determined in dependence of selected baseline classifier
    if baseline_classifier == "SVC":
        kernel = trial.suggest_categorical("kernel", ["linear","rbf"]) # kernel
        C = trial.suggest_loguniform("C",5e-3,5e2) # regularization
        gamma = trial.suggest_categorical("gamma",["scale","auto"])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight) # build model
    elif baseline_classifier == "RF":
        n_estimators = trial.suggest_int("n_estimators", 80,300) # 300 trees should be sufficient.
        max_depth = trial.suggest_int("max_depth",2,40) # as well as a depth of 40
        max_features = trial.suggest_int("max_features", 1, len(np.unique(y_train))) # at most as many features as classes possible
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features = max_features, class_weight=class_weight) # build model
    elif baseline_classifier == "Ridge":
        alpha = trial.suggest_loguniform("alpha", 1e-3,1e4) # regularization
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        model = RidgeClassifier(alpha=alpha, class_weight=class_weight) # build model
        
    # compute metric on stratified shuffle splits
    ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    
    pred = [] # for the predictions
    true = [] # for the true class labels
    # iterate over splits
    for train_index, test_index in ss.split(X_train, y_train):
        # NOTE: X_train is the correlation matrix! So we need to select suitable subsets of that
        x_t, x_v = X_train[np.ix_(train_index, train_index)], X_train[np.ix_(test_index, train_index)]
        y_t, y_v = y_train[train_index], y_train[test_index]
        # build acf classifier
        acf = ACF.ACFClassifier(model, strategy=strategy, variant=variant, n_ref = n_ref, precomputed = True, scale_Corrmat = scale_Corrmat, normalize_AC = normalize_AC)
        # whether we need to consider additional covariables
        if len(X_c_train)!=0:
            x_c_t, x_c_v = X_c_train[train_index,:], X_c_train[test_index,:]
            acf.fit(x_t, y_t, additional_covariable=x_c_t)
            pred.extend(acf.predict(x_v, additional_covariable=x_c_v))
        else:
            acf.fit(x_t, y_t)
            pred.extend(acf.predict(x_v))
        
        true.extend(y_v)
    #with warnings.catch_warnings():
    #warnings.filterwarnings("ignore", category=RuntimeWarning)
    perfmet = metric(true, pred) # compute user-selected metric
    return perfmet


def characterize_RemainingGenes(X, y, n_iter=60, n_splits=10,test_size=0.05, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),return_individual=False, base_clf=None):
    """
    Computes and returns the performance of a default approach on all remaining genes (listwise deletion)
    in 10-fold stratified cross validation on a dataset.
    If no genes (aka features) remain after listwise deletion, random class assignment is used.

    Parameters
    ----------
    X : np.ndarray
        The dataset of shape n_observations x n_features. NA is allowed
    y : np.ndarray
        The corresponding class labels.
    n_iter : int, optional
        Number of hyperparameter optimizations to perform. The default is 60.
    n_splits : int, optional
        Number of StratifiedShuffleSplits used to compute the performance in the hyperparameter optimization. The default is 10.
    test_size : float, optional
        Size of test splits in StratifiedShuffleSplits in percent of split. The default is 0.05.
    metric : callable, optional
        The metric to use. The default is macro-averaged f1-score.
    return_individual : bool, optional
        Whether to return the scores on the individual splits. The default is False.
    base_clf : str or None. optional
        None indicates that the baseline classifier should be chosen from ["SVC", "RF", "Ridge"] as
        part of the hyperparameter optimization. Otherwise, one of the aforementioned algorithms
        are permitted. The default is None

    Returns
    -------
    float
        Total score on the dataset.
    list of floats (optional, if return_individual=True)
        Scores on the individual splits
    list of ints (optional, if return_individual=True)
        Size of training splits
    list of ints (optional, if return_individual=True)
        Size of testing splits

    """
    
    def parallel_studies(train_idx, test_idx):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # select training and testing sets
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # if no genes present (all deleted because of missing values), we need to assign classes randomly
        if X_train.shape[1]==0:
            classes = np.unique(y_train)
            rng = np.random.default_rng()
            y_predicted = rng.choice(classes, size=len(y_test))
            return [y_predicted, y_test]
        
        # perform hyperparameter optimization
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: objective_RemainingGenes(trial, X_train, y_train, n_splits, test_size, metric=metric, base_clf=base_clf), n_trials=n_iter)
        
        # scale X_train and X_test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # selected hyperparameters
        best = study.best_params
        #the selected baseline classifier
        if base_clf == None:
            baseline_classifier = best["baseline_classifier"]
        else:
            baseline_classifier = base_clf
        
        if baseline_classifier == "SVC":
            model = SVC(kernel=best["kernel"],C=best["C"],class_weight=best["class_weight"], gamma=best["gamma"]) # build model
        elif baseline_classifier == "RF":
            model = RandomForestClassifier(n_estimators=best["n_estimators"], max_depth=best["max_depth"], max_features = best["max_features"], class_weight=best["class_weight"]) # build model
        elif baseline_classifier == "Ridge":
            model = RidgeClassifier(alpha=best["alpha"], class_weight=best["class_weight"]) # build model
        
        # fit and predict
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        # return results and true predictions
        return [pred, y_test]
    
    # find positions of NAs
    not_missing = ~np.isnan(X)
    # find columns without any NA
    na_cols = np.all(not_missing, axis=0)
    # reduce X to all columns without NA
    X = X[:, na_cols]
    
    
    # 10-fold Stratified K-Fold
    
    skf = StratifiedKFold(n_splits=10)
    results = Parallel(n_jobs=-1)(delayed(parallel_studies)(train_idx,test_idx) for train_idx,test_idx in skf.split(X, y))
    
    # lists for the results from the individual runs
    
    predictions = []
    true_values = []
    n_train = []
    n_test = []
    individual_metric = []
    
    for r in results:
        predictions.append(r[0])
        true_values.append(r[1])
        individual_metric.append(metric(r[1],r[0]))
        n_train.append(len(y)-len(r[0]))
        n_test.append(len(r[0]))
    
    # make it all one list
    predictions = np.concatenate(predictions)
    true_values = np.concatenate(true_values)
    
    if not return_individual:
        return metric(true_values, predictions)
    else:
        return metric(true_values, predictions), individual_metric, n_train, n_test



def characterize_KNN(corr, y, n_iter=60, n_splits=10, test_size=0.05, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), oversampling=False, return_individual=False, return_neighbors = False):
    """
    Computes and returns the performance of KNN in 10-fold stratified cross validation
    on a dataset with specified correlation matrix.
    

    Parameters
    ----------
    corr : np.ndarray
        The dataset of shape n_observations x n_observations.
    y : np.ndarray
        The corresponding class labels.
    n_iter : int, optional
        Number of hyperparameter optimizations to perform. The default is 60.
    n_splits : int, optional
        Number of StratifiedShuffleSplits used to compute the performance in the hyperparameter optimization. The default is 10.
    test_size : float, optional
        Size of test splits in StratifiedShuffleSplits in percent of split. The default is 0.05.
    metric : callable, optional
        The metric to use. The default is macro-averaged f1-score.
    oversampling : bool, optional
        Whether to use random oversampling. The default is False.
    return_individual : bool, optional
        Whether to return the scores on the individual splits. The default is False.
    return_neighbors : bool, optional
        Whether to return the average number of nearest neighbors selected

    Returns
    -------
    float
        Total score on the dataset.
    list of floats (optional, if return_individual=True)
        Scores on the individual splits
    list of ints (optional, if return_individual=True)
        Size of training splits
    list of ints (optional, if return_individual=True)
        Size of testing splits

    """
    def parallel_studies(train_idx, test_idx):
        optuna.logging.set_verbosity(optuna.logging.WARNING) # silence output
        
        # select training- und test parts of the precomputed correlation matrix
        corr_train = corr[np.ix_(train_idx, train_idx)]
        corr_test = corr[np.ix_(test_idx, train_idx)]
        
        # select corresponding class labels
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # perform hyperparameter optimization
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: objective_KNN(trial, corr_train, y_train, n_splits, test_size, metric, oversampling), n_trials=n_iter)
        
        # selected parameters
        best = study.best_params
        
        # build KNN Classifier
        knn_classifier = CustomKNN.CustomKNN(n_neighbors=best["n_neighbors"], weights =best["weights"], oversampling=oversampling, precomputed=True)
        # fit and predict
        knn_classifier.fit(corr_train, y_train)
        y_predicted = knn_classifier.predict(corr_test)
        # return results for future evaluation, including the selected number of nearest neighbors
        return [y_predicted, y_test, best["n_neighbors"]]
    
    # 10-fold Stratified K-Fold
    
    skf = StratifiedKFold(n_splits=10)
    results = Parallel(n_jobs=-1)(delayed(parallel_studies)(train_idx,test_idx) for train_idx,test_idx in skf.split(corr, y))
    
    predictions = [] # individual predictions per split
    true_values = [] # correct class labels
    n_train = [] # number of training instances
    n_test = [] # number of test instances
    neighbors = [] # selected numbers of nearest neighbors
    individual_metric = [] # individual metric on that specific split
    
    # iterate over the results
    for r in results:
        predictions.append(r[0])
        true_values.append(r[1])
        individual_metric.append(metric(r[1],r[0]))
        n_train.append(len(y)-len(r[0]))
        n_test.append(len(r[0]))
        neighbors.append(r[2])
    predictions = np.concatenate(predictions) # make one list
    true_values = np.concatenate(true_values) # make one list
    average_neighbors = np.mean(neighbors)
    
    total_metric = metric(true_values, predictions) # metric on entire dataset
    
    if not return_individual:
        if not return_neighbors:
            return total_metric
        else:
            return total_metric, average_neighbors
    else:
        if not return_neighbors:
            return total_metric, individual_metric, n_train, n_test
        else:
            return total_metric, individual_metric, n_train, n_test, average_neighbors


def characterize_ACF(corr, y, n_iter=60, n_splits=10, test_size=0.05, baseline_classifier="SVC", metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), variant=None, n_ref=None, return_individual=False, timeout=None, strategy="mean", X_c = []):
    """
    Computes and returns the performance of ACF in 10-fold stratified cross validation
    on a dataset with specified correlation matrix.
    

    Parameters
    ----------
    corr : np.ndarray
        The dataset of shape n_observations x n_observations.
    y : np.ndarray
        The corresponding class labels.
    n_iter : int, optional
        Number of hyperparameter optimizations to perform. The default is 60.
    n_splits : int, optional
        Number of StratifiedShuffleSplits used to compute the performance in the hyperparameter optimization. The default is 10.
    test_size : float, optional
        Size of test splits in StratifiedShuffleSplits in percent of split. The default is 0.05.
    baseline_classifier : str in {'SVC','RF','Ridge'}, optional
        The baseline classifier to use. The default is "SVC".
    metric : callable, optional
        The metric to use. The default is macro-averaged f1-score.
    variant : {None, 'F-ACF'}, optional
        ACF-variant to use. B-ACF does not work here. (Instead, the biased elements need to be masked already.) The default is None.
    n_ref : int, optional
        Number of reference instances to use if F-ACF was selected. The default is None.
    return_individual : bool, optional
        Whether to return the scores on the individual splits. The default is False.
    strategy: str, optional.
        Strategy for ACF. One in ("median", "mean", "kld"). The default is mean.
    X_c : np.ndarray, optional
        Array of shape n_observations x n_features. Additional covariable to consider for ACF. The default is []. (Then it does not contribute.)
        
    Returns
    -------
    float
        Total score on the dataset.
    list of floats (optional, if return_individual=True)
        Scores on the individual splits
    list of ints (optional, if return_individual=True)
        Size of training splits
    list of ints (optional, if return_individual=True)
        Size of testing splits

    """
    def parallel_studies(train_idx, test_idx):
        optuna.logging.set_verbosity(optuna.logging.WARNING) # silence output
        
        # select training- und test parts of the precomputed correlation matrix
        corr_train = corr[np.ix_(train_idx, train_idx)]
        corr_test = corr[np.ix_(test_idx, train_idx)]
        # select training- and test parts of the additiona covariable (optional, since it may be empty)
        if len(X_c)!=0:
            X_c_train = X_c[train_idx,:]
            X_c_test = X_c[test_idx,:]
        
        # select corresponding class labels
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # perform hyperparameter optimization
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        if len(X_c)!=0: # if we have a covariable
            study.optimize(lambda trial: objective_ACF(trial, corr_train, y_train, n_splits, test_size, metric, baseline_classifier, variant, n_ref, X_c_train=X_c_train), n_trials=n_iter)
        else: # if we don't havve a covariable
            study.optimize(lambda trial: objective_ACF(trial, corr_train, y_train, n_splits, test_size, metric, baseline_classifier, variant, n_ref), n_trials=n_iter)
        
        # selected parameters
        best = study.best_params
        
        # build according baseline classifier
        if baseline_classifier == "SVC":
            model = SVC(kernel=best["kernel"],C=best["C"],class_weight=best["class_weight"], gamma=best["gamma"])
        elif baseline_classifier == "RF":
            model = RandomForestClassifier(n_estimators=best["n_estimators"], max_depth=best["max_depth"], max_features = best["max_features"], class_weight=best["class_weight"])
        elif baseline_classifier == "Ridge":
            model = RidgeClassifier(alpha=best["alpha"], class_weight=best["class_weight"])
        
        # build final ACF Classifier
        acf_classifier = ACF.ACFClassifier(model, strategy=strategy, variant=variant, n_ref = n_ref, precomputed = True, scale_Corrmat = best["scale_Corrmat"], normalize_AC = best["normalize_AC"])
        # fit and predict
        if len(X_c)!=0:
            acf_classifier.fit(corr_train, y_train, additional_covariable=X_c_train)
            y_predicted = acf_classifier.predict(corr_test, y_test, additional_covariable=X_c_test)
        else:
            acf_classifier.fit(corr_train, y_train)
            y_predicted = acf_classifier.predict(corr_test, y_test)
            
        # return results for future evaluation
        return [y_predicted, y_test]
    
    # 10-fold Stratified K-Fold
    
    skf = StratifiedKFold(n_splits=10)
    try:
        results = Parallel(n_jobs=-1, timeout=timeout)(delayed(parallel_studies)(train_idx,test_idx) for train_idx,test_idx in skf.split(corr, y))
    except:
        return np.NaN
    
    predictions = [] # individual predictions per split
    true_values = [] # correct class labels
    n_train = [] # number of training instances
    n_test = [] # number of test instances
    individual_metric = [] # individual metric on that specific split
    
    # iterate over the results
    for r in results:
        predictions.append(r[0])
        true_values.append(r[1])
        individual_metric.append(metric(r[1],r[0]))
        n_train.append(len(y)-len(r[0]))
        n_test.append(len(r[0]))
    predictions = np.concatenate(predictions) # make one list
    true_values = np.concatenate(true_values) # make one list
    
    total_metric = metric(true_values, predictions) # metric on entire dataset
    
    if not return_individual:
        return total_metric
    else:
        return total_metric, individual_metric, n_train, n_test


def characterize_DBC(corr, y, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"), return_individual = False):
    """
    Computes and returns the performance of DBC (Wei et al) in 10-fold stratified cross validation
    on a dataset with specified correlation matrix.

    Parameters
    ----------
    corr : np.ndarray
        The dataset of shape n_observations x n_observations.
    y : np.ndarray
        The corresponding class labels.
    metric : callable, optional
        The metric to use. The default is macro-averaged f1-score.
    return_individual : bool, optional
        whether to return the metrics from the individual splits

    Returns
    -------
    float
        Total score on the dataset.

    """
    
    # convert classes to floats
    y = OrdinalEncoder().fit_transform(y.reshape(-1,1)).ravel()
    predictions = [] # list for dbc-predictions
    true_values = [] # list for the true classes
    
    skf = StratifiedKFold(n_splits=10) # ten fold stratified CV
    individual_metric = []
    
    # iterate over 
    for train_idx,test_idx in skf.split(corr, y):
        X_train, y_train = corr[np.ix_(train_idx, train_idx)], y[train_idx] # train set for fitting distributions
        X_test, y_test = corr[np.ix_(test_idx, train_idx)], y[test_idx] # test set
        dbc = DBC.DBCClassifier(precomputed=True) # instance of dbc
        dbc.fit(X_train, y_train)
        ydbc = list(dbc.predict(X_test))
        predictions.extend(ydbc)
        true_values.extend(list(y_test))
        individual_metric.append(metric(list(y_test), ydbc))
    if return_individual:
        return metric(true_values, predictions), individual_metric
    else:
        return metric(true_values, predictions)