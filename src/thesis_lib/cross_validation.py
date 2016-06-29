import numpy as np
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.externals.joblib import delayed, Parallel
from sklearn.cross_validation import StratifiedKFold


__all__ = ['cross_val_multiple_scores']


def fit_and_predict(classifier, X, y, train_idx, test_idx):
    """
    Evaluate a score by cross-validation
    
    Parameters
    ----------
    classifier: object
        Classifier used to 'fit' and 'predict'
    X: array-like (2D)
        Contains the features
    y: array-like (1D)
        Contains the label
    train_idx: array-like (1D)
        Indices of training sample
    train_idx: array-like (1D)
        Indices of testing sample
    
    Return
    ------
    tuple of 2 array-like (1D)
        Ground_truth and predicted labels
    
    References
    ---------
    https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/cross_validation.py#L1441
    """
    classifier.fit(X[train_idx], y[train_idx])
    cur_pred = classifier.predict(X[test_idx])
    return (y[test_idx], cur_pred)


def get_true_and_pred_cross_val(classifier, X, y, cv_iter, n_jobs=-1):
    """
    Fit and predict by cross validation and in parallel.
    
    Parameters
    ----------
    classifier: class
        Classifier used to 'fit' and 'predict'
    X: array-like (2D)
        Contains the features
    y: array-like (1D)
        Contains the labels
    cv_iter: generator / iterator
        Cross validation splitting strategy giving the training and testing indices
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means 'all CPUs'.
    
    Returns
    -------
    list
        List of tuples containing:
        
        - first element: 1D array-like: ground truth labels
        - second element: 1D array-like: predicted labels
    
    References
    ----------
    https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/cross_validation.py#L1351
    """
    parallel = Parallel(n_jobs=n_jobs)
    ys = parallel(delayed(fit_and_predict)(clone(classifier), X, y,
                                           train_idx, test_idx)
                          for train_idx, test_idx in cv_iter)

    return ys


def cross_val_multiple_scores(classifier, X, y, n_folds=10, n_jobs=-1):
    """
    Evaluate multiple score (accuracy, precision, recall, f1, confusion matrix)
    by cross validation.
    
    Under the hood, it :
    
    - uses :class:`sklearn.cross_validation.StratifiedKFold` to split the data
    - fits and predicts the classifer on these splits
    - compute the mean for each metric's result
    
    Parameters
    ----------
    classifier: object
        Classifier used to 'fit' and 'predict'
    X: array-like (2D)
        Contains the features
    y: array-like (1D)
        Contains the labels
    n_folds: integer, optional
        Number of folds
    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means 'all CPUs'.
    
    Return
    ------
    dict
        Dictionary containing the different score
    
    References
    ----------
    http://stackoverflow.com/questions/23339523/sklearn-cross-validation-with-multiple-scores/36570701#36570701
    """
    cv_iter = StratifiedKFold(y, n_folds)
    
    predicted_ys = get_true_and_pred_cross_val(classifier, X, y, cv_iter)
    accuracy = map(lambda tp: accuracy_score(tp[0], tp[1]), predicted_ys)
    precision = map(lambda tp: precision_score(tp[0], tp[1], average='weighted'), predicted_ys)
    recall = map(lambda tp: recall_score(tp[0], tp[1], average='weighted'), predicted_ys)
    f1 = map(lambda tp: f1_score(tp[0], tp[1], average='weighted'), predicted_ys)
    cm = map(lambda tp: confusion_matrix(tp[0], tp[1]), predicted_ys)
    
    cm_sum = np.zeros((4, 4), np.float)
    for i in cm:
        cm_sum += i
    cm_sum /= n_folds
    
    return {
        'cv_accuracy': np.mean(np.fromiter(accuracy, np.float)),
        'cv_precision': np.mean(np.fromiter(precision, np.float)),
        'cv_recall': np.mean(np.fromiter(recall, np.float)),
        'cv_f1': np.mean(np.fromiter(f1, np.float)),
        'cv_confusion_matrix': cm_sum
    }

