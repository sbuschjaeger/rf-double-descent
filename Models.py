import copy
import numpy as np

from joblib import Parallel,delayed

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

"""
So for some reason SKLearn does not store the number of training samples in their classifiers after training. I mean, usually one knows this value and its pretty useless so that makes sense. However, I did not find a way to convince cross_validate to give me the training sizes of each individual bucket and thus I thought this would be the best option.
"""

class BaggingClassifierWithSampleSize(BaggingClassifier):
    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]
        super().fit(X, y, sample_weight)

class RandomForestClassifierWithSampleSize(RandomForestClassifier):
    """RandomForestClassifierWithSampleSize. This is just a wrapper of RandomForestClassifier which also stores the sample size during fit.
    """
    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]
        super().fit(X, y, sample_weight)

class DecisionTreeClassifierWithSampleSize(DecisionTreeClassifier):
    """DecisionTreeClassifierWithSampleSize. This is just a wrapper of DecisionTreeClassifier which also stores the sample size during fit.
    """
    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]
        super().fit(X, y, sample_weight)

class DataAugmentationRandomForestClassifierWithSampleSize(RandomForestClassifier):
    """DataAugmentationRandomForestClassifierWithSampleSize. This classifier first trains a regular DT with max_leaf_nodes nodes. Then the decision boundary of this classifier is sampled using newley generated training data which was augmentated from the original one using gaussian noise. The new training data is 10 times larger than the orignal one. Then a regular random forest is fitted on that new data.

    TODO: Properly introduce  loc / scale / max_leaf_nodes (for the DT) as parameters
    """
    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]

        dt = DecisionTreeClassifierWithSampleSize(max_leaf_nodes=self.max_leaf_nodes, random_state=12345)
        
        dt.fit(X,y, sample_weight)
        tmp = []
        for _ in range(10):
            tmp.append( X + np.random.normal(loc = 0.0, scale = 0.01, size = X.shape) )
        XNew = np.concatenate(tmp)
        YNew = dt.predict_proba(XNew).argmax(axis=1)

        super().fit(XNew, YNew, sample_weight=None)

class DataAugmentationDecisionTreeClassifierWithSampleSize(DecisionTreeClassifierWithSampleSize):
    """DataAugmentationDecisionTreeClassifierWithSampleSize. This classifier first trains a regular RF with max_leaf_nodes nodes and 256 trees. Then the decision boundary of this classifier is sampled using newley generated training data which was augmentated from the original one using gaussian noise. The new training data is 10 times larger than the orignal one. Then a regular decision tree is fitted on that new data.

    TODO: Properly introduce  loc / scale / max_leaf_nodes / n_estimators (for the RF) as parameters
    """
    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]

        rf = RandomForestClassifier(n_estimators=256, max_leaf_nodes=self.max_leaf_nodes, random_state=12345)
        
        rf.fit(X,y, sample_weight)
        tmp = []
        for _ in range(10):
            tmp.append( X + np.random.normal(loc = 0.0, scale = 0.01, size = X.shape) )
        XNew = np.concatenate(tmp)
        YNew = rf.predict_proba(XNew).argmax(axis=1)

        super().fit(XNew, YNew, sample_weight=None)

class HomogeneousForest(BaseEstimator,ClassifierMixin):
    """A HomogeneousForest. This forest receives a DecisionTreeClassifier as base_dt, fits it to the training data and then copies it n_estimator times. Hence it should have the same performance as a single DecisionTreeClassifier. 
    This classifier is used in combination with NCForest to have a base_forest without any diversity so that NCForest can enforce diversity if necessary.
    """
    def __init__(self, base_dt, n_estimators, n_jobs = None, bootstrap = True):
        self.base_dt = base_dt
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap

    def fit(self, X, y, sample_weight = None):
        
        def _fit(e, X, y, sample_weight = None, bootstrap = True):
            n_samples = X.shape[0]
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight

            if bootstrap:
                indices = np.random.randint(0, n_samples, n_samples)
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            
            e.fit(X, y, sample_weight=curr_sample_weight)
            return e

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store some statistics
        self.classes_ = unique_labels(y)
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(self.classes_)

        self.estimators_ = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed( _fit ) (copy.deepcopy(self.base_dt), X, y, sample_weight, self.bootstrap) for _ in range(self.n_estimators)
        )
        # self.estimators_ = []
        # for _ in range(self.n_estimators):
        #     self.estimators_.append(copy.deepcopy(self.base_dt))

    def predict(self, X):
        """ Predict classes using.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted classes. 

        """
        proba = self.predict_proba(X)
        return self.classes_.take(proba.argmax(axis=1), axis=0)

    def predict_proba(self, X):
        """ Predict class probabilities.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The samples to be predicted.

        Returns
        -------
        y : array, shape (n_samples,C)
            The predicted class probabilities. 
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        all_proba = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed( lambda e, X: e.predict_proba(X) ) (e, X) for e in self.estimators_
        )

        all_proba = np.array(all_proba)
        return all_proba.mean(axis=0)