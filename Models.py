import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# So for some reason SKLearn does not store the number of training samples in their classifiers after training. 
# I mean, usually one knows this value and its pretty useless so that makes sense. However, I did not find a way to 
# convince cross_validate to give me the training sizes of each individual bucket and thus I thought this would be 
# the best option.
class RandomForestClassifierWithSampleSize(RandomForestClassifier):

    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]
        super().fit(X, y, sample_weight)

class DataAugmentationRandomForestClassifierWithSampleSize(RandomForestClassifier):

    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]
        # use RF here?
        rf = RandomForestClassifierWithSampleSize(n_estimators=128,max_leaf_nodes=self.max_leaf_nodes, random_state=12345)
        
        rf.fit(X,y)
        #XNew = self.generate_new_training_data(dt, X, 50)
        # #tmp = [X]
        tmp = []
        for _ in range(10):
            #tmp.append( X + np.random.normal(loc = 0.0, scale = 0.1, size = X.shape) )
            tmp.append( X + np.random.normal(loc = 0.0, scale = 0.01, size = X.shape) )
        XNew = np.concatenate(tmp)
        YNew = rf.predict_proba(XNew).argmax(axis=1)

        # TODO igoring sample_weight for now
        super().fit(XNew, YNew, sample_weight=None)

class DataAugmentationDecisionTreeClassifierWithSampleSize(RandomForestClassifier):


    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]
        # use RF here?
        dt = DecisionTreeClassifierWithSampleSize(max_leaf_nodes=self.max_leaf_nodes, random_state=12345)
        
        dt.fit(X,y)
        #XNew = self.generate_new_training_data(dt, X, 50)
        # #tmp = [X]
        tmp = []
        for _ in range(10):
            #tmp.append( X + np.random.normal(loc = 0.0, scale = 0.1, size = X.shape) )
            tmp.append( X + np.random.normal(loc = 0.0, scale = 0.01, size = X.shape) )
        XNew = np.concatenate(tmp)
        YNew = dt.predict_proba(XNew).argmax(axis=1)

        # TODO igoring sample_weight for now
        super().fit(XNew, YNew, sample_weight=None)

class DecisionTreeClassifierWithSampleSize(DecisionTreeClassifier):

    def fit(self, X, y, sample_weight = None):
        self.n_samples_ = X.shape[0]
        super().fit(X, y, sample_weight)