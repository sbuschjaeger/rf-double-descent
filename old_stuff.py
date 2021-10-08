from sklearn.tree import _tree

# class DataAugmentationTreeClassifierWithSampleSize(DecisionTreeClassifier):

#     def __init__(self, 
#         n_estimators = 32,
#         n_jobs = None,
#         max_depth=None, 
#         criterion="gini",
#         splitter="best",
#         min_samples_split=2,
#         min_samples_leaf=1,
#         min_weight_fraction_leaf=0.,
#         max_features=None,
#         random_state=None,
#         max_leaf_nodes=None,
#         min_impurity_decrease=0.,
#         min_impurity_split=None,
#         class_weight=None,
#         ccp_alpha=0.0
#     ):
#         super().__init__(
#             criterion=criterion,
#             splitter=splitter,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             min_weight_fraction_leaf=min_weight_fraction_leaf,
#             max_features=max_features,
#             max_leaf_nodes=max_leaf_nodes,
#             class_weight=class_weight,
#             random_state=random_state,
#             min_impurity_decrease=min_impurity_decrease,
#             min_impurity_split=min_impurity_split,
#             ccp_alpha=ccp_alpha)

#         self.n_estimators = n_estimators
#         self.n_jobs = n_jobs
#         np.random.seed(random_state)

#     def fit(self, X, y, sample_weight = None):
#         self.n_samples_ = X.shape[0]

#         idx = np.random.choice(range(0, len(X)), len(X), replace = True)
#         super().fit(X[idx], y[idx], sample_weight)

#     def predict_proba(self, X, check_input=True):
#         proba = []
#         for _ in range(self.n_estimators):
#             proba.append(
#                 super().predict_proba(X + np.random.normal(loc = 0.0, scale = 0.01, size = X.shape))
#             )

#         # def _predict(X):
#         #     X = X + np.random.normal(loc = 0.0, scale = 0.25, size = X.shape)
#         #     return super().predict_proba(X)

#         # proba = [super().predict_proba(
#         #     X + np.random.normal(loc = 0.0, scale = 0.25, size = X.shape), check_input
#         #     ) for _ in range(self.n_estimators)]
#         # proba = Parallel(n_jobs=self.n_jobs, backend="loky")(
#         #     delayed(_predict) (X) for i in range(self.n_estimators)
#         # )
        
#         proba = np.array(proba)
#         return proba.mean(axis=0)


# class RandomDecisionTreeClassifier(DecisionTreeClassifier):
#     class Node:
#         def __init__(self):
#             self.prediction = None
#             self.feature = None
#             self.split = None
#             self.right = None
#             self.left = None

#     def __init__(self, 
#         n_estimators = 32,
#         s_randomization = 0.1,
#         n_jobs = None,
#         max_depth=None, 
#         criterion="gini",
#         splitter="best",
#         min_samples_split=2,
#         min_samples_leaf=1,
#         min_weight_fraction_leaf=0.,
#         max_features=None,
#         random_state=None,
#         max_leaf_nodes=None,
#         min_impurity_decrease=0.,
#         min_impurity_split=None,
#         class_weight=None,
#         ccp_alpha=0.0
#     ):
#         super().__init__(
#             criterion=criterion,
#             splitter=splitter,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             min_weight_fraction_leaf=min_weight_fraction_leaf,
#             max_features=max_features,
#             max_leaf_nodes=max_leaf_nodes,
#             class_weight=class_weight,
#             random_state=random_state,
#             min_impurity_decrease=min_impurity_decrease,
#             min_impurity_split=min_impurity_split,
#             ccp_alpha=ccp_alpha)

#         self.s_randomization = s_randomization
#         self.n_estimators = n_estimators
#         self.n_jobs = n_jobs
#         np.random.seed(random_state)

#     def fit(self, X, y, sample_weight = None):
#         self.n_samples_ = X.shape[0]
#         super().fit(X, y, sample_weight)

#         # Array of all nodes
#         self.nodes = []

#         # Pointer to the root node of this tree
#         self.head = None

#         sk_tree = self.tree_

#         node_ids = [0]
#         tmp_nodes = [self.Node()]
#         while(len(node_ids) > 0):
#             cur_node = node_ids.pop(0)
#             node = tmp_nodes.pop(0)

#             if sk_tree.children_left[cur_node] == _tree.TREE_LEAF and sk_tree.children_right[cur_node] == _tree.TREE_LEAF:
#                 # Get array of prediction probabilities for each class
#                 proba = sk_tree.value[cur_node][0, :]  
#                 node.prediction = proba / sum(proba) 
#             else:
#                 node.feature = sk_tree.feature[cur_node]
#                 node.split = sk_tree.threshold[cur_node]
            
#             node.id = len(self.nodes)
#             if node.id == 0:
#                 self.head = node

#             self.nodes.append(node)

#             if node.prediction is None:
#                 left_id = sk_tree.children_left[cur_node]
#                 node_ids.append(left_id)
#                 node.left = self.Node()
#                 tmp_nodes.append(node.left)

#                 right_id = sk_tree.children_right[cur_node]
#                 node_ids.append(right_id)
#                 node.right = self.Node()
#                 tmp_nodes.append(node.right)

#     def predict_proba(self, X, check_input=True):
#         def _predict(X):
#             preds = []
#             for x in X:
#                 n = self.head
#                 depth = 0
#                 while(n.prediction is None):
#                     p = 1.5-1/(1+np.exp(-self.s_randomization*depth))

#                     comparison = x[n.feature] <= n.split 

#                     if not np.random.choice([True, False], p = [p, 1-p]):
#                         comparison = not comparison

#                     if comparison:
#                         n = n.left
#                     else:
#                         n = n.right

#                     depth += 1
#                 preds.append(n.prediction)
#             return preds

#         proba = Parallel(n_jobs=self.n_jobs, backend="loky")(
#             delayed(_predict) (X) for i in range(self.n_estimators)
#         )
#         # proba = Parallel(n_jobs=self.n_jobs, backend="threading")(
#         #     delayed(_predict) (X) for i in range(self.n_estimators)
#         # )
        
#         proba = np.array(proba)
#         return proba.mean(axis=0)

# class PoissonDecisionTreeClassifier(DecisionTreeClassifier):
#     def __init__(self, max_depth, subspace_features,
#         criterion="gini",
#         splitter="best",
#         min_samples_split=2,
#         min_samples_leaf=1,
#         min_weight_fraction_leaf=0.,
#         max_features=None,
#         random_state=None,
#         max_leaf_nodes=None,
#         min_impurity_decrease=0.,
#         min_impurity_split=None,
#         class_weight=None,
#         ccp_alpha=0.0
#     ):
#         super().__init__(
#             criterion=criterion,
#             splitter=splitter,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             min_weight_fraction_leaf=min_weight_fraction_leaf,
#             max_features=max_features,
#             max_leaf_nodes=max_leaf_nodes,
#             class_weight=class_weight,
#             random_state=random_state,
#             min_impurity_decrease=min_impurity_decrease,
#             min_impurity_split=min_impurity_split,
#             ccp_alpha=ccp_alpha)

#         self.subspace_features = subspace_features

#     def predict_proba(self, X):
#         return super().predict_proba(X[:, self.features])

#     def predict(self, X):
#         return super().predict(X[:, self.features])

#     def fit(self, X, y, sample_weight=None):
#         if self.max_depth is not None:
#             d = np.random.poisson(self.max_depth - 1) + 1
#             self.max_depth = d
        
#         if self.subspace_features < 1.0:
#             n_features = int(X.shape[1] * self.subspace_features)
#             self.features = np.random.choice(range(X.shape[1]), n_features, replace=False)
#         else:
#             self.features = range(X.shape[1])
#         super().fit(X[:, self.features], y, sample_weight)


def vc(model, X, target):
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeClassifierWithSampleSize)):
        model.estimators_ = [model]

    v = []
    # n_samples = model.n_samples_
    for e in model.estimators_:
        v.append( e.tree_.node_count )

        # d = _lower_vc(e, 0, model.n_features_)
        # tmp = 2*d*np.log(np.exp(1) * n_samples / d)
        # v.append( np.sqrt( tmp / n_samples ) )

    return np.mean(v)

def lower_vc(model, X, target):
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeClassifierWithSampleSize)):
        model.estimators_ = [model]

    def _lower_vc(tree, cur_node, n_features):
        l_id = tree.tree_.children_left[cur_node]
        r_id = tree.tree_.children_right[cur_node]
        
        if (l_id == _tree.TREE_LEAF) and (r_id == _tree.TREE_LEAF):
            return np.floor(np.log2(n_features)) +1 
        elif (l_id == _tree.TREE_LEAF) and (r_id != _tree.TREE_LEAF):
            return _lower_vc(tree, r_id, n_features) + 1
        elif (l_id != _tree.TREE_LEAF) and (r_id == _tree.TREE_LEAF):
            return _lower_vc(tree, l_id, n_features) + 1
        else:
            return _lower_vc(tree, l_id, n_features) + _lower_vc(tree, r_id, n_features)

    v = []
    # n_samples = model.n_samples_
    for e in model.estimators_:
        v.append( _lower_vc(e, 0, model.n_features_) )

        # d = _lower_vc(e, 0, model.n_features_)
        # tmp = 2*d*np.log(np.exp(1) * n_samples / d)
        # v.append( np.sqrt( tmp / n_samples ) )

    return np.max(v)

def upper_vc(model, X, target):
    v = []

    for e in model.estimators_:
        v.append( e.tree_.node_count * np.log2(model.n_features_) )

    return np.max(v) 

# def avg_rademacher_tighter(model, X, target):
#     n_samples = model.n_samples_

#     r = []
#     if isinstance(model, (DecisionTreeClassifier, DecisionTreeClassifierWithSampleSize)):
#         model.estimators_ = [model]

#     for e in model.estimators_:
#         vc_tree = tree_from_sklearn_decision_tree(e) 
#         vc_dim = vcdim_upper_bound(vc_tree.tree, model.n_features_)
#         tmp = 2 * model.n_features_ * np.log(np.exp(1) * n_samples / vc_dim)
#         r.append( np.sqrt( tmp / n_samples ) )

#     return np.mean(r)

# # multi class margin?? 
# # is it soft hinge?
# def margin(model, X, target):
#     probas = model.predict_proba(X)
#     tmp = probas[np.arange(len(probas)), target]

#     probas[np.arange(len(probas)), target] = 0
#     margins = tmp - np.max(probas, axis=1)

#     return np.mean(margins)

def xval(model, X, Y, scoring, cv):
    def _score(model, X, Y, train_idx, test_idx, scoring):
        scores = {}

        imodel = copy.deepcopy(model)
        imodel.fit(X[train_idx], Y[train_idx])

        for key, val in scoring.items():
            train_score = val(imodel, X[train_idx], Y[train_idx])
            test_score = val(imodel, X[test_idx], Y[test_idx])

            scores["test_{}".format(key)] = test_score
            scores["train_{}".format(key)] = train_score
        return scores

    scores = Parallel(n_jobs=5)(
        delayed(_score) (copy.deepcopy(model), X, Y, train_idx, test_idx, scoring) for train_idx, test_idx in cv.split(X, Y)
    )

    score_dict = {}

    for iscores in scores:
        for key, val in iscores.items():
            if key not in score_dict:
                score_dict[key] = []

            score_dict[key].append(val)
    return score_dict

    # scores = {}

    # for train_idx, test_idx in cv.split(X, Y):
    #     imodel = copy.deepcopy(model)
    #     imodel.fit(X[train_idx], Y[train_idx])

    #     for key, val in scoring.items():
    #         train_score = val(imodel, X[train_idx], Y[train_idx])
    #         test_score = val(imodel, X[test_idx], Y[test_idx])

    #         if "test_{}".format(key) not in scores:
    #             scores["test_{}".format(key)] = []
    #         scores["test_{}".format(key)].append(test_score)

    #         if "train_{}".format(key) not in scores:
    #             scores["train_{}".format(key)] = []
    #         scores["train_{}".format(key)].append(train_score)

    # return scores

def soft_hinge(model, X, target):
    probas = model.predict_proba(X)

    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(model.n_classes_)] for y in target] )
    zeros = np.zeros_like(target_one_hot)
    hinge = np.maximum(1.0 - target_one_hot * probas, zeros)
    return hinge.mean()