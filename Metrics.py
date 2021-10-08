import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from Models import DecisionTreeClassifierWithSampleSize

def accuracy(model, X, target):
    return 100.0*accuracy_score(model.predict_proba(X).argmax(axis=1), target)

def soft_hinge(model, X, target):
    probas = model.predict_proba(X)

    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(model.n_classes_)] for y in target] )
    zeros = np.zeros_like(target_one_hot)
    hinge = np.maximum(1.0 - target_one_hot * probas, zeros)
    return hinge.mean()

def avg_rademacher(model, X, target):
    n_samples = model.n_samples_
    D = 2*np.log(model.n_features_ + 2) #20 
    r = []

    if isinstance(model, (DecisionTreeClassifier, DecisionTreeClassifierWithSampleSize)):
        model.estimators_ = [model]

    for e in model.estimators_:
        rad = np.sqrt( D * ( 4 * e.tree_.node_count + 2 ) / np.sqrt(n_samples + 1 ) )
        r.append( rad )
    return np.mean(r)

def n_leaves(model, X, target):
    cnt = 0

    if isinstance(model, (DecisionTreeClassifier, DecisionTreeClassifierWithSampleSize)):
        model.estimators_ = [model]

    for e in model.estimators_:
        stack = [(0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            node_id = stack.pop()

            is_split_node =  e.tree_.children_left[node_id] != e.tree_.children_right[node_id]
            if is_split_node:
                stack.append((e.tree_.children_left[node_id]))
                stack.append((e.tree_.children_right[node_id]))
            else:
                cnt += 1

    return cnt

def n_nodes(model, X, target):
    n_nodes = 0

    if isinstance(model, (DecisionTreeClassifier, DecisionTreeClassifierWithSampleSize)):
        model.estimators_ = [model]

    for e in model.estimators_:
        n_nodes += e.tree_.node_count

    return n_nodes

def effective_height(model, X, target):
    e_height = 0

    if isinstance(model, (DecisionTreeClassifier, DecisionTreeClassifierWithSampleSize)):
        model.estimators_ = [model]

    for e in model.estimators_:
        max_depth = 0
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            node_id, depth = stack.pop()

            if depth > max_depth:
                max_depth = depth

            is_split_node =  e.tree_.children_left[node_id] != e.tree_.children_right[node_id]
            if is_split_node:
                stack.append((e.tree_.children_left[node_id], depth + 1))
                stack.append((e.tree_.children_right[node_id], depth + 1))
        
        e_height += max_depth

    return e_height / len(model.estimators_)

def mse(model, X, target):
    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(model.n_classes_)] for y in target] )

    probas = model.predict_proba(X)
    return np.mean( (probas - target_one_hot)**2 )

def bias_with_base(base_preds, target):
    n_classes = base_preds.shape[2]
    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )

    biases = []
    for pred in base_preds:
        i_loss = ((pred - target_one_hot)**2).mean()
        biases.append( i_loss )
    
    return np.mean(biases)

def bias(model, X, target):
    base_preds = []
    if hasattr(model, "estimators_"):
        for e in model.estimators_:
            base_preds.append(e.predict_proba(X))
    else:
        base_preds.append(model.predict_proba)
    
    base_preds = np.array(base_preds)
    return bias_with_base(base_preds, target)

# def diversity(model, X, target): 
#     lhs = mse(model, X, target)
#     rhs_first = bias(model, X, target)

#     return lhs - rhs_first

def diversity_with_base(base_preds, target):
    n_estimators = base_preds.shape[0]
    n_preds = base_preds.shape[1]
    n_classes = base_preds.shape[2]
    target_one_hot = np.array( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )

    # https://stackoverflow.com/questions/32171917/how-to-copy-a-2d-array-into-a-3rd-dimension-n-times
    eye_matrix = np.eye(n_classes, dtype=np.float64)[np.newaxis,:,:].repeat(n_preds, axis=0)
    D = 2.0 * 1.0 / n_classes * eye_matrix
    fbar = base_preds.mean(axis=0)

    divs = []
    #avg_div = 0
    for pred in base_preds:
        diff = pred - fbar 
        covar = (diff[:,np.newaxis]@(D@diff[:,:,np.newaxis])).squeeze()
        #covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
        #avg_div += 1.0/2.0 * covar.mean()D
        divs.append( 1.0/2.0 * covar )
    
    div1 = np.mean(divs)
    
    # lhs = mse(model, X, target)
    # rhs_first = bias(model, X, target)
    # div2 = rhs_first - lhs
    # print("DIFFERENCE: {}".format(abs(div1-div2)))
    return div1
    #return torch.stack(divs).mean() #sum(dim=0).mean(dim=0)

def diversity(model, X, target):
    base_preds = []
    if hasattr(model, "estimators_"):
        for e in model.estimators_:
            base_preds.append(e.predict_proba(X))
    else:
        base_preds.append(model.predict_proba)
    base_preds = np.array(base_preds)
    return diversity_with_base(base_preds, target)

def c_bound_with_base(base_preds, target):
    n_estimators = base_preds.shape[0]
    n_preds = base_preds.shape[1]
    n_classes = base_preds.shape[2]
    adjusted_target_one_hot = np.array( [ [1.0 if y == i else -1.0 for i in range(n_classes)] for y in target] )

    delta = 0.1 / 2
    m1 = np.sqrt(2.0 / n_preds * np.log(np.sqrt(n_preds) / delta))
    m2 = np.sqrt(2.0 / n_preds * np.log(2.0*np.sqrt(n_preds) / delta))
    cbound_per_class = []
    for i in range(n_classes):
        adjusted_preds = base_preds * 2.0 - 1.0

        mu = (adjusted_preds[:,:,i] * adjusted_target_one_hot[:,i]).mean()
        mu2 = ((adjusted_preds[:,:,i].mean(axis=0))**2).sum()
        
        cbound = 1.0 - (np.maximum(0, mu - m1)**2) / (np.minimum(1, mu2 + m2))
        cbound_per_class.append(cbound)
    
    return np.mean(cbound_per_class)

def c_bound(model, X, target):
    base_preds = []
    if hasattr(model, "estimators_"):
        for e in model.estimators_:
            base_preds.append(e.predict_proba(X))
    else:
        base_preds.append(model.predict_proba)
    base_preds = np.array(base_preds)
    return c_bound_with_base(base_preds, target)