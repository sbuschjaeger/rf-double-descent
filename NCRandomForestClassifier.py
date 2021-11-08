import numpy as np
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from Metrics import avg_accuracy_with_base, bias_with_base, c_bound, bias, c_bound_with_base, diversity, diversity_with_base
        
class NCRandomForestClassifier(BaseEstimator, ClassifierMixin, nn.Module):
    def __init__(self,base_forest, l_reg = 0.0, n_jobs = 1, verbose = True):
        """Generates a new NCRandomForestClassifier. The NCRandomForestClassifier first fits a regular RandomForst (or equivalent model) given by base_forest and then performs Negative Correlation Learning on the leafs of this forest with SGD. 

        Args:
            base_forest (RandomForestClassifier or equivalent): `base_forest` must be a scikit-learn tree ensemble (e.g. RandomForestClassifier). It should offer a `fit(X,y)`, `predict(X)`, 'predict_proba(X)` method and have a `estimators_` field. 
            l_reg (float): The lambda regularization parameter for negative correlation learning. Must be >= 0. Defaults to 0.0.
            n_jobs (int, optional): The number of jobs used by PyTorch (set via `torch.set_num_threads(self.n_jobs)`). Note that this does not affect the number of jobs used to fit the base_forest.Defaults to 1.
            verbose (bool, optional): If True a tqdm progress bar with some statistics is printed. If False, not statistics are printed. Defaults to True.
        """
        #assert l_reg >= 0, "l_reg should be >= 0."
        super().__init__()
        
        self.l_reg = l_reg
        self.base_forest = base_forest
        self.n_jobs = n_jobs
        self.verbose = verbose

    def predict_proba(self, X):
        """Call predict_proba on the base_forest.

        Args:
            X ( (N, d) numpy matrix ): The (N,d) numpy matrix of the data

        Returns:
            (N, C) numpy matrix: The (N,C) numpy matrix of the predicted class probabilities (where C is the number of classes).
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.base_forest.predict_proba(X)

    def predict(self, X):
        """Call predict on the base_forest.

        Args:
            X ( (N, d) numpy matrix ): The (N,d) numpy matrix of the data

        Returns:
            (N) numpy vector: The (N) numpy vector of the predicted class.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.base_forest.predict(X)

    def forward(self, x, idx):
        """Applies the individual trees of base_forest to the given batch x and stacks the leaf predictions from self.leafs into a torch tensor. 

        Args:
            x ((B, d) torch.Tensor): The current (B, d) batch of the data where B is the batch size and d is the number of features. 

        Returns:
            (M, B, C) torch.Tensor : The stacked predictions of all of the M classifiers in the base_forest where C is the number of class probabilities.
        """
        pred = []
        #for i, h in enumerate(self.base_forest.estimators_):
            #idx = h.apply(x.numpy())
            #pred.append(self.leafs[i][idx,:])

        for i in range(len(self.base_forest.estimators_)):
            pred.append(self.leafs[i][idx[:,i]])
        
        return torch.stack(pred)

    def ncl_upper(self, base_preds, target):
        """Computes the upper bound of the NCL loss using the individual losses as well as joint ensemble loss. For the MSE (as implemented here) this upper bound is exact (see References). This implementation assumes 0 <= l_leg <= 1 and smoothly interpolates between minimizing the joint ensemble  loss (f(x)-y)^2 and each model individually \sum_h(h(x) - y)^2 

        Args:
            base_preds ((M, B, C) torch.Tensor): The predictions of each individual classifier on the current batch
            target ((B,) numpy vector / list ): List of corresponding classes for each example in the batch

        References:
            - Buschjäger, Sebastian, Lukas Pfahler, and Katharina Morik. "Generalized negative correlation learning for deep ensembling." arXiv preprint arXiv:2011.02952 (2020). (https://arxiv.org/pdf/2011.02952)
            - Webb, Andrew M., et al. "To Ensemble or Not Ensemble: When does End-To-End Training Fail?." arXiv preprint arXiv:1902.04422 (2019). (http://www.cs.man.ac.uk/~gbrown/publications/ecml2020webb_supp.pdf)

        Returns:
            (1,) Torch.Tensor: The loss.
        """
        n_classes = base_preds.shape[2]
        target_one_hot = torch.tensor( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )
        
        fbar = base_preds.mean(dim=0)
        floss = ((fbar - target_one_hot)**2).mean()

        ilosses = []
        for pred in base_preds:
            ilosses.append( ((pred - target_one_hot)**2).mean() )
        
        ilosses = torch.stack(ilosses)
        return (1.0-self.l_reg)*ilosses.mean() + self.l_reg*floss

    # def bias(self, base_preds, target):
    #     n_estimators = base_preds.shape[0]
    #     n_preds = base_preds.shape[1]
    #     n_classes = base_preds.shape[2]
    #     target_one_hot = torch.tensor( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )

    #     losses = []
    #     for pred in base_preds:
    #         i_loss = ((pred - target_one_hot)**2).mean()
    #         losses.append( i_loss )
        
    #     losses = torch.tensor(losses)
    #     return losses.mean()

    # def diversity(self, base_preds, target):
    #     n_estimators = base_preds.shape[0]
    #     n_preds = base_preds.shape[1]
    #     n_classes = base_preds.shape[2]
    #     target_one_hot = torch.tensor( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )

    #     eye_matrix = torch.eye(n_classes, dtype=torch.float64).repeat(n_preds, 1, 1)
    #     D = 2.0*1.0/n_classes*eye_matrix
    #     fbar = base_preds.mean(dim=0)

    #     divs = []
    #     #avg_div = 0
    #     for pred in base_preds:
    #         diff = pred - fbar 
    #         covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
    #         #avg_div += 1.0/2.0 * covar.mean()
    #         divs.append( 1.0/2.0 * covar )
        
    #     return torch.stack(divs).mean() #sum(dim=0).mean(dim=0)

    # def c_bound(self, base_preds, target):
    #     n_estimators = base_preds.shape[0]
    #     n_preds = base_preds.shape[1]
    #     n_classes = base_preds.shape[2]
    #     adjusted_target_one_hot = torch.tensor( [ [1.0 if y == i else -1.0 for i in range(n_classes)] for y in target] )

    #     delta = 0.1 / 2
    #     m1 = np.sqrt(2.0 / n_preds * np.log(np.sqrt(n_preds) / delta))
    #     m2 = np.sqrt(2.0 / n_preds * np.log(2.0*np.sqrt(n_preds) / delta))
    #     cbound_per_class = []
    #     for i in range(n_classes):
    #         adjusted_preds = base_preds * 2.0 - 1.0

    #         mu = (adjusted_preds[:,:,i] * adjusted_target_one_hot[:,i]).mean()
    #         mu2 = ((adjusted_preds[:,:,i].mean(axis=0))**2).sum()
            
    #         cbound = 1.0 - (np.maximum(0, mu - m1)**2) / (np.minimum(1, mu2 + m2))
    #         cbound_per_class.append(cbound)
        
    #     return np.mean(cbound_per_class)

    def ncl_loss(self, base_preds, target):
        """Computes the exact NCL loss using the second derivative of the MSE loss as well as the individual losses of each classifier. 

        Args:
            base_preds ((M, B, C) torch.Tensor): The predictions of each individual classifier on the current batch
            target ((B,) numpy vector / list ): List of corresponding classes for each example in the batch

        References:
            - Buschjäger, Sebastian, Lukas Pfahler, and Katharina Morik. "Generalized negative correlation learning for deep ensembling." arXiv preprint arXiv:2011.02952 (2020). (https://arxiv.org/pdf/2011.02952)
            - Webb, Andrew M., et al. "To Ensemble or Not Ensemble: When does End-To-End Training Fail?." arXiv preprint arXiv:1902.04422 (2019). (http://www.cs.man.ac.uk/~gbrown/publications/ecml2020webb_supp.pdf)

        Returns:
            (1,) Torch.Tensor: The loss.
        """
        n_estimators = base_preds.shape[0]
        n_preds = base_preds.shape[1]
        n_classes = base_preds.shape[2]
        target_one_hot = torch.tensor( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )

        eye_matrix = torch.eye(n_classes, dtype=torch.float64).repeat(n_preds, 1, 1)
        D = 2.0 * 1.0 / n_classes * eye_matrix
        fbar = base_preds.mean(dim=0)

        losses = []
        # divs = []
        # bias = []
        for pred in base_preds:
            diff = pred - fbar 
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            div = 1.0/2.0 * covar.mean()
            iloss = ((pred - target_one_hot)**2).mean()
            losses.append( iloss - self.l_reg * div)
            # divs.append(div)
            # bias.append(iloss)
        
        # div2 = torch.stack(bias).mean() - ((fbar - target_one_hot)**2).mean()
        # div1 = torch.stack(divs).mean()

        return torch.stack(losses).mean()
        #return losses.mean()

    def fit(self, X, y, sample_weights = None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store some statistics
        self.classes_ = unique_labels(y)
        # self.X_ = X
        # self.y_ = y
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(self.classes_)

        # Fit the base forest
        self.base_forest.fit(X,y,sample_weights)
        
        # The evaluation expects a self.estimators_ field and not a self.base-forest.estimators_ field
        self.estimators_ = self.base_forest.estimators_

        # Buffer all indices from apply() and combine it with the dataloader so that we do not need to call it on every forward() call. This roughly doubles the speed. 
        tmpidx = []
        for i, h in enumerate(self.base_forest.estimators_):
            tmpidx.append(h.apply(X))
        idx = np.array(tmpidx).swapaxes(0,1)

        # TODO make batch size a parameter?
        dataloader = DataLoader(
            TensorDataset(torch.Tensor(X),torch.Tensor(y), torch.tensor(idx)), 
            batch_size=64, shuffle=True
        ) 

        # We cannot optimize directly over the sklearn datastructures. Hence we first copy them into self.leafs which is a PyTorch managed ParameterList
        # We also normalize the leaf values beforehand so that SGD does not have such a difficult time dealing with count data.
        leafs = []
        for tree in self.base_forest.estimators_:
            leafs.append(tree.tree_.value / tree.tree_.value.sum(axis=(1,2))[:,np.newaxis,np.newaxis])
            #leafs[-1] = leafs[-1] + np.random.normal(loc = 0.0, scale = 1e-4, size = leafs[-1].shape)
        self.leafs = nn.ParameterList([ nn.Parameter(torch.from_numpy(l).squeeze(1)) for l in leafs])

        # TODO Make these parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)  
        torch.set_num_threads(self.n_jobs)
        n_epochs = 50

        # Train the model
        for epoch in range(n_epochs):
            avg_loss = 0
            example_cnt = 0
            avg_accuracy = 0
            avg_cbound = 0
            avg_accuracy_per_model = 0
            avg_diversity = 0
            avg_bias = 0
            batch_cnt = 0 

            with tqdm.tqdm(total=X.shape[0], ncols=150, disable=not self.verbose) as pbar:
                for i, (x, y, idx) in enumerate(dataloader):  
                    batch_cnt += 1
                    # Forward pass
                    outputs = self.forward(x, idx)

                    if self.verbose:
                        #avg_accuracy += 100.0 * 1.0 / x.shape[0] * (outputs.mean(dim=0).argmax(axis=1) == y).sum()
                        avg_accuracy += 100.0*accuracy_score(outputs.mean(dim=0).argmax(axis=1).detach().numpy(), y)
                        avg_accuracy_per_model += avg_accuracy_with_base(outputs.detach().numpy(), y)
                        avg_cbound += c_bound_with_base(outputs.detach().numpy(), y)
                        avg_diversity += diversity_with_base(outputs.detach().numpy(), y).item()
                        avg_bias += bias_with_base(outputs.detach().numpy(), y).item()

                    # TODO make this a parameter
                    loss = self.ncl_loss(outputs, y)
                    #loss = self.ncl_upper(outputs, y)
                    avg_loss += loss.item()

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    example_cnt += x.shape[0]

                    desc = '[{}/{}] loss {:2.6f} acc {:2.4f} cbound {:2.6f} avg acc {:2.4f} diversity {:2.4f} bias {:2.4f}'.format(
                        epoch, 
                        n_epochs - 1,
                        avg_loss / batch_cnt,
                        avg_accuracy / batch_cnt,
                        avg_cbound / batch_cnt,
                        avg_accuracy_per_model / batch_cnt,
                        avg_diversity / batch_cnt,
                        avg_bias / batch_cnt
                    )
                    pbar.update(x.shape[0])
                    pbar.set_description(desc)

        # Copy the trained leafs back into the SK-Learn data-structures 
        for i, tree in enumerate(self.base_forest.estimators_):
            h_weights = self.leafs[i].detach().numpy()
            tree.tree_.value[:] = h_weights[:,np.newaxis]