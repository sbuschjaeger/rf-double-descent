import torch
import torch.nn as nn

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from Metrics import bias_with_base, c_bound, bias, c_bound_with_base, diversity, diversity_with_base

class NCRandomForestClassifier(BaseEstimator, ClassifierMixin, nn.Module):
    def __init__(self,base_forest, l_reg, n_jobs = 1, verbose = True):
        assert l_reg >= 0 and l_reg <= 1, "l_reg must be from [0,1]" 

        super().__init__()
        
        self.l_reg = l_reg
        self.base_forest = base_forest
        self.n_jobs = n_jobs
        self.verbose = verbose

    def predict_proba(self, X):
         # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.base_forest.predict_proba(X)

    def predict(self, X):
         # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.base_forest.predict(X)

    def forward(self, x):
        pred = []
        #torch.gather(self.leafs[0,:].squeeze(1), 0, torch.tensor(idx))
        for i, h in enumerate(self.base_forest.estimators_):
            idx = h.apply(x.numpy())
            pred.append(self.leafs[i][idx,:])
        
        return torch.stack(pred)

    def ncl_upper(self, base_preds, target):
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

    # def ncl_loss(self, base_preds, target):
    #     n_estimators = base_preds.shape[0]
    #     n_preds = base_preds.shape[1]
    #     n_classes = base_preds.shape[2]
    #     target_one_hot = torch.tensor( [ [1.0 if y == i else 0.0 for i in range(n_classes)] for y in target] )

    #     eye_matrix = torch.eye(n_classes, dtype=torch.float64).repeat(n_preds, 1, 1)
    #     D = 2.0*eye_matrix
    #     fbar = base_preds.mean(dim=0)

    #     losses = []
    #     for pred in base_preds:
    #         diff = pred - fbar 
    #         covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
    #         div = 1.0/2.0 * covar
    #         i_loss = ((pred - target_one_hot)**2).mean()
    #         losses.append( i_loss - 1.0 / n_estimators * self.l_reg * div)
        
    #     losses = torch.stack(losses, dim = 1)
    #     return losses.mean()

    def fit(self, X, y, sample_weights = None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(self.classes_)

        self.base_forest.fit(X,y)
        # for tree in self.base_forest.estimators_:
        #     tree.tree_.value[:] = tree.tree_.value / tree.tree_.value.sum(axis=(1,2))[:,np.newaxis,np.newaxis]
        
        # The evaluation expects a self.estimators_ field and not a self.base-forest.estimators_ field
        self.estimators_ = self.base_forest.estimators_

        dataloader = DataLoader(
            TensorDataset(torch.Tensor(X),torch.Tensor(y)), 
            batch_size=64, shuffle=True
        ) 

        leafs = []
        for tree in self.base_forest.estimators_:
            leafs.append(tree.tree_.value / tree.tree_.value.sum(axis=(1,2))[:,np.newaxis,np.newaxis])
        #print([l.shape for l in leafs])
        #leafs = np.array(leafs)
        self.leafs = nn.ParameterList([ nn.Parameter(torch.from_numpy(l).squeeze(1)) for l in leafs])
        # print(self.leafs[0].shape)
        #self.leafs = torch.nn.Parameter(torch.from_numpy(leafs).squeeze(2))
        #self.leafs = torch.nn.Parameter(torch.zeros_like(self.leafs))
        
        # Loss and optimizer
        # criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)  
        optimizer = torch.optim.Adam(self.parameters())  

        torch.set_num_threads(self.n_jobs)
        n_epochs = 10

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
                for i, (x, y) in enumerate(dataloader):  
                    batch_cnt += 1
                    # Forward pass
                    outputs = self.forward(x)

                    if self.verbose:
                        avg_accuracy += (outputs.mean(dim=0).argmax(axis=1) == y).sum()
                        for pred in outputs:
                            avg_accuracy_per_model += (pred.argmax(axis=1) == y).sum()
                            
                        avg_cbound += c_bound_with_base(outputs.detach().numpy(), y)
                        avg_diversity += diversity_with_base(outputs.detach().numpy(), y).item()
                        avg_bias += bias_with_base(outputs.detach().numpy(), y).item()

                    #loss = self.ncl_loss(outputs, y)
                    loss = self.ncl_upper(outputs, y)
                    avg_loss += loss.item()

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # print("")
                    # print(self.leafs)
                    # print(self.leafs.grad)

                    example_cnt += x.shape[0]

                    desc = '[{}/{}] loss {:2.6f} acc {:2.4f} cbound {:2.6f} avg acc {:2.4f} diversity {:2.4f} bias {:2.4f}'.format(
                        epoch, 
                        n_epochs - 1,
                        avg_loss / batch_cnt,
                        avg_accuracy / example_cnt,
                        avg_cbound / batch_cnt,
                        avg_accuracy_per_model / (example_cnt * self.base_forest.n_estimators),
                        avg_diversity / batch_cnt,
                        avg_bias / batch_cnt
                    )
                    pbar.update(x.shape[0])
                    pbar.set_description(desc)

        for i, tree in enumerate(self.base_forest.estimators_):
            h_weights = self.leafs[i].detach().numpy()
            tree.tree_.value[:] = h_weights[:,np.newaxis]