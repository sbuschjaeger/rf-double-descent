#!/usr/bin/env python3

import copy
# import joblib
# from joblib.parallel import Parallel
import numpy as np
import pandas as pd
import argparse
import sys
import random

from multiprocessing import Pool

from sklearn.model_selection import cross_validate, StratifiedKFold
# from dask import compute, delayed
# from dask.distributed import Client

# from multiprocessing import Pool
from tqdm import tqdm

from sklearn.utils import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from datasets import get_dataset
from NCRandomForestClassifier import NCRandomForestClassifier

from Metrics import accuracy, avg_accuracy, avg_rademacher, c_bound, n_nodes, n_leaves, effective_height, soft_hinge, mse, bias, diversity
from Models import BaggingClassifierWithSampleSize, DataAugmentationDecisionTreeClassifierWithSampleSize, DataAugmentationRandomForestClassifierWithSampleSize, DecisionTreeClassifierWithSampleSize, HomogeneousForest, RandomForestClassifierWithSampleSize

def beautify_scores(scores):
    """Remove test_ in all scores for a nicer output and compute the mean of each score.

    Args:
        scores (dict): A dictionary of the scores in which each score has a list of the corresponding scores, e.g. scores["test_accuracy"] = [0.88, 0.82, 0.87, 0.90, 0.85]

    Returns:
        dict: A dictionary in which all "test_*" keys have been replaced by "*" and all scores are now averaged.  
    """
    nice_scores = {}
    for key, val in scores.items():
        if "test" in key:
            key = key.replace("test_","")

        nice_scores[key] = np.mean(val)
    return nice_scores

def merge_dictionaries(dicts):
    """Merges the given list of dictionaries into a single dictionary:
    [
        {'a':1,'b':2}, {'a':1,'b':2}, {'a':1,'b':2}
    ]
    is merged into
    {
        'a' : [1,1,1]
        'b' : [2,2,2]
    }

    Args:
        dicts (list of dicts): The list of dictionaries to be merged.

    Returns:
        dict: The merged dictionary.
    """
    merged = {}
    for d in dicts:
        for key, val in d.items():
            if key not in merged:
                merged[key] = [val]
            else:
                merged[key].append(val)
    return merged

def run_eval(cfg):
    """Fits and evalutes the model given the current configuration. The cfg tuple is expected to have the following form
        (model, X, Y, scoring, idx, additional_infos, run_id)
    This function basically extracts the train/test indicies supplied by idx given the current run_id:
        train, test = idx[rid]
        XTrain, YTrain = X[train,:], Y[train]
        XTest, YTest = X[test,:], Y[test]
    and then trains and evaluates a classifier on XTrain/Ytrain and XTest/YTest. Any additional_infos passed to this function are simply returned which makes housekeeping a little easier. 

    Args:
        cfg (tuple): A tuple of the form (model, X, Y, scoring, idx, additional_infos, run_id)

    Returns:
        a tuple of (dict, dict): The first dictionary is the result of the evaluation of the form 
        {
            'test_accuracy': 0.8,
            'train_accuracy': 0.9, 
            # ....
        }. The second dictionary contains the additional_infos passed to this function. 
    """
    model, X, Y, scoring, idx, additional_infos, rid = cfg["model"], cfg["X"], cfg["Y"], cfg["scoring"], cfg["idx"], cfg["additional_infos"], cfg["run_id"]

    train, test = idx[rid]
    XTrain, YTrain = X[train,:], Y[train]
    XTest, YTest = X[test,:], Y[test]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    XTrain = scaler.fit_transform(XTrain)
    XTest = scaler.transform(XTest)
    
    model.fit(XTrain, YTrain)

    scores = {}
    for name, method in scoring.items():
        scores["test_{}".format(name)] = method(model, XTest, YTest)
        scores["train_{}".format(name)] = method(model, XTrain, YTrain)

    return scores, additional_infos

def prepare_xval(cfg, xval):
    """Small helper function which copies the given config xval times and inserts the correct run_id for running cross-validations.

    Args:
        cfg (dict): The configuration.
        xval (int): The number of cross-validation runs.

    Returns:
        list: A list of configurations.
    """
    cfgs = []
    for i in range(xval):
        tmp = copy.deepcopy(cfg)
        tmp["run_id"] = i
        cfgs.append(tmp)
    return cfgs

# def run_xval(cfg):
#     model, X, Y, scoring, kf, n_jobs, additional_infos = cfg["model"], cfg["X"], cfg["Y"], cfg["scoring"], cfg["kf"], cfg["n_jobs"], cfg["additional_infos"]
#     cv_results = cross_validate(model, X, Y, scoring = scoring, cv = kf, return_train_score=True, n_jobs = n_jobs)
#     return {
#         **additional_infos,
#         # "dataset":dataset,
#         # "method":"NCF-{}".format(l),
#         # "max_nodes":mn,
#         # "T":args.n_estimators,
#         **beautify_scores(cv_results)
#     }

def main(args):
    for dataset in args.dataset:
        X, Y = get_dataset(dataset, args.tmpdir)
        
        if X is None or Y is None: 
            exit(1)

        random_state = 42
        np.random.seed(random_state)

        scoring = {
            "accuracy":accuracy,
            "avg_accuracy":avg_accuracy,
            "avg_rademacher":avg_rademacher,
            "n_nodes":n_nodes,
            "n_leaves":n_leaves,
            "effective_height":effective_height,
            "margin":soft_hinge,
            "loss":mse,
            "bias":bias,
            "diversity":diversity,
            "cbound":c_bound
        }

        if dataset == "mnist" or dataset == "fashion":
            idx = np.array( [ (list(range(0,60000)), list(range(60000,70000))) ] , dtype=object)
            args.xval = 1
            #idx = np.array( [ (list(range(1000,7000)), list(range(0,1000))) ] , dtype=object)
        else:
            kf = StratifiedKFold(n_splits=args.xval, random_state=random_state, shuffle=True)
            idx = np.array([(train_idx, test_idx) for train_idx, test_idx in kf.split(X, Y)], dtype=object)

        from collections import Counter
        print("Data: ", X.shape, " ", X[0:2,:])
        print("Labels: ", Y.shape, " ", Counter(Y))
        # sys.exit(1)
        common_config = {
            "X": X,
            "Y": Y, 
            "scoring":scoring, 
            "idx":idx, 
        }

        n_jobs_per_forest = None #min(5, args.n_jobs - n_jobs_per_forest)
        n_jobs_in_pool = args.n_jobs

        #client = Client(n_workers=50) #n_workers = args.n_jobs, threads_per_worker = 1, memory_limit = '32GB
        parallel_backend("threading") #, args.n_jobs

        print("Preparing experiments")
        configs = []
        
        for mn in [4096]:
            ls = [l for l in np.round(np.arange(-20,1.0,0.1),4)] + [l for l in np.round(np.arange(1.0,1.005,0.0001),4)] 
            #ls = np.round(np.arange(-20,-8.0,0.25),4)
            #print(ls)
            #for l in np.round(np.arange(1.0,1.005,0.0001),4):
            for l in ls:
                configs.extend(
                    prepare_xval(
                        {
                            **common_config,
                            "model":NCRandomForestClassifier(
                                base_forest = RandomForestClassifier(n_estimators = args.n_estimators,  max_leaf_nodes=mn, n_jobs = n_jobs_per_forest, random_state=random_state, bootstrap=True),#, max_features=1
                                # base_forest=ExtraTreesClassifier(
                                #     n_estimators=args.n_estimators,
                                #     n_jobs = n_jobs_per_forest,
                                #     max_leaf_nodes=mn
                                # ),
                                # base_forest = BaggingClassifier(
                                #     base_estimator=DecisionTreeClassifier(max_leaf_nodes=mn),
                                #     n_estimators = args.n_estimators,
                                #     n_jobs = n_jobs_per_forest, 
                                #     bootstrap = False,
                                #     max_samples = 0.95
                                # ),
                                # base_forest = HomogeneousForest(
                                #     base_dt=DecisionTreeClassifier(max_leaf_nodes=mn, random_state=random_state),
                                #     n_estimators = args.n_estimators, n_jobs = n_jobs_per_forest
                                # ),
                                l_reg = l, n_jobs = 1, verbose = False
                            ), 
                            "additional_infos": {
                                "dataset":dataset,
                                "method":"NCF-{}".format(l),
                                "max_nodes":mn,
                                "T":args.n_estimators,

                            }
                        },
                        args.xval
                    )
                )
        
        for mn in args.max_nodes:
            configs.extend(
                prepare_xval(
                    {
                        **common_config,
                        "model":RandomForestClassifierWithSampleSize(
                            n_estimators = args.n_estimators, bootstrap = True, max_leaf_nodes=mn, n_jobs = n_jobs_per_forest, random_state=random_state
                        ), 
                        "additional_infos": {
                            "dataset":dataset,
                            "method":"RF",
                            "max_nodes":mn,
                            "T":args.n_estimators
                        }
                    },
                    args.xval 
                )
            )

            configs.extend(
                prepare_xval(
                    {
                        **common_config,
                        "model":DataAugmentationRandomForestClassifierWithSampleSize(
                            n_estimators = args.n_estimators, bootstrap = True, max_leaf_nodes=mn, n_jobs = n_jobs_per_forest, random_state=random_state
                        ),
                        "additional_infos": {
                            "dataset":dataset,
                            "method":"DA-RF",
                            "max_nodes":mn,
                            "T":args.n_estimators
                        }
                    },
                    args.xval
                )
            )

            configs.extend(
                prepare_xval(
                    {
                        **common_config,
                        "model":DecisionTreeClassifierWithSampleSize(max_leaf_nodes=mn, random_state=random_state),
                        "additional_infos": {
                            "dataset":dataset,
                            "method":"DT",
                            "max_nodes":mn,
                            "T":1
                        }
                    }, 
                    args.xval
                )
            )

            configs.extend(
                prepare_xval(
                    {
                        **common_config,
                        "model":DataAugmentationDecisionTreeClassifierWithSampleSize(max_leaf_nodes=mn, random_state=random_state),
                        "additional_infos": {
                            "dataset":dataset,
                            "method":"DA-DT",
                            "max_nodes":mn,
                            "T":1
                        }
                    },
                    args.xval
                )
            )
        
        print("Configured {} experiments. Starting experiments now using {} jobs.".format(len(configs), n_jobs_in_pool))
        pool = Pool(n_jobs_in_pool)
        delayed_metrics = []
        for eval_return in tqdm(pool.imap_unordered(run_eval, configs), total=len(configs)):
            delayed_metrics.append(eval_return)
        
        metrics = []
        names = list(set(["_".join([str(val) for val in cfg.values()]) for _, cfg in delayed_metrics]))
        for n in names:
            tmp = [scores for scores, cfg in delayed_metrics if "_".join([str(val) for val in cfg.values()]) == n]
            cfg = [cfg for _, cfg in delayed_metrics if "_".join([str(val) for val in cfg.values()]) == n][0]

            metrics.append(
                {
                    **beautify_scores(merge_dictionaries(tmp)),
                    **cfg
                }
            )
             
        df = pd.DataFrame(metrics)
        df.to_csv("{}.csv".format(dataset),index=False)

        df = df.sort_values(by=["dataset","max_nodes", "method", "T"])
        with pd.option_context('display.max_rows', None): 
            print(df[["dataset", "method", "max_nodes", "effective_height", "n_leaves", "T", "train_accuracy", "accuracy", "n_nodes", "avg_rademacher", "bias", "diversity"]])

if __name__ == '__main__':
    #sys.argv = ['./run.py', '-x', '5', '-M', '256', '--n_jobs', '1', '--max_nodes', '4096', '-d', 'magic']
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
    parser.add_argument("--max_nodes", help="Maximum number of nodes in the trees. Corresponds to sci-kit learns max_nodes parameter. Can be a list of arguments for multiple experiments", nargs='+', type=int, default=[])
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments. Can be a list of arguments for multiple dataset. Have a look at datasets.py for all supported datasets.",type=str, default=["magic"], nargs='+')
    parser.add_argument("-M", "--n_estimators", help="Number of estimators in the forest.", type=int, default=32)
    parser.add_argument("-x", "--xval", help="Number of cross-validation runs if the dataset does not contain a train/test split.",type=int, default=10)
    parser.add_argument("-t", "--tmpdir", help="Temporary folder in which datasets should be stored.",type=str, default=None)
    args = parser.parse_args()

    if args.max_nodes is None or len(args.max_nodes) == 0:
        args.max_nodes = [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
        #args.max_nodes = list(range(10,100,10)) + list(range(100,1000,50)) + list(range(1000,10000,1000))
    
    main(args)