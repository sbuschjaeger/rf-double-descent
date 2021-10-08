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

from Metrics import accuracy, avg_rademacher, c_bound, n_nodes, n_leaves, effective_height, soft_hinge, mse, bias, diversity
from Models import DataAugmentationDecisionTreeClassifierWithSampleSize, DataAugmentationRandomForestClassifierWithSampleSize, DecisionTreeClassifierWithSampleSize, RandomForestClassifierWithSampleSize

def beautify_scores(scores):
    nice_scores = {}
    for key, val in scores.items():
        if "test" in key:
            key = key.replace("test_","")

        nice_scores[key] = np.mean(val)
    return nice_scores

def merge_dictionaries(dicts):
    merged = {}
    for d in dicts:
        for key, val in d.items():
            if key not in merged:
                merged[key] = [val]
            else:
                merged[key].append(val)
    return merged

def run_eval(cfg):
    model, X, Y, scoring, cv, additional_infos, rid = cfg["model"], cfg["X"], cfg["Y"], cfg["scoring"], cfg["cv"], cfg["additional_infos"], cfg["run_id"]

    train, test = list(cv.split(X, Y))[rid]
    XTrain, YTrain = X[train,:], Y[train]
    XTest, YTest = X[test,:], Y[test]

    model.fit(XTrain, YTrain)

    scores = {}
    for name, method in scoring.items():
        scores["test_{}".format(name)] = method(model, XTest, YTest)
        scores["train_{}".format(name)] = method(model, XTrain, YTrain)

    return scores, additional_infos
    #return ("_".join([str(val) for val in additional_infos.values()]), beautify_scores(scores))

def prepare_xval(cfg, xval):
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
        X, Y = get_dataset(dataset)
        
        if X is None or Y is None: 
            exit(1)

        np.random.seed(12345)

        scoring = {
            "accuracy":accuracy,
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

        kf = StratifiedKFold(n_splits=args.xval, random_state=12345, shuffle=True)

        from collections import Counter
        print("Data: ", X.shape, " ", X[0:2,:])
        print("Labels: ", Y.shape, " ", Counter(Y))

        common_config = {
            "X": X,
            "Y": Y, 
            "scoring":scoring, 
            "cv":kf, 
        }
        n_jobs_per_forest = np.ceil(args.n_jobs / args.xval).astype(int)
        n_jobs_in_pool = max(1, args.n_jobs - n_jobs_per_forest)

        #client = Client(n_workers=50) #n_workers = args.n_jobs, threads_per_worker = 1, memory_limit = '32GB
        parallel_backend("threading") #, args.n_jobs

        print("Preparing experiments")
        configs = []
        for mn in args.max_nodes:
            for l in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            # for l in [0,0.5,1.0]:
                configs.extend(
                    prepare_xval(
                        {
                            **common_config,
                            "model":NCRandomForestClassifier(
                                base_forest = RandomForestClassifier(n_estimators = args.n_estimators, bootstrap = True, max_leaf_nodes=mn, n_jobs = n_jobs_per_forest, random_state=12345),
                                # n_jobs = 5 works well empirically
                                #min(n_jobs_per_forest,5)
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

            configs.extend(
                prepare_xval(
                    {
                        **common_config,
                        "model":RandomForestClassifierWithSampleSize(
                            n_estimators = args.n_estimators, bootstrap = True, max_leaf_nodes=mn, n_jobs = n_jobs_per_forest, random_state=12345
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
                            n_estimators = args.n_estimators, bootstrap = True, max_leaf_nodes=mn, n_jobs = n_jobs_per_forest, random_state=12345
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
                        "model":DecisionTreeClassifierWithSampleSize(max_leaf_nodes=mn, random_state=12345),
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
                        "model":DataAugmentationDecisionTreeClassifierWithSampleSize(max_leaf_nodes=mn, random_state=12345, n_jobs = n_jobs_per_forest),
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
        df = df.sort_values(by=["dataset","max_nodes", "method", "T"])

        with pd.option_context('display.max_rows', None): 
            print(df[["dataset", "method", "max_nodes", "effective_height", "n_leaves", "T", "train_accuracy", "accuracy", "n_nodes", "avg_rademacher", "bias", "diversity"]])
        df.to_csv("{}.csv".format(dataset),index=False)

if __name__ == '__main__':
    # sys.argv = ['run.py', '-x', '5', '-M', '32', '--max_nodes', '32', '--n_jobs', '96']
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--n_jobs", help="No of jobs for processing pool",type=int, default=1)
    parser.add_argument("--max_nodes", help="Maximum number of nodes of the trees. Corresponds to sci-kit learns max_nodes parameter. Can be a list of arguments for multiple experiments", nargs='+', type=int, default=[])
    parser.add_argument("-d", "--dataset", help="Dataset used for experiments",type=str, default=["magic"], nargs='+')
    parser.add_argument("-M", "--n_estimators", help="Number of estimators trained for the base learner.", type=int, default=32)
    parser.add_argument("-x", "--xval", help="Number of X-val runs",type=int, default=10)
    #parser.add_argument("--repeat", help="Number of times training data is repeated for approximators. ",type=int, default=5)
    args = parser.parse_args()

    if args.max_nodes is None or len(args.max_nodes) == 0:
        args.max_nodes = list(range(10,100,10)) + list(range(100,1000,50)) + list(range(1000,10000,1000))
    
    main(args)