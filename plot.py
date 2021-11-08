# %%
from typing import final
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize 
import os
import json 
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator

from Metrics import diversity

def error_leaf_plot(df, methods, name, out_path):
    colors = ["b", "g"]

    fig = plt.figure()
    plt.xscale("log")

    for m, c in zip(methods, colors):
        dff = df.copy()
        
        dff = dff.loc[dff["method"] == m]
        dff.sort_values(["max_nodes"], inplace=True)
        plt.plot(dff["max_nodes"].values, dff["test_error"].values, "{}-".format(c))
        plt.plot(dff["max_nodes"].values, dff["train_error"].values, "{}--".format(c))

    legend = []
    for m in methods:
        legend.append("{} test".format(m))
        legend.append("{} train".format(m))

    #plt.legend(legend, loc="upper right", bbox_to_anchor=(1.32, 1))
    plt.legend(legend) #loc="upper right"
    plt.xlabel("Maximum number of leaf nodes")
    plt.ylabel("Error")
    plt.show()
    fig.savefig(os.path.join(out_path, "{}.pdf".format(name)), bbox_inches='tight')

def rademacher_leaf_plot(df, methods, name, out_path):
    colors = ["b", "g"]

    fig = plt.figure()
    plt.xscale("log")

    for m, c in zip(methods, colors):
        dff = df.copy()
        
        dff = dff.loc[dff["method"] == m]
        dff.sort_values(["max_nodes"], inplace=True)

        plt.plot(dff["max_nodes"].values, dff["avg_rademacher"].values, "{}-".format(c))
        
    plt.legend(methods, loc="lower right")
    plt.xlabel("Maximum number of leaf nodes")
    plt.ylabel("Asymptotic Rademacher Complexity")
    plt.show()
    fig.savefig(os.path.join(out_path, "{}.pdf".format(name)), bbox_inches='tight')

def error_rademacher_plot(df, methods, name, out_path):
    colors = ["b", "g", "r", "y"]

    fig = plt.figure()
    #plt.xscale("log")

    for m, c in zip(methods, colors):
        dff = df.copy()
        
        dff = dff.loc[dff["method"] == m]
        dff.sort_values(["avg_rademacher"], inplace=True)
        plt.plot(dff["avg_rademacher"].values, dff["test_error"].values, "{}-".format(c))
        plt.plot(dff["avg_rademacher"].values, dff["train_error"].values, "{}--".format(c),label='_nolegend_')

    legend = []
    for m in methods:
        legend.append("{}".format(m))
        #legend.append("{} test".format(m))
        #legend.append("{} train".format(m))

    plt.legend(legend, loc="upper right")
    plt.xlabel("Asymptotic Rademacher Complexity")
    plt.ylabel("Error")
    plt.show()
    fig.savefig(os.path.join(out_path, "{}.pdf".format(name)), bbox_inches='tight')

def height_leaf_plot(df, methods, name, out_path):
    colors = ["b", "g"]

    fig = plt.figure()
    plt.xscale("log")

    for m, c in zip(methods, colors):
        dff = df.copy()
        
        dff = dff.loc[dff["method"] == m]
        dff.sort_values(["max_nodes"], inplace=True)

        plt.plot(dff["max_nodes"].values, dff["effective_height"].values, "{}-".format(c))
        
    plt.legend(methods, loc="lower right")
    plt.xlabel("Maximum number of leaf nodes")
    plt.ylabel("Average Height of trees")
    plt.show()
    fig.savefig(os.path.join(out_path, "{}.pdf".format(name)), bbox_inches='tight')

def diversity_plots(df, max_nodes, names, out_path):
    colors = ["b", "g", "r", "y"]
    
    for n in max_nodes:
        dff = df.copy()
        dff = dff.loc[dff["max_nodes"] == n]
        rf_row = dff.loc[ dff["method"] == "RF" ]
        dt_row = dff.loc[ dff["method"] == "DT" ]
        dff = dff[dff['method'].str.contains('NCF')]
        dff[["base_mehod", "lambda"]] = dff["method"].str.split("NCF-", expand=True)

        dff["lambda"] = dff["lambda"].astype(float)
        dff.sort_values(by=["lambda"],inplace=True)

        mind = None
        idx = 0
        for i in range(len(dff)):
            v = dff.iloc[i]
            d = (v["bias"] - rf_row["bias"].values[0])**2 + (v["diversity"] - rf_row["diversity"].values[0])**2 + (v["loss"] - rf_row["loss"].values[0])**2
            if mind is None or d < mind:
                mind = d
                idx = i
        
        rf = dff.iloc[idx]

        fig,ax = plt.subplots(1,1)
        plt.yscale("log")
        plt.plot(dff["lambda"], dff["bias"])
        plt.plot(dff["lambda"], dff["diversity"])
        plt.plot(dff["lambda"], dff["loss"])
        plt.legend([r'Bias $\frac{1}{M}\sum_{i=1}^M (h_i(x) - y)^2$', r'Diversity $\frac{1}{2M} \sum_{i=1}^M {d_i}^T D d_i$', r'Ensemble loss $(f(x)-y)^2$'], loc="upper left")
        plt.ylabel("Mean-Squared-Error")
        plt.xlabel("λ")
        plt.show()
        fig.savefig(os.path.join(out_path, "{}-{}.pdf".format(names[0], n)), bbox_inches='tight')

        selecteddf = dff.loc[ dff["lambda"].between(-3,1)]
        fig,ax = plt.subplots(1,1)
        #plt.yscale("log")
        plt.plot(selecteddf["lambda"], selecteddf["bias"])
        plt.plot(selecteddf["lambda"], selecteddf["diversity"])
        plt.plot(selecteddf["lambda"], selecteddf["loss"])
        #plt.scatter(rf["lambda"], rf["loss"], color="black", s=32, marker="x", zorder=3)
        plt.legend([r'Bias $\frac{1}{M}\sum_{i=1}^M (h_i(x) - y)^2$', r'Diversity $\frac{1}{2M} \sum_{i=1}^M {d_i}^T D d_i$', r'Ensemble loss $(f(x)-y)^2$'], loc="upper left")
        plt.ylabel("Mean-Squared-Error")
        plt.xlabel("λ")
        plt.show()
        fig.savefig(os.path.join(out_path, "{}-{}.pdf".format(names[1], n)), bbox_inches='tight')

        fig,ax = plt.subplots(1,1)
        plt.xscale("log")
        plt.plot(dff["diversity"], dff["test_error"], "b-")
        plt.plot(dff["diversity"], dff["avg_error"], "g-")
        plt.plot(dff["diversity"], dff["train_error"], "b--")
        plt.plot(dff["diversity"], dff["train_avg_error"], "g--")

        plt.text(rf.diversity-0.02,rf.test_error+0.01,'RF')
        plt.scatter(rf.diversity, rf.test_error, color="black", s=32, marker="x", zorder=3)
        plt.text(dff["diversity"].min()-1.7e-3,dt_row["test_error"].values[0]+0.01,'DT',color="black")
        plt.scatter(dff["diversity"].min()-1e-3, dt_row["test_error"].values[0], color="black", s=32, marker="x", zorder=3)

        plt.legend(["test","avg test", "train", "avg train"])# loc="center right"
        plt.xlabel("Diversity")
        plt.ylabel("Error")
        plt.show()
        fig.savefig(os.path.join(out_path, "{}-{}.pdf".format(names[2], n)), bbox_inches='tight')

def cbound_plots(df, max_nodes, out_path):
    colors = ["b", "g", "r", "y"]
    
    fig = plt.figure()
    legend = []
    for n in max_nodes:
        dff = df.copy()
        dff = dff.loc[dff["max_nodes"] == n]
        dff = dff[dff['method'].str.contains('NCF')]

        dff[["base_mehod", "lambda"]] = dff["method"].str.split("-", expand=True)
        plt.plot(dff["train_cbound"], dff["test_error"])
        #legend.append("Max n_l {}".format(n))
        legend.append(r'$n_l = {}$'.format(n))
    plt.legend(legend, loc="upper left")
    plt.xlabel("C-Bound")
    plt.ylabel("Error")
    plt.show()
    fig.savefig("{}-cbound.pdf".format(out_path, n), bbox_inches='tight')


#%%
def plot_all(dataset, out_path):
    out_path = os.path.join(out_path, dataset)
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    df = pd.read_csv("{}.csv".format(dataset))

    df["test_error"] = 1.0 - df["accuracy"] / 100.0
    df["avg_error"] = 1.0 - df["avg_accuracy"] / 100.0
    df["train_error"] = 1.0 - df["train_accuracy"] / 100.0
    df["train_avg_error"] = 1.0 - df["train_avg_accuracy"] / 100.0

    error_leaf_plot(df, ["DT", "RF"], "h1a", out_path)
    rademacher_leaf_plot(df, ["DT", "RF"], "h1b", out_path)
    height_leaf_plot(df, ["DT", "RF"], "h1c", out_path)

    error_leaf_plot(df, ["DA-DT", "DA-RF"], "h2a", out_path)
    rademacher_leaf_plot(df, ["DA-DT", "DA-RF"], "h2b", out_path)
    height_leaf_plot(df, ["DA-DT", "DA-RF"], "h2c", out_path)

    error_rademacher_plot(df, ["DT", "RF", "DA-DT", "DA-RF"], "h3", out_path)

    diversity_plots(df, [4096], ["h4a", "h4b", "h4c"], out_path)

# eeg, adult, magic, bank, nomao
for d in ["eeg", "adult", "magic", "bank", "nomao"]:
    plot_all(d, "plots/")


# %%
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset to plot. ",type=str, default="magic", type=str)
    parser.add_argument("-o", "--out", help="Folder to store plots",type=str, default=".", type=str)
    args = parser.parse_args()

    plot_all(args.data, args.out_path)
