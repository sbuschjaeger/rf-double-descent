# %%
from typing import final
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize 
import os
import json 
from IPython.display import display, HTML
import matplotlib.pyplot as plt

def error_leaf_plot(df, methods, out_path):
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

    plt.legend(legend, loc="upper right", bbox_to_anchor=(1.32, 1))
    plt.xlabel("Maximum number of leaf nodes")
    plt.ylabel("Error")
    plt.show()
    fig.savefig("{}.pdf".format(out_path), bbox_inches='tight')

df = pd.read_csv("magic.csv")

df["test_error"] = 1.0 - df["accuracy"] / 100.0
df["train_error"] = 1.0 - df["train_accuracy"] / 100.0

error_leaf_plot(df, ["DT", "RF"], "h1")
error_leaf_plot(df, ["DA-DT", "DA-RF"], "h3")

# display(df)

# %%

def rademacher_leaf_plot(df, methods, out_path):
    colors = ["b", "g"]

    fig = plt.figure()
    plt.xscale("log")

    for m, c in zip(methods, colors):
        dff = df.copy()
        
        dff = dff.loc[dff["method"] == m]
        dff.sort_values(["max_nodes"], inplace=True)

        plt.plot(dff["max_nodes"].values, dff["avg_rademacher"].values, "{}-".format(c))
        
    plt.legend(methods, loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.xlabel("Maximum number of leaf nodes")
    plt.ylabel("Asymptotic Rademacher")
    plt.show()
    fig.savefig("{}.pdf".format(out_path), bbox_inches='tight')

rademacher_leaf_plot(df, ["DT", "RF"], "h2")
rademacher_leaf_plot(df, ["DA-DT", "DA-RF"], "h4")

# %%

def error_rademacher_plot(df, methods, out_path):
    colors = ["b", "g", "r", "y"]

    fig = plt.figure()
    #plt.xscale("log")

    for m, c in zip(methods, colors):
        dff = df.copy()
        
        dff = dff.loc[dff["method"] == m]
        dff.sort_values(["avg_rademacher"], inplace=True)
        plt.plot(dff["avg_rademacher"].values, dff["test_error"].values, "{}-".format(c))
        plt.plot(dff["avg_rademacher"].values, dff["train_error"].values, "{}--".format(c))

    legend = []
    for m in methods:
        legend.append("{} test".format(m))
        legend.append("{} train".format(m))

    plt.legend(legend, loc="upper right", bbox_to_anchor=(1.32, 1))
    plt.xlabel("Asymptotic Rademacher")
    plt.ylabel("Error")
    plt.show()
    fig.savefig("{}.pdf".format(out_path), bbox_inches='tight')

df = pd.read_csv("magic.csv")

df["test_error"] = 1.0 - df["accuracy"] / 100.0
df["train_error"] = 1.0 - df["train_accuracy"] / 100.0

error_rademacher_plot(df, ["DT", "RF", "DA-DT", "DA-RF"], "h5")
#error_rademacher_plot(df, ["DA-DT", "DA-RF"], "h6")
