# Code for the paper "There is no Double-Descent in Random Forest"

This repository contains the code to run the experiments for our paper called "There is no Double-Descent in Random Forest". In the paper we highlight experiments on the TODO dataset, but this implementation also supports more datasets out of the box. Most of the code should be somewhat commented and self-explanatory given the two caveats below. To run the experiments simply clone this repository

    git@github.com:sbuschjaeger/rf-double-descent.git

(Optional) Build the conda environment and activate it:

    conda env creat -f environment.yml --force
    conda activate rfdd

Run experiments on the `adult` dataset with M = 256 trees over a 5 fold cross validation with different number of `max_nodes` with 96 threads:

    ./run.py -x 5 -M 256 --n_jobs 96 --max_nodes 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 -d adult

**Important 1**: This will run all experiments with `96` threads. The experiments are executed in a `multiprocessing.Pool` environment which means that the *entire* dataset is copied for each cross-validation run. Hence this may take a decent amount of memory (up to 200GB) and some time. 

**Important 2**: The command-line argument `n_jobs` only determines the total number of threads the processing pool, but *not* the total number of threads used by this script. We currently supply `n_jobs = n_jobs_per_forest = None` to scikit-learns `RandomForestClassifier` when fitting the (initial) Random Forst. Hence, scikit-learns uses a heuristic to choose the number of jobs used for fitting the RF. If required, then you can set `n_jobs_per_forest` in the script manually (line 132). 

**Important 3**: Datasets which are not found in the tempfolder (issued by `tempfile.gettmpdir()` which likely points to `/tmp` on Linux systems) are automatically downloaded. If you have already downloaded the datasets or you simply do not like the temp folder you can set this via `--tmpdir ${your_new_tmp_dir}`.

Plot the results on the `adult` dataset and store the them in the current folder:

    ./plot.py -d adult -o .

Alternativley, `plot.py` is also divided into execution cells which you can run via an inline interpreter (e.g. VSCode or a Juypter Notebook).