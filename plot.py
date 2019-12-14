from __future__ import division  # For Python 2
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import util

FILE_DIR = Path(__file__).resolve().parent
PLOT_DIR = FILE_DIR.joinpath("plots")
DATA_DIR = FILE_DIR.joinpath("results/cifar10")

DASH_STYLES = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)]


def parse_arguments():
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data', '-d', dest='data_dir', default=DATA_DIR)
    args = PARSER.parse_args()
    return args


def parse_config(conf_dir, name):
    file_name = conf_dir.joinpath(f"{name}.json")
    with open(file_name) as conf_file:
        return json.load(conf_file)


def np_dict_to_pd(np_dict, key, df_type="float64"):
    pd_frame = pd.DataFrame(
        {k: pd.Series(v) for k, v in np_dict[key].items()})
    return pd_frame.astype(df_type)


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L),
                                           strides=(S * n, n))


def compute_rolling_df_mean(pd_df, roll):
    rolling_df = pd_df.rolling(roll).mean().dropna()
    return rolling_df.reset_index(drop=True)


def compute_rolling_df_99p(pd_df, roll):
    rolling_df = pd_df.rolling(roll, center=True).quantile(.01).dropna()
    return rolling_df.reset_index(drop=True)


def normalize_df_min_max(pd_df):
    df_max = np.nanmax(pd_df.values)
    df_min = np.nanmin(pd_df.values)
    normalized_df = (pd_df - df_min) / (df_max - df_min)
    return normalized_df


def normalize_df_min_max_range(pd_df, df_min, df_max):
    normalized_df = (pd_df - df_min) / (df_max - df_min)
    return normalized_df


def normalize_df_tanh(pd_df, df_min, df_max):
    df_mean = np.mean(pd_df.values)
    df_std = np.std(pd_df.values)
    normalized_df = np.tanh(0.01(pd_df - df_mean) / df_std + 1)
    return normalized_df


def normalize_df_z_score(pd_df):
    df_mean = np.nanmean(pd_df.values)
    df_std = np.nanstd(pd_df.values)
    normalized_df = (pd_df - df_mean) / df_std
    return normalized_df


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop("Training Loss", axis=1)
    df.index.name = "Epoch"
    return df


def plot_results(data_dir, plot_dir=PLOT_DIR, test_id=""):
    data_dir = Path(f"{data_dir}")
    conf_name = "test_config"
    config = parse_config(data_dir, conf_name)
    modes = config["modes"]
    epochs = config["epochs"]
    teacher_name = config["teacher_name"] + "_teacher"
    student_name = config["student_name"]
    dfs = {}
    for mode in modes:
        mode = mode.lower()
        mode_path = data_dir.joinpath(mode)
        csv_path = mode_path.joinpath(f"{student_name}_train.csv")
        try:
            dfs[mode] = read_csv(csv_path)
        except FileNotFoundError:
            print(f"Results for {mode} not found, ignoring...")
    teacher_path = data_dir.joinpath(f"{teacher_name}_val.csv")
    dfs["teacher"] = read_csv(teacher_path)
    df = pd.concat(dfs.values(), axis=1, keys=dfs.keys())
    print(df.max().sort_values(ascending=True))
    df = compute_rolling_df_mean(df, 10)
    if (len(modes) + 1) > len(DASH_STYLES):
        print("Too many lines to plot!")
        return

    sns.lineplot(data=df, palette="tab10",
                 style="event", dashes=DASH_STYLES)
    plot_dir = Path(plot_dir).joinpath(test_id)
    util.check_dir(plot_dir)
    plt_name = f"{epochs}_epochs_{teacher_name}_to_{student_name}"
    plt_name = Path(plot_dir).joinpath(plt_name)
    plt.savefig(f"{plt_name}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f"{plt_name}.png", bbox_inches='tight', pad_inches=0.05)
    plt.gcf().clear()


if __name__ == '__main__':
    args = parse_arguments()
    plot_results(args.data_dir)
