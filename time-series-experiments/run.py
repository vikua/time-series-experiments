import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd

from data import train_test_split_index


def train(args):
    if args.data.endswith(".xlsx"):
        data = pd.read_excel(args.data)
    else:
        data = pd.read_csv(args.data)

    target = data[args.target].values

    x_train_idx, y_train_idx, x_test_idx, y_test_idx = train_test_split_index(
        target.shape[0], args.fdw, args.fw, args.test_size, args.seed
    )
    x_train = target[x_train_idx]
    y_train = target[y_train_idx]
    x_test = target[x_test_idx]
    y_test = target[y_test_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test transformer")
    parser.add_argument(
        "--data", help="path to input data", required=True,
    )
    parser.add_argument(
        "--target", help="target column name", required=True,
    )
    parser.add_argument(
        "--test_size", default=0.2, type=float, help="pct of test data",
    )
    parser.add_argument(
        "--fdw", default=28, type=int, help="number of steps to look back",
    )
    parser.add_argument(
        "--fw", default=7, type=int, help="number of steps to predict into the future",
    )
    parser.add_argument("--seed", default=0xC0FFEE, type=int, help="random seed")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train(args)
