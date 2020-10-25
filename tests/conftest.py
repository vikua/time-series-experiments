import numpy as np
import pandas as pd

from time_series_experiments.utils.data import train_test_split_index


RANDOM_SEED = 0xC0FFEE


def generate_walk(n, datatype="float", random_seed=RANDOM_SEED):
    if isinstance(n, int):
        n = (n,)
    random_state = np.random.RandomState(random_seed)
    if datatype == "int":
        values = random_state.randint(low=0, high=n[0], size=n)
    else:
        values = random_state.randn(*n).cumsum(axis=0)
        values = (values - np.mean(values, axis=0)) / np.std(values, axis=0)
    return values


def generate_target(
    nrows, freq="1H", start="2018-01-01 00:00:00", random_seed=RANDOM_SEED
):
    t = pd.date_range(start=start, freq=freq, periods=nrows)
    n = len(t)
    t0 = pd.Timestamp("2018-01-01 00:00:00")
    t_day = (t - t0) / pd.Timedelta(1, "D")
    t_week = (t - t0) / pd.Timedelta(7, "D")
    t_year = (t - t0) / pd.Timedelta(365, "D")
    y = generate_walk(n, random_seed=random_seed)
    y += np.sin(2 * np.pi * t_year)
    y += 0.5 * np.sin(2 * np.pi * t_week)
    y += 0.25 * np.sin(2 * np.pi * t_day)
    y += 0.05 * np.std(y) * np.random.randn(n)
    return y


def simple_seq_data(nrows, freq, fdw, fw, test_size):
    y = generate_target(nrows, freq=freq)
    x_train_idx, y_train_idx, x_test_idx, y_test_idx = train_test_split_index(
        y.shape[0], fdw, fw, test_size, random_seed=RANDOM_SEED
    )
    x_train = y[x_train_idx]
    y_train = y[y_train_idx]
    x_test = y[x_test_idx]
    y_test = y[y_test_idx]
    return (
        np.expand_dims(x_train, axis=-1),
        y_train,
        np.expand_dims(x_test, axis=-1),
        y_test,
    )
