import pytest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from time_series_experiments.pipeline.tasks import (
    Wrap,
    TaskData,
    OrdCat,
    OneHot,
    DateFeatures,
    TargetLag,
)
from time_series_experiments.pipeline.data import take_columns, ColumnType
from time_series_experiments.pipeline.dataset import VarType


def test_imputer_wrapper():
    x = np.random.random((1000, 1))
    nans = np.random.choice(x.shape[0], size=100)
    x[nans] = np.nan

    data = TaskData(X=x, column_names=["x"], column_types=[0])

    task = Wrap(SimpleImputer(strategy="constant", fill_value=-1))
    res = task.fit_transform(data)
    assert np.unique(res.X[nans]).shape[0] == 1
    assert np.unique(res.X[nans])[0] == -1

    task = Wrap(SimpleImputer(strategy="mean"))
    res = task.fit_transform(data)
    assert np.unique(res.X[nans]).shape[0] == 1
    assert np.isclose(np.unique(res.X[nans])[0], np.mean(x[~np.isnan(x)]))

    task = Wrap(SimpleImputer(strategy="median", add_indicator=True))
    res = task.fit_transform(data)
    assert res.X.shape[1] == 2
    assert np.all(np.isclose(np.unique(res.X[:, 1][nans]), np.array([1])))
    assert np.isclose(np.unique(res.X[:, 0][nans])[0], np.median(x[~np.isnan(x)]))


def test_imputer_wrapper_multiple_cols():
    xs = []
    for i in range(3):
        x = np.random.random((1000, 1))
        nans = np.random.choice(x.shape[0], size=100)
        x[nans] = np.nan
        xs.append(x)
    x = np.concatenate(xs, axis=1)

    data = TaskData(X=x, column_names=["x1", "x2", "x3"], column_types=[0])

    task = Wrap(SimpleImputer(strategy="median", add_indicator=True))
    res = task.fit_transform(data)
    assert res.X.shape[1] == 6
    assert res.column_names == ["SimpleImputer-{}".format(i) for i in range(6)]


@pytest.mark.parametrize("use_other", [True, False])
def test_ordcat_task(use_other):
    x1 = np.random.choice(["a", "b", "c"], size=1000)
    x2 = np.random.choice(["1", "2", "3", "4", "5", "6"], size=1000)

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    data = TaskData(
        X=x,
        column_names=["x1", "x2"],
        column_types=[ColumnType(VarType.NUM), ColumnType(VarType.NUM)],
    )

    task = OrdCat(min_support=0, use_other=use_other, handle_unknown="error")
    res = task.fit_transform(data)
    assert res.column_names == ["x1", "x2"]
    assert res.column_types == [
        ColumnType(VarType.CAT, level=5 if use_other else 4),
        ColumnType(VarType.CAT, level=8 if use_other else 7),
    ]

    expected = OrdinalEncoder().fit_transform(data.X)
    if use_other:
        expected = expected + 2
    else:
        expected = expected + 1
    assert np.all(np.isclose(res.X, expected))


def test_ordcat_task_handle_unknown():
    x1 = np.random.choice(["a", "b", "c"], size=1000)
    x2 = np.random.choice(["1", "2", "3", "4", "5", "6"], size=1000)

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    data = TaskData(
        X=x,
        column_names=["x1", "x2"],
        column_types=[ColumnType(VarType.NUM), ColumnType(VarType.NUM)],
    )

    task = OrdCat(min_support=0, use_other=False, handle_unknown="missing")
    res = task.fit_transform(data)
    assert res.column_names == ["x1", "x2"]
    assert res.column_types == [
        ColumnType(VarType.CAT, level=4),
        ColumnType(VarType.CAT, level=7),
    ]

    expected = OrdinalEncoder().fit_transform(data.X)
    expected = expected + 1
    assert np.all(np.isclose(res.X, expected))

    # transform with new categories
    x1 = np.random.choice(["a", "c", "d"], size=1000)
    x2 = np.random.choice(["2", "3", "5", "6", "7"], size=1000)

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    new_data = TaskData(
        X=x,
        column_names=["x1", "x2"],
        column_types=[ColumnType(VarType.NUM), ColumnType(VarType.NUM)],
    )
    res = task.transform(new_data)

    mask = x1 == "d"
    results = res.X[:, 0][mask]
    assert np.unique(results) == np.array([0])

    mask = x2 == "7"
    results = res.X[:, 1][mask]
    assert np.unique(results) == np.array([0])


def test_ordcat_task_handle_unknown_error():
    x1 = np.random.choice(["a", "b", "c"], size=1000)
    x2 = np.random.choice(["1", "2", "3", "4", "5", "6"], size=1000)

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    data = TaskData(
        X=x,
        column_names=["x1", "x2"],
        column_types=[ColumnType(VarType.NUM), ColumnType(VarType.NUM)],
    )

    task = OrdCat(min_support=0, use_other=False, handle_unknown="error")
    res = task.fit_transform(data)
    assert res.column_names == ["x1", "x2"]
    assert res.column_types == [
        ColumnType(VarType.CAT, level=4),
        ColumnType(VarType.CAT, level=7),
    ]

    expected = OrdinalEncoder().fit_transform(data.X)
    expected = expected + 1
    assert np.all(np.isclose(res.X, expected))

    # transform with new categories
    x1 = np.random.choice(["a", "c", "d"], size=1000)
    x2 = np.random.choice(["2", "3", "5", "6", "7"], size=1000)

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    new_data = TaskData(
        X=x,
        column_names=["x1", "x2"],
        column_types=[ColumnType(VarType.NUM), ColumnType(VarType.NUM)],
    )
    with pytest.raises(ValueError) as excinfo:
        res = task.transform(new_data)
    assert "Found unknown categories" == str(excinfo.value)


def test_onehot_task():
    x1 = np.random.choice(["a", "b", "c"], size=1000)
    x2 = np.random.choice(["1", "2", "3", "4", "5", "6"], size=1000)

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    data = TaskData(
        X=x,
        column_names=["x1", "x2"],
        column_types=[ColumnType(VarType.NUM), ColumnType(VarType.NUM)],
    )

    task = OneHot()
    res = task.fit_transform(data)
    assert res.column_names == [
        "x1_0",
        "x1_1",
        "x1_2",
        "x2_0",
        "x2_1",
        "x2_2",
        "x2_3",
        "x2_4",
        "x2_5",
    ]
    assert all([x == ColumnType(VarType.NUM) for x in res.column_types])

    expected = OneHotEncoder().fit_transform(data.X)
    assert np.all(np.isclose(res.X.todense(), expected.todense()))


def test_date_features_extractor_task():
    x1 = pd.date_range(start="2020-10-17", periods=5000, freq="5D")
    x2 = pd.date_range(start="2007-06-06", periods=5000, freq="21S")

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    data = TaskData(
        X=x,
        column_names=["x1", "x2"],
        column_types=[ColumnType(VarType.NUM), ColumnType(VarType.NUM)],
    )

    task = DateFeatures()
    res = task.fit_transform(data)

    assert res.X.shape[1] == len(task.COMPONENTS) * 2

    x1_cols = [c for c in res.column_names if c.startswith("x1")]
    x2_cols = [c for c in res.column_names if c.startswith("x2")]

    x1_data = take_columns(res, x1_cols)
    x2_data = take_columns(res, x2_cols)

    extractors = {
        "year": lambda x: x.year,
        "month": lambda x: x.month,
        "week": lambda x: x.week,
        "day_of_month": lambda x: x.day,
        "day_of_week": lambda x: x.dayofweek,
        "hour": lambda x: x.hour,
        "minute": lambda x: x.minute,
        "second": lambda x: x.second,
    }
    for actual, expected, name in [(x1_data, x1, "x1"), (x2_data, x2, "x2")]:
        for k, v in extractors.items():
            feature_name = "{} - {}".format(name, k)
            assert feature_name in actual.column_names
            expected_val = np.array([v(x) for x in expected])
            idx = actual.column_names.index(feature_name)
            assert np.all(np.isclose(actual.X[:, idx], expected_val))

            idx = actual.column_names.index(feature_name)
            col_type = actual.column_types[idx]

            if "year" in feature_name:
                assert col_type.var_type == VarType.NUM
            else:
                assert col_type.var_type == VarType.CAT


@pytest.mark.parametrize("order", [1, 3, 5])
def test_target_lag(order):
    x = np.random.random((1000,))
    y = np.random.random((1000,))

    data = TaskData(
        X=x, y=y, column_names=["x"], column_types=[ColumnType(VarType.NUM)],
    )

    df = pd.DataFrame({"x": x, "y": y})
    df["lag"] = df["y"].shift(order)
    lag = df["lag"].values

    task = TargetLag(order=order, handle_nan="drop")
    new_data = task.fit_transform(data)

    lag_index = new_data.column_names.index(TargetLag.PATTERN.format(order))
    assert np.all(np.isclose(lag[order:], new_data.X[:, lag_index]))
    assert len(new_data.column_names) == 2
    assert len(new_data.column_types) == 2
    assert new_data.column_names[-1] == TargetLag.PATTERN.format(order)
    assert new_data.column_types[-1] == ColumnType(VarType.LAG)
