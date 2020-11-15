import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from time_series_experiments.pipeline.tasks import Wrap, TaskData, OrdCat, OneHot


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


def test_ordcat_task():
    x1 = np.random.choice(["a", "b", "c"], size=1000)
    x2 = np.random.choice(["1", "2", "3", "4", "5", "6"], size=1000)

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    data = TaskData(X=x, column_names=["x1", "x2"], column_types=[0, 0])

    task = OrdCat()
    res = task.fit_transform(data)
    assert res.column_names == ["x1", "x2"]
    assert res.column_types == [3, 6]

    expected = OrdinalEncoder().fit_transform(data.X)
    assert np.all(np.isclose(res.X, expected))


def test_onehot_task():
    x1 = np.random.choice(["a", "b", "c"], size=1000)
    x2 = np.random.choice(["1", "2", "3", "4", "5", "6"], size=1000)

    x = np.hstack([np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1))])
    data = TaskData(X=x, column_names=["x1", "x2"], column_types=[0, 0])

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
    assert all([x == 0 for x in res.column_types])

    expected = OneHotEncoder().fit_transform(data.X)
    assert np.all(np.isclose(res.X.todense(), expected.todense()))
