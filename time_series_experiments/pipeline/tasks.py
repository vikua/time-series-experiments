import abc

import attr
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator

from .data import TaskData, ColumnType
from .dataset import VarType


class Task(abc.ABC):
    @abc.abstractmethod
    def fit(self, data: TaskData):
        pass

    @abc.abstractmethod
    def transform(self, data: TaskData) -> TaskData:
        pass

    def fit_transform(self, data: TaskData) -> TaskData:
        return self.fit(data).transform(data)


class Wrap(Task):
    def __init__(self, task: BaseEstimator, type_override: VarType = None):
        self._task = task
        self._type_override = type_override

    def fit(self, data: TaskData) -> Task:
        self._task.fit(data.X, data.y)
        return self

    def transform(self, data: TaskData) -> TaskData:
        _X = self._task.transform(data.X)

        column_names = data.column_names
        column_types = data.column_types

        new_cols = _X.shape[1] - data.X.shape[1]
        if new_cols != 0:
            column_names = [
                "{}-{}".format(type(self._task).__name__, i) for i in range(_X.shape[1])
            ]
            column_types = [ColumnType(VarType.NUM)] * len(column_names)

        if self._type_override is not None:
            column_types = [ColumnType(self._type_override) for _ in column_types]

        return attr.evolve(
            data, X=_X, column_types=column_types, column_names=column_names
        )


class OrdCat(Task):
    def __init__(
        self,
        min_support=5,
        use_other=True,
        ordering="lexografical",
        handle_unknown="missing",
    ):
        self._min_suppoort = min_support
        self._use_other = use_other
        self._ordering = ordering
        self._handle_unknown = handle_unknown

        self._missing_category = 0
        self._other_category = 0
        if self._use_other:
            self._other_category = 1

        if self._ordering not in ["lexografical", "frequency"]:
            raise ValueError("Unknown value {} for ordering".format(self._ordering))

        if self._handle_unknown not in ["error", "missing"]:
            raise ValueError(
                "Unknown value {} for handle_unknown".format(self._handle_unknown)
            )

        self.mapping_ = dict()
        self.levels_ = dict()

    def fit(self, data: TaskData) -> Task:
        for i in range(data.X.shape[1]):
            vals, counts = np.unique(data.X[:, i], return_counts=True)

            if self._ordering == "frequency":
                order = np.argsort(counts)
            else:
                order = np.argsort(vals)

            vals = vals[order]
            counts = counts[order]

            categories = np.arange(vals.shape[0]) + self._other_category + 1

            categories[counts < self._min_suppoort] = self._other_category
            self.mapping_[i] = {val: cat for val, cat in zip(vals, categories)}

            categories = set(list(categories) + [0] + [self._other_category])
            self.levels_[i] = list(sorted(categories))

        return self

    def transform(self, data: TaskData) -> TaskData:
        if len(self.mapping_.keys()) != data.X.shape[1]:
            raise ValueError("Unexpected number of columns")

        _X = []
        levels = []
        for i in range(data.X.shape[1]):
            cats = self.mapping_[i].copy()
            arr = data.X[:, i]

            vals = np.unique(arr)
            unknown_categories = {
                v: self._missing_category for v in vals if v not in cats
            }
            if unknown_categories and self._handle_unknown == "error":
                raise ValueError("Found unknown categories")

            cats.update(unknown_categories)
            keys = list(cats.keys())
            values = list(cats.values())

            sort_idx = np.argsort(keys)
            idx = np.searchsorted(keys, arr, sorter=sort_idx)
            out = np.asarray(values)[sort_idx][idx]

            _X.append(out)
            levels.append(len(self.levels_[i]))

        _X = np.vstack(_X).T

        column_types = [ColumnType(VarType.CAT, level=x) for x in levels]
        return attr.evolve(data, X=_X, column_types=column_types)


class OneHot(Task):
    def __init__(self, handle_unknown="error"):
        self._handle_unknown = handle_unknown
        self._enc = None

    def fit(self, data: TaskData) -> Task:
        self._enc = OneHotEncoder(handle_unknown=self._handle_unknown)
        self._enc.fit(data.X)
        return self

    def transform(self, data: TaskData) -> TaskData:
        _X = self._enc.transform(data.X)
        column_names = []
        for col, categories in zip(data.column_names, self._enc.categories_):
            new_cols = ["{}_{}".format(col, i) for i in range(len(categories))]
            column_names.extend(new_cols)
        column_types = [ColumnType(VarType.NUM) for _ in column_names]
        return attr.evolve(
            data, X=_X, column_names=column_names, column_types=column_types
        )


class DateFeatures(Task):
    COMPONENTS = {
        "year": VarType.NUM,
        "month": VarType.CAT,
        "week": VarType.CAT,
        "day_of_month": VarType.CAT,
        "day_of_week": VarType.CAT,
        "hour": VarType.CAT,
        "minute": VarType.CAT,
        "second": VarType.CAT,
    }

    EXTRACTORS = {
        "year": np.vectorize(lambda x: x.year),
        "month": np.vectorize(lambda x: x.month),
        "week": np.vectorize(lambda x: x.week),
        "day_of_month": np.vectorize(lambda x: x.day),
        "day_of_week": np.vectorize(lambda x: x.dayofweek),
        "hour": np.vectorize(lambda x: x.hour),
        "minute": np.vectorize(lambda x: x.minute),
        "second": np.vectorize(lambda x: x.second),
    }

    PATTERN = "{} - {}"

    def __init__(self, components=None):
        if not components:
            components = list(self.COMPONENTS.keys())

        unknown = set(components) - set(self.COMPONENTS.keys())
        if unknown:
            raise ValueError("Unknown datetime components {}".format(unknown))

        self._components = components

    def fit(self, data: TaskData) -> Task:
        return self

    def transform(self, data: TaskData) -> TaskData:
        X = data.X
        converter = np.vectorize(lambda x: pd.Timestamp(x))
        X = converter(X)

        _X = []
        column_names = []
        column_types = []
        for component in self._components:
            func = self.EXTRACTORS[component]
            _X.append(func(X))
            column_names.extend(
                [self.PATTERN.format(c, component) for c in data.column_names]
            )
            column_types.extend(
                [ColumnType(self.COMPONENTS[component]) for _ in range(X.shape[1])]
            )

        _X = np.concatenate(_X, axis=1)
        return attr.evolve(
            data, X=_X, column_names=column_names, column_types=column_types
        )


class TargetLag(Task):
    PATTERN = "lag_{}"

    def __init__(self, order=1, handle_nan="drop"):
        self._order = order
        self._handle_nan = handle_nan

        if self._handle_nan not in ["drop", "impute"]:
            raise ValueError(
                "Unknown value {} for handle_nan param".format(self._handle_nan)
            )

        self._impute_val = None

    def fit(self, data: TaskData) -> Task:
        self._impute_val = np.mean(data.y)
        return self

    def transform(self, data: TaskData) -> TaskData:
        X = data.X
        y = data.y

        lag = np.roll(y, self._order)
        np.put(lag, range(self._order), np.nan)

        mask = np.isnan(lag)
        if self._handle_nan == "drop":
            X = X[~mask]
            y = y[~mask]
            lag = lag[~mask]
        elif self._handle_nan == "impute":
            lag[mask] = self._impute_val

        column_names = data.column_names + [self.PATTERN.format(self._order)]
        column_types = data.column_types + [ColumnType(VarType.LAG)]
        lag = np.reshape(lag, (-1, 1))
        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))
        X = np.concatenate([X, lag], axis=1)

        return attr.evolve(
            data, X=X, y=y, column_names=column_names, column_types=column_types
        )
