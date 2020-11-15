import abc

import attr
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
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
    def __init__(self):
        self._enc = None

    def fit(self, data: TaskData) -> Task:
        self._enc = OrdinalEncoder()
        self._enc.fit(data.X)
        return self

    def transform(self, data: TaskData) -> TaskData:
        _X = self._enc.transform(data.X)
        column_types = [
            ColumnType(VarType.CAT, level=len(x)) for x in self._enc.categories_
        ]
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
