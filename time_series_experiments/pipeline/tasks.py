import abc

import attr
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator

from .data import TaskData


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
    def __init__(self, task: BaseEstimator):
        self._task = task

    def fit(self, data: TaskData) -> Task:
        self._task.fit(data.X, data.y)
        return self

    def transform(self, data: TaskData) -> TaskData:
        _X = self._task.transform(data.X)
        return attr.evolve(data, X=_X)


class OrdCat(Task):
    def __init__(self):
        self._enc = None

    def fit(self, data: TaskData) -> Task:
        self._enc = OrdinalEncoder()
        self._enc.fit(data.X)
        return self

    def transform(self, data: TaskData) -> TaskData:
        _X = self._enc.transform(data.X)
        cardinality = [len(x) for x in self._enc.categories_]
        return attr.evolve(data, X=_X, column_types=cardinality)


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
        column_types = [0 for _ in column_names]
        return attr.evolve(
            data, X=_X, column_names=column_names, column_types=column_types
        )


class DateFeatures(Task):
    COMPONENTS = [
        "year",
        "month",
        "week",
        "day_of_month",
        "day_of_week",
        "hour",
        "minute",
        "second",
    ]

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
            components = self.COMPONENTS

        unknown = set(components) - set(self.COMPONENTS)
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
        for component in self._components:
            func = self.EXTRACTORS[component]
            _X.append(func(X))
            column_names.extend(
                [self.PATTERN.format(c, component) for c in data.column_names]
            )

        _X = np.concatenate(_X, axis=1)
        column_types = [0 for _ in column_names]

        return attr.evolve(
            data, X=_X, column_names=column_names, column_types=column_types
        )
