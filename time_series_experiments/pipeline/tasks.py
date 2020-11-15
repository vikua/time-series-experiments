import abc

import attr
from sklearn.preprocessing import OrdinalEncoder
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
