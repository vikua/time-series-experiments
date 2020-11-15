from typing import List

import attr

from .tasks import Task
from .data import TaskData, take_columns, combine
from .dataset import VarType


@attr.s
class Step(object):
    name: str = attr.ib()
    task: Task = attr.ib()
    features: List[str] = attr.ib(default=None)
    types: List[VarType] = attr.ib(default=None)

    @types.validator
    def types_validator(self, attr, value):
        if value is not None and self.features is not None:
            raise ValueError("features and types cannot be used together")


class Pipeline(Task):
    def __init__(self, steps: List[Step]):
        self._steps = steps

    def fit(self, data: TaskData):
        output = data
        for step in self._steps:
            output = step.task.fit_transform(output)
        return self

    def transform(self, data: TaskData) -> TaskData:
        output = data
        for step in self._steps:
            output = step.task.transform(output)
        return output

    def fit_transform(self, data: TaskData) -> TaskData:
        return self.fit(data).transform(data)


class ColumnsProcessor(Task):
    def __init__(self, branches: List[Step]):
        self._branches = branches

    def fit(self, data: TaskData):
        for branch in self._branches:
            branch_data = take_columns(data, branch.features, branch.types)
            branch.task.fit(branch_data)
        return self

    def transform(self, data: TaskData) -> TaskData:
        all_outputs = []
        for branch in self._branches:
            branch_data = take_columns(data, branch.features, branch.types)
            output = branch.task.transform(branch_data)
            all_outputs.append(output)

        return combine(all_outputs)
