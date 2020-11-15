import itertools
from typing import List, Dict, Any

import attr
import numpy as np
import pandas as pd

from .dataset import DatasetConfig


@attr.s
class TaskMetadata(object):
    target_column = attr.ib(default=None)
    date_column = attr.ib(default=None)
    series_id_column = attr.ib(default=None)
    original_feature_types: Dict[str, Any] = attr.ib(default={})


@attr.s
class TaskData(object):
    X: np.ndarray = attr.ib()
    column_names: List[str] = attr.ib()
    column_types: List[int] = attr.ib()
    y: np.ndarray = attr.ib(default=None)
    metadata: TaskMetadata = attr.ib(default=None)


def to_task_data(
    X: pd.DataFrame, y: np.ndarray = None, config: DatasetConfig = None
) -> TaskData:
    column_names = X.columns.tolist()
    column_types = [0 for _ in column_names]
    metadata = TaskMetadata(
        target_column=config.target_col if config else None,
        date_column=config.date_col if config else None,
        series_id_column=config.series_id_col if config else None,
        original_feature_types=config.feature_types if config else None,
    )
    return TaskData(
        X=X.values,
        y=y,
        column_names=column_names,
        column_types=column_types,
        metadata=metadata,
    )


def take_columns(data: TaskData, columns: List[str]) -> TaskData:
    if not columns:
        return data

    indexes = [data.column_names.index(c) for c in columns]

    column_names = [data.column_names[i] for i in indexes]
    column_types = [data.column_types[i] for i in indexes]
    X = data.X[:, indexes]
    return TaskData(
        X=X,
        column_names=column_names,
        column_types=column_types,
        y=data.y,
        metadata=data.metadata,
    )


def combine(data_list: List[TaskData]) -> TaskData:
    X = np.concatenate([d.X for d in data_list], axis=1)
    y = data_list[0].y
    metadata = data_list[0].metadata
    column_names = list(itertools.chain(*[d.column_names for d in data_list]))
    column_types = list(itertools.chain(*[d.column_types for d in data_list]))

    return TaskData(
        X=X,
        column_names=column_names,
        column_types=column_types,
        y=y,
        metadata=metadata,
    )
