import itertools
from typing import List, Dict, Any

import attr
import numpy as np
import pandas as pd

from .dataset import DatasetConfig, VarType


@attr.s
class TaskMetadata(object):
    target_column = attr.ib(default=None)
    date_column = attr.ib(default=None)
    series_id_column = attr.ib(default=None)
    original_feature_types: Dict[str, Any] = attr.ib(default={})


@attr.s
class ColumnType(object):
    var_type: VarType = attr.ib(default=VarType.OBJ)
    level: int = attr.ib(default=0)


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
    column_types = [ColumnType(var_type=VarType.OBJ) for _ in column_names]

    if config:
        column_names = [c for c in column_names if c in config.feature_types]
        column_types = [
            ColumnType(config.feature_types.get(col, VarType.OBJ))
            for col in column_names
        ]
        X = X[column_names]

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


def take_columns(
    data: TaskData, columns: List[str] = None, types: List[VarType] = None
) -> TaskData:
    if not columns and not types:
        return data

    if columns and types:
        raise ValueError("columns and types cannot be used together")

    if columns:
        indexes = [data.column_names.index(c) for c in columns]
    elif types:
        types = set(types)
        indexes = [i for i, c in enumerate(data.column_types) if c.var_type in types]

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


def sliding_window(arr, window_size):
    """ Takes and arr and reshapes it into 2D matrix

    Parameters
    ----------
    arr: np.ndarray
        array to reshape
    window_size: int
        sliding window size

    Returns
    -------
    new_arr: np.ndarray
        2D matrix of shape (arr.shape[0] - window_size + 1, window_size)
    """
    (stride,) = arr.strides
    arr = np.lib.index_tricks.as_strided(
        arr,
        (arr.shape[0] - window_size + 1, window_size),
        strides=[stride, stride],
        writeable=False,
    )
    return arr
