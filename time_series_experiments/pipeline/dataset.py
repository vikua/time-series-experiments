from typing import Dict
from enum import Enum

import numpy as np
import attr
import pandas as pd
import ciso8601


class VarType(Enum):
    NUM = 1
    CAT = 2
    TXT = 3
    DATE = 4


@attr.s
class DatasetConfig(object):
    path: str = attr.ib()
    date_col: str = attr.ib()
    target_col: str = attr.ib()
    series_id_col: str = attr.ib(
        validator=attr.validators.optional(attr.validators.instance_of(str))
    )
    feature_types: Dict[str, VarType] = attr.ib(attr.validators.instance_of(dict))

    @feature_types.validator
    def validate_feature_types(self, attribute, value):
        if not all([isinstance(x, VarType) for x in value.values()]):
            raise ValueError("feature_types values should be instances of VarType enum")


def read_dataset(config: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(config.path)
    df[config.date_col] = df[config.date_col].apply(
        lambda x: ciso8601.parse_datetime(x)
    )
    return df


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
