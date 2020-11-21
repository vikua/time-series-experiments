import pytest
import numpy as np

from time_series_experiments.pipeline.sequences import TimeSeriesSequence
from time_series_experiments.pipeline.data import TaskData, ColumnType, sliding_window

from ..conftest import RANDOM_SEED


@pytest.fixture
def data():
    random_state = np.random.RandomState(RANDOM_SEED)

    x = random_state.random((1000, 5))
    y = random_state.random((1000,))

    return TaskData(
        X=x,
        column_names=["x1", "x2", "x3", "x4", "x5"],
        column_types=[
            ColumnType(),
            ColumnType(),
            ColumnType(),
            ColumnType(),
            ColumnType(),
        ],
        y=y,
    )


@pytest.mark.parametrize(
    "fdw, fw, batch_size", [(28, 7, 32), (168, 24, 64), (10, 10, 16)]
)
def test_time_series_sequence(data, fdw, fw, batch_size):
    seq = TimeSeriesSequence(data, fdw=fdw, fw=fw, batch_size=batch_size)
    assert len(seq) == int(np.ceil((data.X.shape[0] - fdw - fw) / batch_size))

    idx = sliding_window(np.arange(data.X.shape[0]), fdw + fw)

    for i in range(len(seq)):
        start = i * batch_size
        end = start + batch_size

        batch = idx[start:end]
        x_batch = batch[:, :fdw]
        y_batch = batch[:, fdw:]

        assert np.all(np.isclose(seq[i][0], data.X[x_batch]))
        assert np.all(np.isclose(seq[i][1], data.y[y_batch]))
