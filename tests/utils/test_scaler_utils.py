import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from time_series_experiments.utils import NoOpScaler, ScalerWrapper
from ..conftest import RANDOM_SEED


@pytest.mark.parametrize(
    "scaler_cls, scaler_args",
    [
        (MinMaxScaler, {"feature_range": (0.01, 0.99)}),
        (StandardScaler, {}),
        (NoOpScaler, {}),
    ],
)
@pytest.mark.parametrize("log_transform", [True, False])
def test_scaler_wrapper_with_minmax(scaler_cls, scaler_args, log_transform):
    random_state = np.random.RandomState(RANDOM_SEED)
    data = random_state.randint(1, 1000, (100, 5)).astype("float")

    scaler = ScalerWrapper(scaler_cls(**scaler_args), log_transform=log_transform)
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    reference_scaler = scaler_cls(**scaler_args)
    reference_data = np.reshape(data, (-1, 1))
    if log_transform:
        reference_data = np.log(reference_data)
    reference_scaler.fit(reference_data)
    expected = reference_scaler.transform(reference_data)
    expected = np.reshape(expected, data.shape)
    assert np.array_equal(expected, scaled_data)

    unscaled_data = scaler.inverse_transform(scaled_data)
    assert np.all(np.isclose(data, unscaled_data))
