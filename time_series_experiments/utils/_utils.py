import numpy as np
from tensorflow import keras

from ..pipeline.dataset import sliding_window


def get_initializer(name, seed):
    if name in ["zero", "ones"]:
        return keras.initializers.get(name)
    else:
        return keras.initializers.get({"class_name": name, "config": {"seed": seed}})


def create_decoder_inputs(y, go_token=0):
    assert len(y.shape) == 2

    go_tokens = np.empty((y.shape[0], 1, 1))
    go_tokens.fill(go_token)

    decoder_inputs = y[:, :-1, np.newaxis]
    decoder_inputs = np.concatenate([go_tokens, decoder_inputs], axis=1)
    return decoder_inputs


def train_test_split_index(
    size, fdw_steps, fw_steps, test_size, random_seed, shuffle_train=True
):
    idx = np.arange(size)
    idx = sliding_window(idx, fdw_steps + fw_steps)

    train_size = idx.shape[0] - int(idx.shape[0] * test_size)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    if shuffle_train:
        random_state = np.random.RandomState(random_seed)
        shuffle_idx = random_state.permutation(train_idx.shape[0])
        train_idx = train_idx[shuffle_idx]

    x_train_idx = train_idx[:, :fdw_steps]
    y_train_idx = train_idx[:, fdw_steps:]
    x_test_idx = test_idx[:, :fdw_steps]
    y_test_idx = test_idx[:, fdw_steps:]

    return x_train_idx, y_train_idx, x_test_idx, y_test_idx
