import numpy as np


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


def train_test_split_index(size, fdw_steps, fw_steps, test_size, random_seed):
    idx = np.arange(size)
    idx = sliding_window(idx, fdw_steps + fw_steps)

    train_size = idx.shape[0] - int(idx.shape[0] * test_size)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    random_state = np.random.RandomState(random_seed)
    shuffle_idx = random_state.permutation(train_idx.shape[0])
    train_idx = train_idx[shuffle_idx]

    x_train_idx = train_idx[:, :fdw_steps]
    y_train_idx = train_idx[:, fdw_steps:]
    x_test_idx = test_idx[:, :fdw_steps]
    y_test_idx = test_idx[:, fdw_steps:]

    return x_train_idx, y_train_idx, x_test_idx, y_test_idx
