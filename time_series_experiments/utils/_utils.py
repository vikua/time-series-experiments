import numpy as np
from tensorflow import keras


def rmse(y_pred, y_test):
    return np.sqrt(np.mean((y_pred - y_test) ** 2))


def get_initializer(name, seed):
    if name in ["zero", "ones"]:
        return keras.initializers.get(name)
    else:
        return keras.initializers.get({"class_name": name, "config": {"seed": seed}})
