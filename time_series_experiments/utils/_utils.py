from tensorflow import keras


def get_initializer(name, seed):
    if name in ["zero", "ones"]:
        return keras.initializers.get(name)
    else:
        return keras.initializers.get({"class_name": name, "config": {"seed": seed}})
