import keras


def main():
    pass


def save_model(model, filename):
    """Saves a model using the given filename as the base name

    Creates two files:
      1. Architecture file (JSON)
      2. Weights file (HDF5)
    """
    with open('{}.json'.format(filename), 'w') as arch_file:
        arch_file.write(model.to_json())
    model.save_weights('{}.h5'.format(filename))


def load_model(filename):
    """Loads a model from architecture and weight files with filename"""
    with open('{}.json'.format(filename), 'r') as arch_file:
        model = keras.models.model_from_json(arch_file.read())
    model.load_weights('{}.h5'.format(filename))
    return model


if __name__ == '__main__':
    main()
