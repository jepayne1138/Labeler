import keras
import math
import numpy as np

IN_SIZE = 150
OUT_SIZE = 35
SAMPLES = 1000
SEED = 0

np.random.seed(SEED)


def main():
    in_train = np.random.rand(SAMPLES, IN_SIZE)
    out_train = np.zeros((SAMPLES, OUT_SIZE))
    # Set true output labels
    for row in out_train:
        col = np.random.randint(OUT_SIZE)
        row[col] += 1

    network = create_netword(in_train, out_train, nb_epoch=5, validation_split=0.1)


def create_netword(in_train, out_train, **kwargs):
    model = create_model(len(in_train[0]), len(out_train[0]), **kwargs)
    # Train the model
    model.fit(in_train, out_train, **kwargs)


def create_model(
        input_size, output_size, hidden_size=None,
        activation='relu', dropout=0.5,
        learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False,
        loss='categorical_crossentropy', **extra):
    if hidden_size is None:
        hidden_size = geometric_mean(input_size, output_size)
    model = keras.models.Sequential()
    # Input layer
    model.add(
        keras.layers.Dense(
            hidden_size, input_dim=input_size, activation=activation
        )
    )
    model.add(keras.layers.Dropout(dropout))
    # Hidden layer
    model.add(keras.layers.Dense(output_size, activation='softmax'))

    # Optimizer
    sgd = keras.optimizers.SGD(
        lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov
    )

    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
    return model


def save_model(model, filename, **extra):
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


def geometric_mean(x, y):
    """Geometric mean of two values as nearest int"""
    return int(math.sqrt(x * y))


if __name__ == '__main__':
    main()
