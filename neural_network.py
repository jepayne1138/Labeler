import keras
import math
import pickle
import numpy as np

IN_SIZE = 150
OUT_SIZE = 3
SAMPLES = 10000
SEED = 0

NETWORK_NAME = 'trained_network'

np.random.seed(SEED)


def main():
    with open('training_input.npy', 'rb') as train_input:
        in_train = np.load(train_input)
    with open('training_output.npy', 'rb') as train_output:
        out_train = np.load(train_output)

    network_arguments = {
        'epoch': 1000,
        'momentum': 0.8,
        'validation_split': 0.1
    }
    network = create_network(in_train, out_train, **network_arguments)

    save_model(network, NETWORK_NAME, network_arguments)


def create_network(in_train, out_train, **kwargs):
    model = create_model(len(in_train[0]), len(out_train[0]), **kwargs)
    # Train the model
    train_model(model, in_train, out_train, **kwargs)
    return model


def create_model(
        input_size, output_size, hidden_size=None,
        activation='relu', dropout=0.5, **compile_args):
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
    compile_model(model, **compile_args)
    return model


def compile_model(
        model, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False,
        loss='categorical_crossentropy', **extra):
    # Optimizer
    sgd = keras.optimizers.SGD(
        lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov
    )

    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])


def train_model(
        model, in_train, out_train,
        batch_size=32, epoch=10, verbose=1, validation_split=0.0, **extra):
    model.fit(
        in_train, out_train,
        batch_size=batch_size, nb_epoch=epoch, verbose=verbose,
        validation_split=validation_split
    )


def save_model(model, filename, arguments):
    """Saves a model using the given filename as the base name

    Creates two files:
      1. Architecture file (JSON)
      2. Weights file (HDF5)
    """
    with open('{}.json'.format(filename), 'w') as arch_file:
        arch_file.write(model.to_json())
    model.save_weights('{}.h5'.format(filename))
    # Save arguments
    with open('{}.p'.format(filename), 'wb') as args_file:
        pickle.dump(arguments, args_file)


def load_model(filename):
    """Loads a model from architecture and weight files with filename"""
    with open('{}.json'.format(filename), 'r') as arch_file:
        model = keras.models.model_from_json(arch_file.read())
    model.load_weights('{}.h5'.format(filename))
    with open('{}.p'.format(filename), 'rb') as args_file:
        network_args = pickle.load(args_file)
    compile_model(model, **network_args)
    return model


def geometric_mean(x, y):
    """Geometric mean of two values as nearest int"""
    return int(math.sqrt(x * y))


if __name__ == '__main__':
    main()
