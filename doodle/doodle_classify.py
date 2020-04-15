from NN.NeuralNetNumpy import NeuralNet
import numpy as np
import matplotlib.pyplot as plt

data_path = 'doodle/data/'
doodle = ['Dog', 'Airplane', 'Guitar']


def select_n_samples(file, start, size):
    arr = np.load(file)
    arr = arr[start:start+size]
    return np.array([a.flatten() for a in arr])


def select_random_samples(file, size):
    arr = np.load(file)
    arr = arr[np.random.choice(arr.shape[0], size, replace=False)]
    return np.array([a.flatten() for a in arr])


def get_training_data(size, i=0):
    dogs = select_n_samples(data_path+'dogs.npy', int(i*size), size)
    airplanes = select_n_samples(data_path+'airplanes.npy', int(i*size), size)
    guitars = select_n_samples(data_path+'guitars.npy', int(i*size), size)

    return np.vstack((dogs, airplanes, guitars)) / 255


def get_testing_data(size):
    dogs = select_random_samples(data_path+'dogs.npy', size)
    airplanes = select_random_samples(data_path+'airplanes.npy', size)
    guitars = select_random_samples(data_path+'guitars.npy', size)

    return np.vstack((dogs, airplanes, guitars)) / 255


def train_epoch(nn, train_size, index):
    train = get_training_data(train_size, index)
    for i in range(train.shape[0]):
        sample_index = np.random.choice(train.shape[0], replace=False)
        sample_inputs = train[sample_index]
        sample_labels = get_label(sample_index, train_size)
        nn.train(sample_inputs, sample_labels)


def batch_test(nn, test):
    correct = 0
    for i in range(test.shape[0]):
        sample_inputs = test[i]
        sample_labels = get_label(i, test.shape[0] / 3, position=True)
        output = nn.predict(sample_inputs)
        if output.argmax() == sample_labels:
            correct += 1

    return round(correct / test.shape[0] * 100, 4)


def main(train_size, test_size, epochs=1):
    nn = NeuralNet([784, 64, 3])
    nn.learning_rate = 0.2
    test_data = get_testing_data(test_size)
    for i in range(epochs):
        train_epoch(nn, train_size, i)
        print('{}% accuracy'.format(batch_test(nn, test_data)))
    print('Done')
    return nn


def get_label(index, size, position=False):
    if position:
        if index < size:
            return 0
        elif index < 2 * size:
            return 1
        elif index < 3 * size:
            return 2
    else:
        if index < size:
            return [1, 0, 0]
        elif index < 2 * size:
            return [0, 1, 0]
        elif index < 3 * size:
            return [0, 0, 1]


if __name__ == '__main__':
    nn = main(2000, 300, 10)
