from NN.NeuralNetNumpy import NeuralNet
from NN.ActivationFunction import ReLU
import numpy as np
import matplotlib.pyplot as plt

dogs = np.load('doodle/data/dogs200.npy')
airplanes = np.load('doodle/data/airplanes200.npy')
guitars = np.load('doodle/data/guitars200.npy')

train = np.vstack((dogs, airplanes, guitars)) / 255
test = np.load('doodle/data/test.npy') / 255

labels = {
    'dog': [1, 0, 0],
    'airplane': [0, 1, 0],
    'guitar': [0, 0, 1]
}


def get_label(index, num=2000):
    if index < num:
        return labels['dog']
    elif index < 2 * num:
        return labels['airplane']
    elif index < 3 * num:
        return labels['guitar']


def get_doodle(val):
    if val == 0:
        return 'dog'
    if val == 1:
        return 'airplane'
    if val == 2:
        return 'guitar'


nn = NeuralNet([train.shape[1], 64, 16, 3])
nn.learning_rate = 0.2

for i in range(train.shape[0]):
    sample_index = np.random.choice(train.shape[0], replace=True)
    sample_inputs = train[sample_index]
    sample_labels = get_label(sample_index)
    nn.train(sample_inputs, sample_labels)


index = np.random.choice(test.shape[0])
a = nn.predict(test[index])
get_doodle(a.argmax())
plt.imshow(test[index].reshape(28, 28))
