from NeuralNet import NeuralNet, sigmoid
import sys
from random import random, choice
import matplotlib.pyplot as plt
sys.path.insert(1, '/Users/pranavrao/Documents/playground/NN')

bools = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]

### Stocastic Test Example (sigmoid) ###
nn = NeuralNet(2, 3, 1)
nn.learning_rate = 0.2
nn.activation_function = sigmoid
size = 10000
with open('/Users/pranavrao/Documents/playground/xor/stocastic_sigmoid.txt', 'w') as stocastic:
    stocastic.write('Hyperparameters: \n')
    stocastic.write("{}\n".format(nn))
    stocastic.write("\tSize: {}\n\n".format(size))
    for i in range(size):
        inputs = choice(bools)
        labels = [inputs[0] ^ inputs[1]]
        nn.train(inputs, labels)
    for b in bools:
        stocastic.write("{}\t".format(b))
        stocastic.write("{}\n".format(nn.predict(b)))
stocastic.close()


x1 = [random() for _ in range(3000)]
x2 = [random() for _ in range(3000)]
Y = [nn.predict([x1[i], x2[i]]) for i in range(len(x1))]
plt.figure(figsize=[12, 10])
plt.scatter(x1, x2, c=Y)
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title
plt.savefig('/Users/pranavrao/Documents/playground/xor/prediction.png')
