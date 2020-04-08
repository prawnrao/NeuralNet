import sys
from random import random, choice
import matplotlib.pyplot as plt
sys.path.insert(1, '/Users/pranavrao/Documents/playground/NN')
from NeuralNet import NeuralNet, bools, ReLU, sigmoid

def xor_logic(inputs):
    return [1] if inputs[0] + inputs[1] == 1 else [0]

### Batch Test Example (sigmoid) ###
nn = NeuralNet(2,3,1)
nn.learning_rate=0.2
size = 100
num = 100
with open( 'batch_sigmoid.txt', 'w') as batch:
    batch.write('Hyperparameters: \n')
    batch.write("{}\n".format(nn))
    batch.write("\tBatch Size: {}\n".format(size))
    batch.write("\t# of Batches: {}\n\n".format(num))
    for i in range(num):
        batch.write('Batch No. {}\n'.format(i+1))
        nn.batch_train(bools, batch_size=size)
        for b in bools:
            batch.write("{}\t".format(b))
            batch.write("{}\n".format(nn.predict(b)))
        batch.write('\n\n')
batch.close()

### Batch Test Example (ReLU) ###
nn = NeuralNet(2,3,1)
nn.learning_rate=0.2
nn.activation_function=ReLU
size = 100
num = 100
with open( 'batch_ReLU.txt', 'w') as batch:
    batch.write('Hyperparameters: \n')
    batch.write("{}\n".format(nn))
    batch.write("\tBatch Size: {}\n".format(size))
    batch.write("\t# of Batches: {}\n\n".format(num))
    for i in range(num):
        batch.write('Batch No. {}\n'.format(i+1))
        nn.batch_train(bools, batch_size=size)
        for b in bools:
            batch.write("{}\t".format(b))
            batch.write("{}\n".format(nn.predict(b)))
        batch.write('\n\n')
batch.close()

### Stocastic Test Example (ReLU) ###
nn = NeuralNet(2,3,1)
nn.learning_rate=0.2
nn.activation_function=ReLU
size = 10000
with open( 'stocastic_ReLU.txt', 'w') as stocastic:
    stocastic.write('Hyperparameters: \n')
    stocastic.write("{}\n".format(nn))
    stocastic.write("\tSize: {}\n\n".format(size))
    for i in range(size):
        inputs = choice(bools)
        labels = xor_logic(inputs)
        nn.train(inputs, labels)
    for b in bools:
        stocastic.write("{}\t".format(b))
        stocastic.write("{}\n".format(nn.predict(b)))
stocastic.close()

### Stocastic Test Example (sigmoid) ###
nn = NeuralNet(2,3,1)
nn.learning_rate=0.2
size = 10000
with open( 'stocastic_sigmoid.txt', 'w') as stocastic:
    stocastic.write('Hyperparameters: \n')
    stocastic.write("{}\n".format(nn))
    stocastic.write("\tSize: {}\n\n".format(size))
    for i in range(size):
        inputs = choice(bools)
        labels = xor_logic(inputs)
        nn.train(inputs, labels)
    for b in bools:
        stocastic.write("{}\t".format(b))
        stocastic.write("{}\n".format(nn.predict(b)))
stocastic.close()


x1 = [random() for _ in range(3000)]
x2 = [random() for _ in range(3000)]
Y = [nn.predict([x1[i], x2[i]]) for i in range(len(x1))]
plt.figure(figsize=[12,10])
plt.scatter(x1, x2, c=Y)
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title
plt.savefig( 'prediction.png')
