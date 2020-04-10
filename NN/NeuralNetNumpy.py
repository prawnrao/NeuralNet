import numpy as np
from ActivationFunction import sigmoid

class NeuralNet(object):
    """Neural Network Object"""

    def __init__(self, shape:list, activation_func=sigmoid, learning_rate=0.1):
        """Constuctor for the neural network"""
        self.input_nodes = shape[0]
        self.output_nodes = shape[-1]
        self.hidden_nodes = shape[1:-1]

        self.biases = [np.random.rand(size, 1) for size in shape]
        del self.biases[0]
        self.weights = [np.random.rand(shape[i], shape[i-1]) for i in \
                        range(1:len(shape))]

        self.__activation_function = activation_func
        self.__learning_rate = learning_rate

    @property
    def activation_function(self):
        return self.__activation_function

    @activation_function.setter
    def activation_function(self, activation_func):
        self.__activation_function = activation_func

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, val):
        self.__learning_rate = val

    def feed_forward(self, inputs):
        """ Applying weights, bias and activation function to
            input and hiddens layer to get the output.

            H = activation(W_ih x I + bias_h)
            O = sigmoid(W_ho x H + bias_o)
        """
        forward = [inputs]
        for i, w in enumerate(self.weights):
            val = np.matmul(w, forward[i])
            val = val + self.biases[i]
            val = self.activation_function.func(val)
            forward[i+1] = val

        return forward

    def calculate_errors(self, output, labels):
        """ Calcultes the errors for the output and hidden layers"""
        labels = np.array(labels)
        output_errors = labels - forward[-1]
        errors = [output_errors]

        for i, w in enumerate(reversed(self.weights)):
            val = np.matmul(errors[i], w.T)

        return errors

    def back_propogate(self, forward, labels):
        """ Propogates and assigns the weights back from the outputs
            to the inputs
        """
        backward = reversed(forward)
        errors = self.calculate_errors(forward[-1], labels)
        gradients = []
        gradients_w = []

        for i, b in enumerate(backward):
            grad = self.activation_func.dfunc(b)
            grad = self.learning_rate * errors[i] * grad
            grad_w = np.matmul(grad, backward[i+1].T)

            gradients[i] = grad
            gradients_w[i] = grad_w

        return gradients, gradients_w


    def train(self, inputs, labels, stocastic=True):
        """ Trains the neural net, with the option of stocastic
            or batch training
        """
        inputs = np.vstack(inputs)
        forward = self.feed_forward(inputs)

        gradients, gradients_w = self.back_propogate(forward, labels)

        for i, grad in enumerate(gradients):
            self.biases[-i-1] + grad

        for i, grad in enumerate(gradients_w):
            self.weights[-i-1] + grad

    def batch_train(self, inputs_array, label_function,
                    batch_size=100):
        """ Method used to train the neural net in batches"""
        for i in range(batch_size):
            inputs = choice(inputs_array)
            label = label_function(inputs)
            result = self.train(inputs, label, stocastic=False)
            if i == 0:
                hidden_gradient = result[0]
                delta_w_ih = result[1]
                output_gradient = result[2]
                delta_w_ho = result[3]
            else:
                hidden_gradient = hidden_gradient + result[0]
                delta_w_ih = delta_w_ih + result[1]
                output_gradient = output_gradient+ result[2]
                delta_w_ho = delta_w_ho + result[3]

        self.output_bias = self.output_bias + output_gradient
        self.ho_weights = self.ho_weights + delta_w_ho
        self.hidden_bias = self.hidden_bias + hidden_gradient
        self.ih_weights = self.ih_weights + delta_w_ih


    def predict(self, inputs):
        """ Method to test the neural net for a given input"""
        inputs = np.vstack(inputs)
        return self.feed_forward(inputs)[1]

    def __repr__(self):
        string = ""
        string += "\tInputs: {}".format(self.input_nodes)
        string +="\n\tHidden: {}".format(self.hidden_nodes)
        string += "\n\toutput: {}".format(self.output_nodes)
        string += "\n\tActivation: {}".format(self.__activation_function.name)
        string += "\n\tLearning Rate: {}".format(self.__learning_rate)

        return string
