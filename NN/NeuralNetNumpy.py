import numpy as np
from ActivationFunction import sigmoid

class NeuralNet(object):
    """Neural Network Object"""

    def __init__(self, input_nodes:int, hidden_nodes:list, output_nodes:int,
                 activation_func=sigmoid, learning_rate=0.1):
        """Constuctor for the neural network"""
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.hh_weights = []
        self.hidden_bias = []

        self.ih_weights = np.random.rand(self.hidden_nodes[0], self.input_nodes)
        for i in range(self.hidden_nodes):
            self.hidden_bias[i] = np.random.rand(self.hidden_nodes, 1)
        self.ho_weights = np.random.rand(self.output_nodes, self.hidden_nodes[-1])
        self.output_bias = np.random.rand(self.output_nodes, 1)

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
        # Hidden nodes values
        hidden = np.matmul(self.ih_weights, inputs)
        hidden = hidden + self.hidden_bias
        hidden = self.activation_function.func(hidden)

        # Output nodes values
        output = np.matmul(self.ho_weights, hidden)
        output = output + self.output_bias
        output = self.activation_function.func(output)

        return hidden, output

    def errors(self, output, labels):
        """ Calcultes the errors for the output and hidden layers"""
        labels = np.array(labels)
        output_errors = labels - output
        hidden_errors = np.matmul(self.ho_weights.T, output_errors)

        return hidden_errors, output_errors

    def back_propogate(self, inputs, hidden, output, labels):
        """ Propogates and assigns the weights back from the outputs
            to the inputs
        """
        hidden_errors, output_errors = self.errors(output, labels)

        output_gradient = sigmoid.dfunc(output)
        output_gradient = self.learning_rate * output_errors * output_gradient
        delta_w_ho = np.matmul(output_gradient, hidden.T)

        hidden_gradient = self.activation_function.dfunc(hidden)
        hidden_gradient = self.learning_rate * hidden_errors * hidden_gradient
        delta_w_ih = np.matmul(hidden_gradient, inputs.T)

        return hidden_gradient, delta_w_ih, output_gradient, delta_w_ho


    def train(self, inputs, labels, stocastic=True):
        """ Trains the neural net, with the option of stocastic
            or batch training
        """
        #Converting inputs to matrix object
        inputs = np.vstack(inputs)
        hidden, output = self.feed_forward(inputs)
        if stocastic:
            hidden_gradient, delta_w_ih, output_gradient, delta_w_ho = \
            self.back_propogate(inputs, hidden, output, labels)
            self.output_bias = self.output_bias + output_gradient
            self.ho_weights = self.ho_weights + delta_w_ho
            self.hidden_bias = self.hidden_bias + hidden_gradient
            self.ih_weights = self.ih_weights + delta_w_ih
        else:
            return self.back_propogate(inputs, hidden, output, labels)

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
