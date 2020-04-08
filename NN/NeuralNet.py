from Matrix import Matrix
import math
from random import choice

def xor_logic(inputs):
    return [1] if inputs[0] + inputs[1] == 1 else [0]

def and_logic(inputs):
    return [1] if inputs[0] + inputs[1] == 2 else [0]

def or_logic(inputs):
    return [1] if inputs[0] + inputs[1] != 0 else [0]

bools = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

def __sig(x):
    """f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + math.e**(-x))

def __dsig(x):
    """f'(x) = f(x) * (1 - f(x))"""
    return x * (1 - x)

def __relu(x):
    """f(x) = 0 for x < 0 \n\t     = x for x >= 0"""
    return max(0, x)

def __drelu(x):
    """f'(x) = 0 for x <= 0 \n\t      = 1 for x >= 0"""
    return 0 if x <= 0 else 1


class ActivationFunction(object):
    """Activation Function Object"""
    def __init__(self, func, dfunc, name=""):
        """
        Constructor for activation function object

        Inputs:
        -------
            func: executable function
            dfunc: derivitive of func
        """
        self.name = name
        self.func = func
        self.dfunc = dfunc

    def __repr__(self):
        return """{} \n\t{} \n\t{}\n""".format(self.name,
                                               self.func.__doc__,
                                               self.dfunc.__doc__)

sigmoid = ActivationFunction(__sig, __dsig, 'Sigmoid')

ReLU = ActivationFunction(__relu, __drelu, 'ReLU')

class NeuralNet(object):
    """Neural Network Object"""

    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 activation_func=sigmoid, learning_rate=0.1):
        """Constuctor for the neural network"""
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.ih_weights = Matrix.random_matrix(self.hidden_nodes,
                                               self.input_nodes)
        self.hidden_bias = Matrix(self.hidden_nodes, 1)
        self.ho_weights = Matrix.random_matrix(self.output_nodes,
                                               self.hidden_nodes)
        self.output_bias = Matrix(self.output_nodes, 1)

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
        hidden = Matrix.matmul(self.ih_weights, inputs)
        hidden.add(self.hidden_bias, inplace=True)
        hidden.apply(self.activation_function.func, inplace=True)

        # Output nodes values
        output = Matrix.matmul(self.ho_weights, hidden)
        output.add(self.output_bias, inplace=True)
        output.apply(sigmoid.func, inplace=True)

        return hidden, output

    def errors(self, output, labels):
        """ Calcultes the errors for the output and hidden layers"""
        labels = Matrix.from_array(labels)
        output_errors = labels.add(output.negative)
        hidden_errors = Matrix.matmul(self.ho_weights.T, output_errors)

        return hidden_errors, output_errors

    def back_propogate(self, inputs, hidden, output, labels):
        """ Propogates and assigns the weights back from the outputs
            to the inputs
        """
        hidden_errors, output_errors = self.errors(output, labels)

        output_gradient = output.apply(sigmoid.dfunc)
        output_gradient = self.learning_rate * Matrix.Hadamard(output_errors,
                                                               output_gradient)
        delta_w_ho = Matrix.matmul(output_gradient, hidden.T)

        hidden_gradient = hidden.apply(self.activation_function.dfunc)
        hidden_gradient = self.learning_rate * Matrix.Hadamard(hidden_errors,
                                                               hidden_gradient)
        delta_w_ih = Matrix.matmul(hidden_gradient, inputs.T)

        return hidden_gradient, delta_w_ih, output_gradient, delta_w_ho


    def train(self, inputs, labels, stocastic=True):
        """ Trains the neural net, with the option of stocastic
            or batch training
        """
        #Converting inputs to matrix object
        inputs = Matrix.from_array(inputs)
        hidden, output = self.feed_forward(inputs)
        if stocastic:
            hidden_gradient, delta_w_ih, output_gradient, delta_w_ho = \
            self.back_propogate(inputs, hidden, output, labels)
            self.output_bias.add(output_gradient, inplace=True)
            self.ho_weights.add(delta_w_ho, inplace=True)
            self.hidden_bias.add(hidden_gradient, inplace=True)
            self.ih_weights.add(delta_w_ih, inplace=True)
        else:
            return self.back_propogate(inputs, hidden, output, labels)

    def batch_train(self, inputs_array, label_function=xor_logic,
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
                hidden_gradient.add(result[0], inplace=True)
                delta_w_ih.add(result[1], inplace=True)
                output_gradient.add(result[2], inplace=True)
                delta_w_ho.add(result[3], inplace=True)

        self.output_bias.add(output_gradient, inplace=True)
        self.ih_weights.add(delta_w_ih, inplace=True)
        self.hidden_bias.add(hidden_gradient, inplace=True)
        self.ho_weights.add(delta_w_ho, inplace=True)


    def predict(self, inputs):
        """ Method to test the neural net for a given input"""
        inputs = Matrix.from_array(inputs)
        return self.feed_forward(inputs)[1].data[0][0]

    def __repr__(self):
        string = ""
        string += "\tInputs: {}".format(self.input_nodes)
        string +="\n\tHidden: {}".format(self.hidden_nodes)
        string += "\n\toutput: {}".format(self.output_nodes)
        string += "\n\tActivation: {}".format(self.__activation_function.name)
        string += "\n\tLearning Rate: {}".format(self.__learning_rate)

        return string
