from numpy import e as exp
from numpy import vectorize


def __sig(x):
    """f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + exp**(-x))


def __dsig(x):
    """f'(x) = f(x) * (1 - f(x))"""
    return x * (1 - x)


def __relu(x):
    """f(x) = 0 for x < 0 \n\t     = x for x >= 0"""
    return x if x > 0 else 0


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

ReLU = ActivationFunction(vectorize(__relu), vectorize(__drelu), 'ReLU')
