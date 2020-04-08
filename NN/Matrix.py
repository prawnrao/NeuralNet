from random import random

class Matrix(object):
    """ Matrix object which is used to compute math needed for the NeuralNet class."""
    def __init__(self, rows: int, cols: int):
        self.__rows = rows
        self.__cols = cols
        self.__shape = (rows, cols)
        
        self.data = [0] * rows
        for i in range(rows):
            self.data[i] = [0] * cols
    
    def __mul__(self, val):
        """ Scalar multiplication"""
        return self.apply(lambda x: x * val)
    
    def __rmul__(self, val):
        return self.__mul__(val)
    
    def __add__(self, val):
        """ Scalar addition"""
        return self.apply(lambda x: x + val)
    
    def __radd__(self, val):
        return self.__add__(val)
    
    def __pow__(self, val):
        """ Scalar power"""
        return self.apply(lambda x: x ** val)
    
    def __repr__(self):
        """ Return method"""
        string = "Matrix {}".format(self.shape)
        for i in range(self.rows):
            string += "\n {}".format(self.data[i])
        return string 
    
    def __str__(self):
        """ To string method"""
        string = ""
        
        for i in range(self.rows):
            string += "{} \n".format(self.data[i])
        return string 
    
    @property
    def rows(self):
        """ Number of rows"""
        return self.__rows
    
    @property
    def cols(self):
        """ Number of columns"""
        return self.__cols
    
    @property
    def shape(self):
        """ (rows, columns)"""
        return self.__shape
    
    @property
    def T(self):
        """ Transpose of Matrix"""
        m = Matrix(self.cols, self.rows)
        for i in range(m.rows):
            for j in range(m.cols):
                m.data[i][j] = self.data[j][i]
        return m
    
    @property
    def trace(self):
        """ Trace of Matrix"""
        Tr = 0
        for i in range(self.rows):
            Tr += self.data[i][i]
        return Tr
    
    @property
    def total(self):
        """ Sum of all elements"""
        tot = 0
        for i in range(self.rows):
            for j in range(self.cols):
                tot += self.data[i][j]
        return tot
    
    @property
    def negative(self):
        """ Negative of matrix"""
        return self * -1
    
    @property
    def __min(self):
        """ Minimum element of matrix"""
        return min(self.to_array())
    
    @property
    def __max(self):
        """ Maximum element of matrix"""
        return max(self.to_array())
    
    def add(self, b, inplace=False):
        """ Sum of 2 matrices, can return or perform inplace"""
        if not inplace:
            m = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    if b.cols == 1:
                        m.data[i][j] = self.data[i][j] + b.data[i][0]
                    else:
                        m.data[i][j] = self.data[i][j] + b.data[i][j]
            return m
        
        for i in range(self.rows):
            for j in range(self.cols):
                if b.cols == 1:
                    self.data[i][j] = self.data[i][j] + b.data[i][0]
                else:
                    self.data[i][j] = self.data[i][j] + b.data[i][j]
    
    def to_array(self):
        """ Matrix to array"""
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr

    def range_normalise(self, maximum=None, minimum=None):
        """ Elements are made to range between 0 -> 1
            To be used for positive numbers only.
        """
        if maximum is None:
            maximum = self.__max
            
        if minimum is None:
            minimum = self.__min
            
        return self.apply(lambda x: (x - minimum) / (maximum  - minimum))
        
    def apply(self, func, inplace=False):
        """ Applies a function to each element of the matrix, can return or perform inplace"""
        if not inplace:
            m = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    m.data[i][j] = func(self.data[i][j])
            return m
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] = func(self.data[i][j])
                
    @staticmethod
    def from_array(arr: list):
        """ Instantiates a 1d matrix from an array"""
        m = Matrix(len(arr), 1)
        for i in range(m.rows):
            m.data[i][0] = arr[i]
        return m
              
    @staticmethod
    def random_matrix(rows: int, cols: int):
        """ Matrix of random numbers between 0, 1"""
        m = Matrix(rows, cols)
        return m.apply(lambda x: round((random() * 2) - 1, 3))
    
    @staticmethod
    def matmul(a, b):
        """ Matrix multiplication of 2 matrices"""
        m = Matrix(a.rows, b.cols)
        for i in range(m.rows):
            for j in range(m.cols):
                ele = 0
                for k in range(a.cols):
                    ele += a.data[i][k] * b.data[k][j]
                m.data[i][j] = round(ele, 3)
        return m
        
    @staticmethod
    def I(size: int):
        """ Identity Matrix for a given size"""
        m = Matrix(size, size)
        for i in range(size):
            m.data[i][i] = 1
        return m
    
    @staticmethod
    def Hadamard(a, b):
        """ Calculates the hadamard product of two matrices"""
        m = Matrix(a.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                m.data[i][j] = a.data[i][j] * b.data[i][j]
        return m
    
    @staticmethod
    def scalar_product(a, b):
        """ Calculates the dot product of two matricies"""
        if not same_dimention(a, b):
            raise(Exception)
    
        tot = 0
        for i in range(a.rows):
            for j in range(a.cols):
                tot += (a.data[i][j] * b.data[i][j])
        return tot
           
    @staticmethod
    def same_dimention(a, b):
        """ Checks the dimensions of two matrices"""
        return True if a.shape == b.shape else False
