# coding: utf-8

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# # implementing a perceptron learning algo in python

# # an object-oriented perceptiron API


class Perceptron(object):
    """ Perceptron classifier
    
    Parameters 
    ------------
    etea : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset
    random_state : int
        Random number generator seed for random weight
        initialization.

    Attributes
    ------------
    w_ " 1d-array
        Weights after fitting
    errors_ " list
        Number of misclassifications (updates) in each epoch
    
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1): # this is two features: eta, n_iter (passes over the training dataset)
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        -----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, when n_examples is the number of exammples and
        n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values.
        
        Returns
        -------
        self : object

        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # scale is standard deviation
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.across(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# training a perceptron model on the iris dataset

# reading in the iris data ! Please remember

# works verified
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', s)

df = pd.read_csv(s,
                 header=None,
                 encoding='utf-8')

df.tail()

# works verified

df = pd.read_csv('../iris.data', 
                header=None, encoding='utf-8')
df.tail()