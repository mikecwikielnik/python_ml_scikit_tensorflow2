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

