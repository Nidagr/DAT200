# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:05:23 2018

@author: oliver

Lecture 4 the first exercise, keep name as perceptron.py because
it is called from another file.
"""

import numpy as np


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        # the random weights initialized, loc=0 sets them around 0.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # for each row in X and each y
                update = self.eta * (target - self.predict(xi))
                # Here we do the weight updates
                # wj = wj + eta*(y_i-yhat_i)*x_i for j>0
                self.w_[1:] += update * xi
                # w0 = w0 + eta*(y_i-yhat_i)   (x0=1)
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # The net input is computed in this function as 
    # dot product of all X values and weights, x_i, w_i, then
    # added together with w0 (x0=1)
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    
    # Here we do the threshold function/predictions. 
    # if net input funtion of x is greater than 0, return 1,
    # otherwise -1. Return array of labels.
    
    # the dimension of the prediction is mx1
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


"""
The weights are updated in every iteration, n_iter times.

Net input is computed n_iter times, because every time weight 
updates we call predict() which calls net_input().

If the errors is not zero we add them to a list of errors. The
errors are eta*(y_i-yhat_i).
"""


