# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:45:06 2019
@author: Mictofile
"""

import numpy as np

def gradient_descent(x, y):
    
    theta0_curr = 0    
    theta1_curr = 0
    
    n = len(x)
    epochs = 10000
    learning_rate = 0.01
    
    for i in range(epochs):
        
        y_predicted = theta0_curr + theta1_curr * x 
        loss = (1/n) * sum((y-y_predicted)**2)
        theta1_derivative = -(2/n) * sum(x * (y - y_predicted))
        theta0_derivative = -(2/n) * sum(y - y_predicted)  
        theta1_curr = theta1_curr - learning_rate * theta1_derivative
        theta0_curr = theta0_curr - learning_rate * theta0_derivative
        print ("theta1 {}, theta0 {}, loss {} iteration {}".format(theta1_curr,theta0_curr,loss, i))
    
X = np.array([1, 2, 3, 4, 5])
Y = np.array([5, 7, 9, 11, 13])

gradient_descent(X, Y)
