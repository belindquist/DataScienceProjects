#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of different ML algorithms (both "home-made" and from public packages) on a common dataset. 
Created on Fri Apr 13 17:52:13 2018

@author: blindqui
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from DecisionTree import CartTree


if __name__ == "__main__":        
        
    #Generate random dataset
    
    nData = 10000
    err_size = 0.3
    rng = np.random.RandomState(1)
    
    x1_max = 2
    x2_max = 4
    x1 = x1_max*rng.rand(nData,1)
    x2 = x2_max*rng.rand(nData,1)
    err = rng.normal(0,err_size, nData).reshape(nData,1)
    
    
    print("Generating 1-dimensional data with {} samples".format(nData))
#    print(type(x1))
#    print(x1.mean())
#    print(x2.mean())
#    print(err.mean())
    
    y_val = 3.5 + 2*np.sin(3*x1 + 0.4) + err
    
    
    
    tree_1 = DecisionTreeRegressor(max_depth=3)    
    tree_1.fit(x1, y_val)

    #X_test = np.arange(0, 5, 0.1)[:,np.newaxis]
    X_test = np.arange(0, x1_max, 0.02)
    nlen = len(X_test)
    X_test = X_test.reshape(nlen,1)
    #print(type(X_test))
    #print(len(X_test))
    #print(X_test)
    
    y_pred = tree_1.predict(X_test)
    plt.figure()
    
    plt.scatter(x1, y_val)
    plt.plot(X_test, y_pred, color='red')
    
    
    
    my_tree = CartTree(max_depth=3, min_events_split=10)  
    
    
    my_tree.fit(x1, y_val)


    y_mypred = my_tree.predict(X_test)
        
    my_tree.print()
    
    
    ####plt.scatter(x1, y_val)
    plt.plot(X_test, y_mypred, color='orange')
    
    plt.show()
