#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 17:52:13 2018

@author: blindqui
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor




def add_to_buffer(input_buffer, index, new_string):
    #print("buffer size  = ",len(input_buffer))
    #print("adding to buffer index ",index)
    str1 = input_buffer[index]
    l1 = list(str1)
    for i in range(len(l1)):
        if i<len(new_string) and new_string[i] != " ":
            l1[i] = new_string[i]
    
    input_buffer[index] = ''.join(l1)
    
#str1 = "he 1 w      "
#str2 = "ab "
#l1 = list(str1)
##print(str1|str2)
#l1[9:] = str2
#print(''.join(l1))



def calc_sumsq_err(vals):
    
    num_vals = len(vals)
    if num_vals == 0:
        return 0
    else:
        sum_val = 0
        sum_sq = 0 

        for value in vals:
            sum_val += value
            sum_sq += value*value
    
        # sum_i (x_i - mu)^2
        #  =  sum_i (x_i)^2 - 2*mu*sum_i x_i  + N*mu^2
        #  =  sum_sq   - 2*mu * sum_val + mu*sum_val
        # =  sum_sq - sum_val*sum_val/N
        # = N * [mean(x^2) - mean(x)*mean(x)]

        return sum_sq - sum_val*sum_val/num_vals
        
   
class TreeNode:
    
    def __init__(self, x_vals, y_vals, level=1):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.left = None
        self.right = None
        self.idx = -1
        self.cut_val = 0
        self.level = level
        
    def get_score(self, idx, cut):
        y_left = []
        y_right = []
        for i in range(len(self.x_vals)):
            if self.x_vals[i][idx]  < cut:
                y_left.append( self.y_vals[i] )
            else:
                y_right.append( self.y_vals[i] )
                
        #print("left size ",len(y_left)," right size ",len(y_right))
        sumsqerr_left = calc_sumsq_err(y_left)
        sumsqerr_right = calc_sumsq_err(y_right)
        
        return sumsqerr_left + sumsqerr_right
    
    def divide(self, idx, cut, min_size):
        y_left = []
        x_left = []
        y_right = []
        x_right = []
        for i in range(len(self.x_vals)):
            if self.x_vals[i][idx]  < cut:
                x_left.append( self.x_vals[i])
                y_left.append( self.y_vals[i] )
            else:
                x_right.append( self.x_vals[i])
                y_right.append( self.y_vals[i] )
        left_tree = None 
        right_tree = None
        
        if len(y_left) >= min_size and len(y_right)>=min_size:
                
            left_tree = TreeNode(x_left, y_left, level=self.level+1)
            right_tree = TreeNode(x_right, y_right,level=self.level+1)
            self.left = left_tree
            self.right = right_tree
            self.idx = idx
            self.cut_val = cut

        return left_tree, right_tree
        
        

class CartTree:
    
    def __init__(self, max_depth=4, min_events_split=8):
        self.__max_depth=max_depth
        self.__base_node = None
        self.__min_events_split = min_events_split

    def fit(self, x_data, y_data):
        
        self.__x_vals = x_data.copy()
        self.__y_vals = y_data.copy()
        
        print("fitting {} events".format(len(self.__x_vals)))

        node = self.create_base_node()

        self.split_node(node)

        
    def create_base_node(self):
        
        base_node = TreeNode(self.__x_vals, self.__y_vals, level=1)
        self.__base_node = base_node
        return base_node
        
        
    def split_node(self,base_node):
        #recursively split the nodes
        
        #compute left_node, right_node
        #this will involve looping through variables
        #and rows, and trying n_vars*n_rows different
        #splits, and then choosing the best of these splits
        
        
        #print("splitting ",type(base_node))
        
        n_rows = len(base_node.x_vals)
        #print("n_rows = {}".format(n_rows))

        if base_node.level<self.__max_depth and  n_rows>=self.__min_events_split:

            n_vars = len(base_node.x_vals[0])
            #print("n_vars = {}".format(n_vars))

            started=False
            best_score = 1.0e15
            best_idx = -1
            best_cut = 0

            for var_idx in range(n_vars):            
                for var_row in range(n_rows):

                    cut_val = base_node.x_vals[var_row][var_idx]
                    #print("Checking split for ",var_idx," < ",cut_val)
                    score = base_node.get_score(var_idx,cut_val)
                    #print("score = ",score)
                    if not(started) or score < best_score:
                        started=True
                        best_score = score
                        best_idx = var_idx
                        best_cut = cut_val

            #print("Best split value: var{} < {}; score = {}".format( best_idx, best_cut, best_score))
            min_size = 1
            left_node, right_node = base_node.divide(best_idx, best_cut, min_size)
            if left_node and right_node:
                self.split_node(left_node)
                self.split_node(right_node)
            else:
                #print("Rejecting this split.  Finished splitting this node.")
                pass
        else:
            #print("Done splitting")
            pass
            
    def get_pred(self,x_val):
        
        node = self.__base_node
        at_bottom=False
        while not at_bottom:
            
            if node.left:
                if x_val[ node.idx ] < node.cut_val:
                    node = node.left
                else:
                    node = node.right
            else:
                at_bottom=True

        #print("in get_pred", np.sum(node.y_vals))
        return np.sum(node.y_vals)/len(node.y_vals)
    
    def predict(self, x_vals):
        my_pred = []
        for x in x_vals:
            my_pred.append( self.get_pred(x) )
            
        #print(my_pred)
        return np.array(my_pred)
    
    def print(self): 
        start_level=1
        depth = self.__max_depth-start_level + 1
        buffer = [ " "*100 for i in range(depth)]
        self.print_node(self.__base_node,start_level=start_level,leaf_loc=0,buffer=buffer)
        for line in buffer:
            print(line)
        
    def print_node(self,node,start_level=1,leaf_loc=0,buffer=[]):
        #print("node lvl ",node.level, "idx ",node.idx, " < ",node.cut_val, "location = ",leaf_loc)
        
        #x_width = pow(2,self.__max_depth-start_level) 
        
        y_loc = self.__max_depth-node.level
        x_base = 0.5 + 0.5*pow(2,y_loc)  #1, 1.5, 2.5, 4.5
        x_loc = x_base + leaf_loc*pow(2,y_loc)
        #print("x, y = {},{}".format(x_loc,y_loc))
        node_string = "ix{}<{:.2f}".format(node.idx,node.cut_val)
        node_string = int(8*x_loc)*" " + node_string
        #print(node_string)
        add_to_buffer(buffer,node.level-start_level, node_string)
        
        if node.left and node.right:
            self.print_node(node.left, start_level, leaf_loc=2*leaf_loc, buffer=buffer)
            self.print_node(node.right, start_level, leaf_loc=2*leaf_loc+1, buffer=buffer)

#        0
#    0       1  
#  0   1   2   3
# 0 1 2 3 4 5 6 7    
        
        
        
#random dataset

nData = 1000
err_size = 0.3
rng = np.random.RandomState(1)

x1_max = 2
x2_max = 4
x1 = x1_max*rng.rand(nData,1)
x2 = x2_max*rng.rand(nData,1)
err = rng.normal(0,err_size, nData).reshape(nData,1)

#help(rng)

print(type(x1))
print(x1.mean())
print(x2.mean())
print(err.mean())

#y_val = 3.5 + 2*np.sin(3*x1 + 0.4) + err
y_val = 3.5 + 2*np.sin(3*x1 + 0.4) + err


#print(y_val.mean())
#print(len(x1))

#print(len(y_val))

#print(err)
#print(x1)
#print(y_val)

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

#plt.show()

        
    
my_tree = CartTree(max_depth=4, min_events_split=10)  #hmm. what I call max_depth=4 is the same as sklearn's max_depth=3
    
my_tree.fit(x1, y_val)


#print(type(y_pred))
#y_mypred = my_tree.get_pred(X_test[4])
#print("my predction for {} = {} ".format(X_test[4],y_mypred))
y_mypred = my_tree.predict(X_test)

#print(type(y_mypred))

my_tree.print()

#print(y_mypred)
#print(y_pred)

#y_mypred = my_tree.predict(X_test)
####plt.figure()

####plt.scatter(x1, y_val)
plt.plot(X_test, y_mypred, color='orange')

plt.show()