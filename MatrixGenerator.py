# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:37:39 2018

@author: abhij
"""
from keras.datasets import boston_housing
import pandas as pd
import numpy as np
from itertools import combinations
from itertools import product

def CombinationGenerator(features,arith):
    combis = []
    upper_limit  = min(5,len(features)+1)
    for i in range(2,upper_limit):
        sub_arith = arith
        # for k in range(i-2):
        #     for l in range(len(arith)):
        #         sub_sub_arith = arith
        #     sub_arith = sub_arith + 
        sub_arith = list(combinations(arith*(i-1),i-1))
        
        sub = list(combinations(features,i))
        
        combis.append(list(product(sub,sub_arith)))
    # for o in range(len(sub_arith)):
    #         print(len(sub_arith[o]))
    return combis   

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
features = list(range(len(x_train[0])))
arith = ['+','*']
feat = ['A','B','C','D']

all_combination = CombinationGenerator(feat,arith)
for i in range(len(all_combination)):
    all_combin_set = set(all_combination[i])
    all_combination[i] = list(all_combin_set)

for i in range(len(all_combination)):
    print(len(all_combination[i]))

