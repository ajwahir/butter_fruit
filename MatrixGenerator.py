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
        print(sub_arith)
        s = len(arith)
        for k in range(i-2):
            p = len(sub_arith)
            for x in range(p):
                for y in range(s):
                    sub_arith.append(sub_arith[x] + arith[y]))
        # for k in range(i-2):
        #     for ari in arith:
        #         sub_arith.append(ari) 

        sub = list(combinations(features,i))
        print(list(sub_arith))
        combis.append(list(product(sub,sub_arith)))

    return combis   

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
features = list(range(len(x_train[0])))
arith = ['+','*']
feat = ['A','B','C','D']

all_combination = CombinationGenerator(feat,arith)

for i in range(len(all_combination)):
    print(len(all_combination[i]))

