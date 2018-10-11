# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:37:39 2018

@author: abhij
"""
# from keras.datasets import boston_housing
import pandas as pd
import numpy as np
from itertools import combinations
from itertools import product

def CombinationGenerator(features,arith):
    combis = []
    upper_limit  = min(5,len(features)+1)
    for i in range(2,upper_limit):
        sub_arith = arith;
        for k in range(i-2):
            sub_arith = product(sub_arith,arith)
        sub = list(combinations(features,i))
#        s = len(sub)
#        for c in range(s):
#            combis.append(product(sub[c],sub_arith))
        
        combis.append(list(product(sub,sub_arith)))
#        sub = list(combinations(features,i))
#        for k in range(i-1):
#            for ari in arith:
#                sub.append(ari)
#        combis.append(sub)
    
    return combis

full_train_data = pd.read_csv('train.csv')
features = list(full_train_data.columns.values)
feat = ['A','B','C']
arith= ['+','*']


print(len(CombinationGenerator(feat,arith)))