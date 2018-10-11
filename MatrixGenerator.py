# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:37:39 2018

@author: abhij
"""

import pandas as pd
import numpy as np
from itertools import combinations

def CombinationGenerator(features,arith):
    combis = []
    upper_limit  = max(5,len(features)+1)
    for i in range(2,upper_limit):
        combis.append(combinations(features,i))
    return combis

full_train_data = pd.read_csv('train.csv')
#combis = CombinationGeneration()