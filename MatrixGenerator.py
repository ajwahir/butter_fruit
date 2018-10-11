# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:37:39 2018

@author: abhij
"""
from keras.datasets import boston_housing
import pandas as pd
import numpy as np
from itertools import combinations
from itertools import permutations
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

def PermutationGenerator(features,arith):
    permu = []
    upper_limit  = min(5,len(features)+1)
    for i in range(2,upper_limit):
        sub_arith = arith
        # for k in range(i-2):
        #     for l in range(len(arith)):
        #         sub_sub_arith = arith
        #     sub_arith = sub_arith + 
        sub_arith = list(permutations(arith*(i-1),i-1))
        
        sub = list(permutations(features,i))
        
        permu.append(list(product(sub,sub_arith)))
    return permu   

def applyFeature(x_train,combination):
    s = len(x_train)
    z = np.zeros((s,1))
    col = combination[0]
    # print("col",combination)
    l = len(col)
    sign = combination[1]
    for k in range(s):
        
        # print("jo",x_train[k][col[0]])
        temp = x_train[k][col[0]]
        # print("OhMYGod")
        for j in range(l-1):
            # if(k == 0):
                # print(x_train[k][col[j+1]])
            if(sign[j] == '*'):
                temp = temp * x_train[k][col[j+1]]
            elif(sign[j] == '+'):
                temp = temp + x_train[k][col[j+1]]
        z[k][0] = temp    
    x_train = np.append(x_train, z, axis=1)
    return x_train

def applyFeatureDrop(x_train,permu):
    s = len(x_train)
    z = np.zeros((s,1))
    col = permu[0]
    # print("col",combination)
    l = len(col)
    sign = permu[1]
    for k in range(s):
        
        # print("jo",x_train[k][col[0]])
        temp = x_train[k][col[0]]
        # print("OhMYGod")
        for j in range(l-1):
            # if(k == 0):
                # print(x_train[k][col[j+1]])
            if(sign[j] == '*'):
                temp = temp * x_train[k][col[j+1]]
            elif(sign[j] == '+'):
                temp = temp + x_train[k][col[j+1]]
        z[k][0] = temp    
    x_train = np.append(x_train, z, axis=1)
    x_train = np.delete(x_train,[permu[0][0]])
    return x_train

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
features = list(range(len(x_train[0])))
arith = ['+','*']
# feat = ['A','B','C','D']
all_combinations = []
# all_combination = CombinationGenerator(features,arith)
all_combination = PermutationGenerator(features,arith)
for i in range(len(all_combination)):
    all_combin_set = set(all_combination[i])
    all_combinations.append(list(all_combin_set))

# for i in range(len(all_combination)):
#     print(len(all_combination[i]))

all_combination = []
for i in range(len(all_combinations)):
    all_combination += all_combinations[i]

#-------- Machine Learning --------#

from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

def LearnAndPredictTest(train,all_combination,x_test, y_test):
    print("Started Learning")
    K = 3
    final_scores = []
    for i in range(len(all_combination)):
        # print('all',all_combination)
        x_train = applyFeature(train,all_combination[i])
        test_x = applyFeature(x_test,all_combination[i])
        kf = KFold(n_splits=K, random_state = 3228, shuffle = True)
        mse_scores = []
        for train_index,test_index in kf.split(x_train):
            train_x,valid_x = x_train[train_index],x_train[test_index]
            train_y,valid_y = y_train[train_index],y_train[test_index]
            regr = linear_model.LinearRegression()
            regr.fit(train_x, train_y)
            pred = regr.predict(test_x)
            ms_error = mse(y_test,pred)
            mse_scores.append(ms_error) 
        mse_mean = sum(mse_scores)/len(mse_scores)
        final_scores.append(mse_mean)
        print("Final Score : ",mse_mean)
    return final_scores

def LearnAndPredict(train,all_combination):
    print("Started Learning")
    K = 3
    final_scores = []
    for i in range(len(all_combination)):
        # print('all',all_combination)
        x_train = applyFeature(train,all_combination[i])
        kf = KFold(n_splits=K, random_state = 3228, shuffle = True)
        mse_scores = []
        for train_index,test_index in kf.split(x_train):
            train_x,valid_x = x_train[train_index],x_train[test_index]
            train_y,valid_y = y_train[train_index],y_train[test_index]
            regr = linear_model.LinearRegression()
            regr.fit(train_x, train_y)
            pred = regr.predict(valid_x)
            ms_error = mse(valid_y,pred)
            mse_scores.append(ms_error) 
        mse_mean = sum(mse_scores)/len(mse_scores)
        final_scores.append(mse_mean)
        print("Final Score : ",mse_mean)
    return final_scores

# final_scores = LearnAndPredict(x_train,all_combination,x_test, y_test)
final_scores = LearnAndPredict(x_train,all_combination)
result_metric = pd.DataFrame({'error':final_scores,'feature':all_combination})
good_metric = result_metric.ix[result_metric['error'].idxmin()]

# kf = KFold(n_splits=3, random_state = 3228, shuffle = True)
# mse_scores = []
# for train_index,test_index in kf.split(x_train):
#     train_x,valid_x = x_train[train_index],x_train[test_index]
#     train_y,valid_y = y_train[train_index],y_train[test_index]
#     regr = linear_model.LinearRegression()
#     regr.fit(train_x, train_y)
#     pred = regr.predict(x_test)
#     ms_error = mse(y_test,pred)
#     mse_scores.append(ms_error) 
#     mse_mean = sum(mse_scores)/len(mse_scores)
#     print("Final Score : ",mse_mean)