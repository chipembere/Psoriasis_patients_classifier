#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:34:11 2019

@author: brianmusonza

Hyper-parameter search
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from itertools import product,chain
from sklearn.model_selection import train_test_split

import joblib
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve, f1_score
from scipy import interp
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold
from itertools import product,chain
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold,cross_val_score,\
ParameterGrid
from sklearn.preprocessing import normalize
import pprint
import joblib
from sklearn.metrics import make_scorer,roc_auc_score

from skopt import BayesSearchCV
from skopt.utils import use_named_args

# load more libraries
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

import skopt
from skopt.space import Integer, Real
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper

notebookstart= time.time()
# read in the train and test data from csv files

Random_state = np.random.RandomState(0)
dataframe = pd.read_excel(r"/Users/brianmusonza/Documents/venv/Data/data_for_classification_edited.xlsx")
df2 = dataframe.replace(r'\s+', np.NaN)

data_= df2.drop(['PASI.END.WEEK.7',\
    'PASI.END.WEEK.8','PASI.END.WEEK.9','PASI.END.WEEK.10','PASI.END.WEEK.11'], axis = 1)

ds = data_#.iloc[:96]
    
# # Drop missing data
# print(ds.shape)
# ds = ds.dropna()
# print(ds.shape)

# split data into X and y
X6 = data_.iloc[:,0:-1]  #W0-2
# X6 = normalize(X6)
y = data_.iloc[:,-1]

# Reduce the number of classes to 2, classes 2 and 1 are 
# merged together to form a new class (0) and class 3 
# becomes class 1.

y = np.array([data_.iloc[:,-1]])
b = Binarizer(2)
b_scaled = b.fit_transform(y)[0]
y = b_scaled


# Form a dataframe with the imputed x_values and binarized y_values.
d_x = pd.DataFrame(X6)#, columns=['Week_0', 'Week_1', 'Week_2','Week_3','Week_4','Week_5', 'Week_6'])
d_x['target'] = b_scaled


df1 = d_x # Complete set binarized

ds = df1.dropna()
#ds = ds.iloc[:99]

X = ds.iloc[:,1:-4]

X = normalize(X)

y = ds.iloc[:,-1]

# Validation set
ds2 = df1.iloc[100:]
ds2 = ds2.dropna()
X_2 = ds2.iloc[:,1:-1]
# Normalize 
X_2 = normalize(X_2)

y_2 = ds2.iloc[:,-1]

test_set = X_2####
t_set = pd.DataFrame(test_set)###
test_label = y_2###
t_set['target'] = test_label###


# Reporting util for different optimizers
def report_perf(optimizer, X, y, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time.time()
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           +u"\u00B1"+" %.3f") % (time.time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params

# Converting roc-auc score into a scorer suitable for model selection
roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

# Setting a 5-fold stratified cross-validation (note: shuffle=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=57)

# Initializing a CatBoostClassifier
clf = CatBoostClassifier(learning_rate=0.9,
                                depth=3,#3
                                iterations=59,#500#59
                                logging_level='Silent',
                                eval_metric='AUC',
                                border_count=10,
                                random_seed=57,
                                thread_count=4,
                                l2_leaf_reg=1,#10
                                fold_len_multiplier = 31.1)


# Defining your search space
#
search_spaces = {'iterations': Integer(30, 59, 'identity'),
                 'random_strength':Real(0.1,1, 'uniform'),
                 'l2_leaf_reg':Real(0.1, 11, 'uniform'),
                'colsample_bylevel': Real(0.1, 1.0, 'uniform'),
                 'depth': Integer(1, 8, 'normalize'),
                 'logging_level':['Silent'],
                 'eval_metric':['AUC'],
                 'loss_function':['Logloss'],
                 'boosting_type':['Ordered'],
                 'random_seed':Integer(1, 3000),
                 'learning_rate': Real(0.05, 1.0, 'uniform'),
                 'border_count': Integer(1, 25),
                 'fold_len_multiplier': Real(1.1, 62.1, prior='uniform')}

#search_spaces = {'iterations': Integer(30, 59, 'identity'),
#                 'l2_leaf_reg':Real(0.1, 30, 'uniform'),
#                'colsample_bylevel': Real(0.1, 1.0, 'uniform'),
#                 'depth': Integer(1, 16, 'normalize'),
#                 'logging_level':['Silent'],
#                 'eval_metric':['Accuracy'],
#                 'loss_function':['Logloss'],
#                 'boosting_type':['Ordered'],
#                 'learning_rate': Real(0.05, 1.0, 'uniform'),
#                 'border_count': Integer(1, 25),
#                 'fold_len_multiplier': Real(1.1, 1.16, prior='uniform')}

# Setting up BayesSearchCV
opt = BayesSearchCV(clf,
                    search_spaces,
                    scoring=roc_auc,
                    cv=skf,
                    n_iter=5,
                    n_points=100,
                    n_jobs=1,  # use just 1 job with CatBoost in order to avoid segmentation fault
                    return_train_score=False,
                    refit=True,
                    optimizer_kwargs={'base_estimator': 'ET'},#'GP', 'RF', 'ET'
                    random_state=57)

# Running the optimization
best_params = report_perf(opt, X, y,'CatBoost', 
                          callbacks=[VerboseCallback(20), 
                                     DeadlineStopper(60*30)])

print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
