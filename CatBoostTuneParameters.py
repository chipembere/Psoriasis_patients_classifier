#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:34:11 2019

@author: brianmusonza

Hyper-parameter search
"""

import time 
notebookstart= time.time()
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import joblib
import catboost.utils as cbu
from sklearn.preprocessing import Binarizer
from sklearn.feature_selection import RFECV
from itertools import product, chain
from tqdm import tqdm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
#import seaborn as sns
import catboost as cb
from sklearn.metrics import roc_curve, auc, roc_curve, f1_score
from scipy import interp
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold
from itertools import product,chain
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold,cross_val_score,\
ParameterGrid


# load more libraries
import hyperopt
import sys
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

Random_state = np.random.RandomState(0)
dataframe = pd.read_excel(r"/Users/brianmusonza/Documents/venv/Data/data_for_classification.xlsx")
df2 = dataframe.replace(r'\s+', np.NaN)

data_= df2.drop(['PASI.END.WEEK.7',\
    'PASI.END.WEEK.8','PASI.END.WEEK.9','PASI.END.WEEK.10','PASI.END.WEEK.11'], axis = 1)


ds = data_#.iloc[:99]
    
# # Drop missing data
# print(ds.shape)
# ds = ds.dropna()
# print(ds.shape)

# split data into X and y
X6 = ds.iloc[:,0:-4] # W0-2
y = ds.iloc[:,-1]

# Reduce the number of classes to 2, classes 2 and 1 are 
# merged together to form a new class (0) and class 3 
# becomes class 1.

y = np.array([ds.iloc[:,-1]])
b = Binarizer(2)
b_scaled = b.fit_transform(y)[0]
y = b_scaled

# Form a dataframe with the imputed x_values and binarized y_values.
d_x = pd.DataFrame(X6)#, columns=['Week_0', 'Week_1', 'Week_2','Week_3','Week_4','Week_5', 'Week_6'])
d_x['Class'] = b_scaled

df6 = d_x
df6 = df6.dropna()
X = df6.iloc[:,1:-1].values
# Pre_processing
X = normalize(X)
y = df6.iloc[:,-1].values

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
for train_index, test_index in rskf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
# model = CatBoostClassifier(learning_rate=0.17855555555555558,
#                                 depth=3,
#                                 iterations=100,
#                                 logging_level='Silent',
#                                 eval_metric='AUC',
#                                 border_count=10,
#                                 thread_count=4,
#                                 l2_leaf_reg=3,
#                                 fold_len_multiplier = 1.16,
#                                 loss_function = 'Logloss')#best so far

model = CatBoostClassifier(learning_rate=0.9,
                                depth=3,#3
                                iterations=59,#500#59
                                logging_level='Silent',
                                eval_metric='AUC',
                                border_count=10,
                                random_seed=57,
                                thread_count=4,
                                l2_leaf_reg=1,#10
                                fold_len_multiplier = 31.1)

class UciAdultClassifierObjective(object):
    def __init__(self, dataset, const_params, fold_count):
        self._dataset = dataset
        self._const_params = const_params.copy()
        self._fold_count = fold_count
        self._evaluated_count = 0
        
    def _to_catboost_params(self, hyper_params):
        return {
            'learning_rate': hyper_params['learning_rate'],
            'depth': hyper_params['depth'],
            'l2_leaf_reg': hyper_params['l2_leaf_reg'],
            'iterations': hyper_params['iterations'],
            'border_count': hyper_params['border_count']}
    
    # hyperopt optimizes an objective using `__call__` method (e.g. by doing 
    # `foo(hyper_params)`), so we provide one
    def __call__(self, hyper_params):
        # join hyper-parameters provided by hyperopt with hyper-parameters 
        # provided by the user
        params = self._to_catboost_params(hyper_params)
        params.update(self._const_params)
        
        print('evaluating params={}'.format(params), file=sys.stdout)
        sys.stdout.flush()
        
        # we use cross-validation for objective evaluation, to avoid overfitting
        scores = cb.cv(
            pool=self._dataset,
            params=params,
            fold_count=self._fold_count,
            partition_random_seed=57,
            verbose=False)
        
        # scores returns a dictionary with mean and std (per-fold) of metric 
        # value for each cv iteration, we choose minimal value of objective 
        # mean (though it will be better to choose minimal value among all folds)
        # because noise is additive
        max_mean_auc = np.max(scores['test-AUC-mean'])
        print('evaluated score={}'.format(max_mean_auc), file=sys.stdout)
        
        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count), file=sys.stdout)
        
        # negate because hyperopt minimizes the objective
        return {'loss': -max_mean_auc, 'status': hyperopt.STATUS_OK}

def find_best_hyper_params(dataset, const_params, max_evals=10):    
    # we are going to optimize these three parameters, though there are a lot more of them (see CatBoost docs)
    parameter_space = {
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.1, 1.0),
        'depth': hyperopt.hp.randint('depth', 16),
        'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 1, 16),
        'iterations':hyperopt.hp.uniform('iterations',50,60),
        'border_count':hyperopt.hp.randint('border_count', 255),
        'fold_len_multiplier':hyperopt.hp.randint('fold_len_multiplier', 61.1)}
    objective = UciAdultClassifierObjective(dataset=dataset, const_params=const_params, fold_count=6)
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=objective,
        space=parameter_space,
        algo=hyperopt.rand.suggest,
        max_evals=max_evals,
        rstate=np.random.RandomState(seed=42))
    return best

def train_best_model(X, y, const_params, max_evals=10, use_default=False):
    # convert pandas.DataFrame to catboost.Pool to avoid converting it on each 
    # iteration of hyper-parameters optimization
    dataset = cb.Pool(X, y)
    
    if use_default:
        # pretrained optimal parameters
        best = {'depth': 3, 'fold_len_multiplier': 41.1, 'iterations': 50, 'learning_rate': 0.1}
    else:
        best = find_best_hyper_params(dataset, const_params, max_evals=max_evals)
    
    # merge subset of hyper-parameters provided by hyperopt with hyper-parameters 
    # provided by the user
    hyper_params = best.copy()
    hyper_params.update(const_params)
    
    # drop `use_best_model` because we are going to use entire dataset for 
    # training of the final model
    hyper_params.pop('use_best_model', None)
    
    model = cb.CatBoostClassifier(**hyper_params)
    model.fit(dataset, verbose=False)
    
    return model, hyper_params

# make it True if your want to use GPU for training
have_gpu = False
# skip hyper-parameter optimization and just use provided optimal parameters
use_optimal_pretrained_params = False
# number of iterations of hyper-parameter search
hyperopt_iterations = 5

const_params = dict({
    'task_type': 'GPU' if have_gpu else 'CPU',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC', 
    'custom_metric': ['AUC'],
    'iterations': 59,
    'random_seed': 57})

model, params = train_best_model(
    X_train, y_train, 
    const_params, 
    max_evals=hyperopt_iterations, 
    use_default=use_optimal_pretrained_params)
print('best params are {}'.format(params), file=sys.stdout)


def calculate_score_on_dataset_and_show_graph(X, y, model):
    import sklearn.metrics
    import matplotlib.pylab as pl
    pl.style.use('ggplot')
    
    dataset = cb.Pool(X, y)
    fpr, tpr, _ = cbu.get_roc_curve(model, dataset, plot=True)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc

calculate_score_on_dataset_and_show_graph(X_test, y_test, model)

print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))