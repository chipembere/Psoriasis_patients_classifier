#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:51:33 2019

@author: brianmusonza

This script will carry out pre-processing,
train, save and test catboost based model on the 
data set with 96 patients.
The second part of this script will test trained models 
on the data set with 962 patients.
Bar charts showing model performances will be printed.
"""

import time
notebookstart= time.time()
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler, MinMaxScaler, RobustScaler, minmax_scale
from sklearn.preprocessing import PowerTransformer,power_transform,robust_scale, scale, maxabs_scale, quantile_transform
import impyute as impy
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn import metrics
from scipy import interp
from sklearn.preprocessing import normalize
import joblib
from sklearn.preprocessing import Binarizer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split, RandomizedSearchCV, RepeatedKFold
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


Random_state = np.random.RandomState(0)

# load the datasets
dataframe = pd.read_excel(r"/Users/brianmusonza/Documents/venv/Data/data_for_classification.xlsx")
# Define missing values as np.NaN
df2 = dataframe.replace(r'\s+', np.NaN)
# Drop Weeks 7 to 11
data_= df2.drop(['PASI.END.WEEK.7',\
    'PASI.END.WEEK.8','PASI.END.WEEK.9','PASI.END.WEEK.10','PASI.END.WEEK.11'], axis = 1)
# Split the dataset by rows
ds = data_.iloc[:96]

# Define inputdata and labels
X6 = ds.iloc[:,1:-1]
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

# Split the dataset into segments
X5 = df6.iloc[:,0:-2]
df5 = pd.DataFrame(X5)
df5['Class'] = b_scaled

X4 = df5.iloc[:,0:-2]
df4 = pd.DataFrame(X4)
df4['Class'] = b_scaled

X3 = df4.iloc[:,0:-2]
df3 = pd.DataFrame(X3)
df3['Class'] = b_scaled

X2 = df3.iloc[:,0:-2]
df2 = pd.DataFrame(X2)
df2['Class'] = b_scaled

X1 = df2.iloc[:,0:-2]
df1 = pd.DataFrame(X1)
df1['Class'] = b_scaled

# Each X represents a combination of datapoints
X_groups = [ df1, df2,df3,df4,df5,df6]
combinations = ['W0-1', 'W0-2', 'W0-3', 'W0-4', 'W0-5','W0-6']


# Logging for Visual Comparison
log_cols=["Datapoints", "AUC", "Accuracy", "A_PR", 'Inde_x']
log = pd.DataFrame(columns=log_cols) 

# Loop through all the datapoint combinations/segments.
for name,i in zip(combinations, X_groups):
    
    # Classifier 
    model = CatBoostClassifier(learning_rate=0.39,
                                depth=3,#4#3#12
                                iterations=60,#500#59
                                logging_level='Silent',
                                eval_metric='AUC',
                                boosting_type='Ordered',
                                border_count=7,#4,7,9
                                random_seed=43,
                                thread_count=4,
                                l2_leaf_reg=7,#1#10#3,15
                                fold_len_multiplier=1.6)#best
    
#    model = CatBoostClassifier(learning_rate=1,
#                                depth=4,#3
#                                random_strength=0.9,
#                                iterations=60,#500#59
#                                logging_level='Silent',
#                                eval_metric='AUC',
#                                boosting_type='Ordered',
#                                border_count=10,
#                                random_seed=43,
#                                thread_count=4,
#                                l2_leaf_reg=1,#10
#                                fold_len_multiplier=1.6)
    
    # Data
    df = i
    # Drop entities missing values.
    df = df.dropna()
    
    # input data.
    X = df.iloc[:,0:-1].values
        
    # Impute missing values.

#      X = impy.mice(X)[0]
    
    # Preprocessing
    X=normalize(X)
    
    # labels.
    y = df.iloc[:,-1].values
    
    # X length.
    n_m = len(X)

    # Model and segment names
    m_name = model.__class__.__name__
    zita = name
    # Lists to save scores from loop
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    accs = []
    aps = []
    i = 0
    # Set repeated stratified Kfold parameters
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    # Split the dataset into a varitey of unique training ans testing sets to minimise bias
    for train, test in rskf.split(X, y):
        
        model.fit(X[train], y[train],use_best_model=True,eval_set=(X[test], y[test]))
        probas_ = model.predict_proba(X[test])
        y_score = model.predict(X[test])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        # Compute accuracy, roc_auc and average_precision_recall
        aucs.append(roc_auc)
        acc = accuracy_score(y_score, y[test])
        accs.append(acc)
        average_pre = average_precision_score(y[test], probas_[:, 1])
        aps.append(average_pre)
        
        # Save trained model
        model.save_model('mods/%s_CatT-Cset-%0.2f_model.cbm'%(zita, i))
        
        i += 1
        
    # mean true_positives.
    mean_tpr = np.mean(tprs, axis=0)
    # mean accuracy.
    mean_acc = np.mean(accs)
    # mean average_precision_score.
    mean_aps = np.mean(aps)
    mean_tpr[-1] = 1.0
    # mean area under the curve.
    mean_auc = auc(mean_fpr, mean_tpr)
    # entering the data into a dataframe.
    log_entry = pd.DataFrame([[name, mean_auc*100, mean_acc*100, mean_aps*100, n_m]], columns=log_cols)
    log = log.append(log_entry)
    # Standard deviation score
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    

# Setting bar plot labels, titles and values.
textz = ('%0.2f KFold iterations\n No imputation' % (i))
log = log.set_index('Datapoints')
ax = log[['AUC', 'Accuracy', 'A_PR']].plot(kind='bar', width=0.8 ,figsize=(14,8), color=['dodgerblue', 'slategray', 'brown'], fontsize=13);
ax.set_alpha(0.8)
ax.set_title('%s Tuned Performance'%(m_name),fontsize=18)
ax.set_ylabel("%", fontsize=13);
ax.set_yticks(np.arange(0, 100, step=5))

l1=log.loc['W0-1','Inde_x'] 
l2=log.loc['W0-2','Inde_x']
l3=log.loc['W0-3','Inde_x']
l4=log.loc['W0-4','Inde_x'] 
l5=log.loc['W0-5','Inde_x'] 
l6=log.loc['W0-6','Inde_x'] 
ax.set_xlabel('Datapoints \n (Rows in brackets)', fontsize=14, labelpad=7)
ax.set_xticklabels(["W0-1 \n (%i)"%(l1), "W0-2\n (%i)"%(l2), 
                    "W0-3\n (%i)"%(l3), "W0-4\n (%i)"%(l4), 
                    "W0-5\n (%i)"%(l5),"W0-6\n (%i)"%(l6)], rotation=0, fontsize=12)
ax.text(0.67, 1.1, textz, fontsize=11, color='green',\
        verticalalignment='bottom', horizontalalignment='right',\
        transform=ax.transAxes)
# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.01, i.get_height()+.6, \
            str(round((i.get_height()), 1)), fontsize=13, color='black', rotation=0)
#plt.savefig('/Users/brianmusonza/Documents/venv/Projects/Time-Series-Predictor/fname.png')



# Split the dataset by rows
# Validation Dataset
ds = data_.iloc[96:] # Remove iloc to check for complete set

# Define inputdata and labels
X6 = ds.iloc[:,1:-1]
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

# Split the dataset into segments
X5 = df6.iloc[:,0:-2]
df5 = pd.DataFrame(X5)
df5['Class'] = b_scaled

X4 = df5.iloc[:,0:-2]
df4 = pd.DataFrame(X4)
df4['Class'] = b_scaled

X3 = df4.iloc[:,0:-2]
df3 = pd.DataFrame(X3)
df3['Class'] = b_scaled

X2 = df3.iloc[:,0:-2]
df2 = pd.DataFrame(X2)
df2['Class'] = b_scaled

X1 = df2.iloc[:,0:-2]
df1 = pd.DataFrame(X1)
df1['Class'] = b_scaled

# Each X represents a combination of datapoints
X_groups = [ df1, df2,df3,df4,df5,df6]
combinations = ['W0-1', 'W0-2', 'W0-3', 'W0-4', 'W0-5','W0-6']


# Logging for Visual Comparison
log_cols=["Datapoints", "AUC", "Accuracy", "A_PR", 'Inde_x']
log = pd.DataFrame(columns=log_cols) 

# Loop through all the datapoint combinations/segments.
for name,i in zip(combinations, X_groups):
    model = CatBoostClassifier()
    df = i
    # Drop entities missing values.
    df = df.dropna()
    
    i = 0
    
    # input data.
    X = df.iloc[:,0:-1].values
        
    # Impute missing values.
    #X = impy.mode(X)#[0]
    
    # Preprocessing
    X = normalize(X)
    
    # labels.
    y = df.iloc[:,-1].values
    # X length.
    n_m = len(X)
    # Model and segment names
    m_name = model.__class__.__name__
    zita = name
    # Load trained model
    # model = joblib.load('%smodel.pkl'%(zita))
    model = model.load_model('mods/%s_CatT-Cset-%0.2f_model.cbm'%(zita, i))
    # Lists tosave scores from loop
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    accs = []
    aps = []
    
    # Set repeated stratified Kfold parameters
    rskf = RepeatedStratifiedKFold(n_splits=5,n_repeats=2,  random_state=42)
    # Split the dataset into a varitey of unique training ans testing sets to minimise bias
    
    for train, test in rskf.split(X, y):
        
        probas_ = model.predict_proba(X[test])
        y_score = model.predict(X[test])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        # Compute accuracy, roc_auc and average_precision_recall
        aucs.append(roc_auc)
        acc = accuracy_score(y_score, y[test])
        accs.append(acc)
        average_pre = average_precision_score(y[test], probas_[:, 1])
        aps.append(average_pre)
    
        i += 1
        
    # mean true_positives.
    mean_tpr = np.mean(tprs, axis=0)
    # mean accuracy.
    mean_acc = np.mean(accs)
    # mean average_precision_score.
    mean_aps = np.mean(aps)
    mean_tpr[-1] = 1.0
    # mean area under the curve.
    mean_auc = auc(mean_fpr, mean_tpr)
    # entering the data into a dataframe.
    log_entry = pd.DataFrame([[name, mean_auc*100, mean_acc*100, mean_aps*100, n_m]], columns=log_cols)
    log = log.append(log_entry)
    # Standard deviation score
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
# Setting bar plot labels, titles and values.
textz = ('%0.2f 5-Fold iterations\n No imputation\n Dataset (B)' % (i))
log = log.set_index('Datapoints')
ax = log[['AUC', 'Accuracy', 'A_PR']].plot(kind='bar', width=0.8 ,figsize=(14,8), color=['orange', 'yellow', 'turquoise'], fontsize=13);
ax.set_alpha(0.8)
ax.set_title('%s Tuned and Trained Model Validation Performance'%(m_name),fontsize=16)
ax.set_ylabel("%", fontsize=13);
ax.set_yticks(np.arange(0, 100, step=5))
l1=log.loc['W0-1','Inde_x'] 
l2=log.loc['W0-2','Inde_x']
l3=log.loc['W0-3','Inde_x']
l4=log.loc['W0-4','Inde_x'] 
l5=log.loc['W0-5','Inde_x'] 
l6=log.loc['W0-6','Inde_x'] 
ax.set_xlabel('\n Datapoints \n (Number of entities)', fontsize=14, labelpad=7)
ax.set_xticklabels(["W0-1 \n (%i)"%(l1), "W0-2\n (%i)"%(l2), 
                    "W0-3\n (%i)"%(l3), "W0-4\n (%i)"%(l4), 
                    "W0-5\n (%i)"%(l5),"W0-6\n (%i)"%(l6)], rotation=0, fontsize=12)
ax.text(0.67, 1.1, textz, fontsize=11, color='green',\
        verticalalignment='bottom', horizontalalignment='right',\
        transform=ax.transAxes)
# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.01, i.get_height()+.6, \
            str(round((i.get_height()), 1)), fontsize=13, color='black', rotation=0)


plt.show()

#save figure
#plt.savefig('/Users/brianmusonza/Documents/venv/Projects/Time-Series-Predictor/fname3.png')

#print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


