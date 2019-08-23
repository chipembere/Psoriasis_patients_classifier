from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from hyperparameter_hunter import Environment, CVExperiment
from hyperparameter_hunter import RandomForestOptPro, Real, Integer, Categorical
import pandas as pd
import numpy as np
import time
import impyute as impy
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Binarizer
from sklearn.metrics import auc, roc_curve, f1_score
from sklearn.datasets import fetch_covtype
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

#
#import skopt
#from skopt.space import Integer, Real
#from skopt import gp_minimize
#from skopt.utils import use_named_args
#from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper

notebookstart= time.time()
# read in the train and test data from csv files

Random_state = np.random.RandomState(0)
dataframe = pd.read_excel(r"/Users/brianmusonza/Documents/venv/Data/data_for_classification_edited.xlsx")
df2 = dataframe.replace(r'\s+', np.NaN)

data_= df2.drop(['PASI.END.WEEK.7',\
    'PASI.END.WEEK.8','PASI.END.WEEK.9','PASI.END.WEEK.10','PASI.END.WEEK.11'], axis = 1)
#data_= data_.dropna()
ds = data_#.iloc[:99]
    
# # Drop missing data
# print(ds.shape)
# ds = ds.dropna()
# print(ds.shape)

# split data into X and y
X6 = data_.iloc[:,0:-1].values  #W0-2
X6 = impy.mice(X6)

X6 = normalize(X6)
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

ds = df1#.dropna()

env = Environment(
    train_dataset=ds,
    results_path="/Users/brianmusonza/Documents/venv/Data/HyperparameterHunterAssets/Hunter2",
    target_column="target",
    metrics=dict(f1=lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro")),
    cv_type="StratifiedKFold",
    cv_params=dict(n_splits=5, random_state=42),
)

# Now that HyperparameterHunter has an active `Environment`, we can do two things:
#################### 1. Perform Experiments ####################
experiment = CVExperiment(
    model_initializer=LGBMClassifier,
    model_init_params=dict(boosting_type="gbdt", 
                           num_leaves=31, 
                           max_depth=-1, 
                           subsample=0.5),
)

# And/or...
#################### 2. Hyperparameter Optimization ####################
optimizer = RandomForestOptPro(iterations=10, random_state=32)
optimizer.forge_experiment(
    model_initializer=LGBMClassifier,
    model_init_params=dict(
        boosting_type=Categorical(["gbdt", "dart"]),
        num_leaves=Integer(10, 40),
        max_depth=-1,
        subsample=Real(0.3, 0.7)
    ),
)
optimizer.go()

print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
