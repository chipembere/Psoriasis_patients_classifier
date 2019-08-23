# Psoriasis_patients_classifier

Code used in the "__A gradient boosting decision tree forecaster for predicting Psoriasis PASI trajectories__ " research project. The provided scripts help load data, pre-processing, training, testing and plots results.
This also includes scripts for Bayesian hyper-parameter search and a script to plot PASI trajectories.

# Dependancies
numpy
pandas
matplotlib
sklearn
catboost
lightgbm
impyute
pprint
scipy
skopt
tqdm
hyperopt

# Notes

The scripts require the path to the data set containing PASI trajectories.
A 'mods/' folder must be created to save trained models. The number of cross-validation iterations can be adjusted.

The lgbm-models-.py file will train, test and report on the performances of all three models based on the 96 patient data set, 62 patient data set and a merger of both respectively.
