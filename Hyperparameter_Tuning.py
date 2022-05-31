# import library
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb



def optimize(params, param_names, x, y, model):
    
    # convert params to dictionary
    params = dict(zip(param_names, params))

    # initialize model with current parameters
    if model == 'xgb':
        clf = XGBClassifier()
    
    # initialize stratified k fold
    kfold = KFold(n_splits=5)
    i = 0
    
    # initialize auc scores list
    auc_scores = []
    
    #loop over all folds
    for index in kfold.split(X = x, y = y):
        train_index, test_index = index[0], index[1]
        
        x_train = x.iloc[train_index, :]
        y_train = y[train_index]

        x_test = x.iloc[test_index, :]
        y_test = y[test_index]
        
        #fit model
        clf.fit(x_train, y_train)
        
        y_pred = clf.predict_proba(x_test)
        y_pred_pos = y_pred[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_pos)
        print(f'Current parameters of fold number {i} -> {params}')
        print(f'AUC score of test {i} f {auc}')

        i += 1
        auc_scores.append(auc)
        
    return -1 * np.mean(auc_scores)