import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def compute_CV(X, y, model, k):
    kfold = KFold(n_splits=k)
    auc_scores = []
    i = 0
    
    for index in kfold.split(X = X, y = y):
        
        train_index, val_index = index[0], index[1]
        
        i += 1 
        X_train = X.iloc[train_index, :]
        y_train = y[train_index]
        X_val = X.iloc[val_index, :]
        y_val = y[val_index]
        
        model.fit(X_train, y_train)
        y_predict = model.predict_proba(X_val)
        y_predict_prob = y_predict[:, 1]
        
        auc_score = roc_auc_score(y_val, y_predict_prob)
        print(f'AUC Score of {i} Fold is : {auc_score}')
        auc_scores.append(auc_score)
    print('-----------------------------------------------')
    print(f'Average AUC Score of {k} Folds is : {np.mean(auc_scores)}')