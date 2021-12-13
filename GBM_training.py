from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import random, copy
from sklearn.metrics import roc_auc_score
from itertools import product

print("Loading data ...", end="")
X=pd.read_csv('X_training.csv',index_col=0)
Y=pd.read_csv('Y_training.csv',index_col=0)
X_validation=pd.read_csv('X_validation.csv',index_col=0)
Y_validation=pd.read_csv('Y_validation.csv',index_col=0)   
print("done")

params_dict = {'max_depth': [2, 3, 4, 5, 6, 7], 'min_child_weight': [3, 1, 0.3, 0.1], "max_delta_step": [1, 0.3, 0.1],\
                'subsample': [0.3, 0.4, 0.5, 0.6, 0.7], 'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7], 'gamma': [3, 10, 30, 100],
                'eta': [0.1, 0.3, 0.5, 0.7], 'reg_alpha': [0.03, 0.1, 0.3, 1]}
params_comb = list(product(*params_dict.values()))
random.shuffle(params_comb)

cvSplit = 5
skf = StratifiedKFold(n_splits=cvSplit)
best_auc = 0.5
n_iters_max = 10000
print("Random GridsearchCV ...")    
for x in params_comb[:n_iters_max]:
    kwargs = {'n_estimators':500, 'objective':'binary:logistic', 'scale_pos_weight':sum(Y['TRTMNT_SUCCESS_1'])/sum(Y['TRTMNT_SUCCESS_0']), 
            'n_jobs':1, 'tree_method':'exact', 'verbosity':0}
    params_temp = dict(zip(list(params_dict.keys()), list(x)))
    kwargs.update(params_temp)
    auc_ave = 0
    best_auc_test = 0.5
    Y_validation_score = np.zeros(X_validation.shape[0])
    Y_training_score = np.zeros(X.shape[0])
    model_list = []
    for train_index, test_index in skf.split(X, Y['TRTMNT_SUCCESS_0']):              
        scale_pos_weight = sum(Y.iloc[train_index]['TRTMNT_SUCCESS_1']) / sum(Y.iloc[train_index]['TRTMNT_SUCCESS_0'])
        model = XGBClassifier(**kwargs)
        model.fit(X.iloc[train_index], Y.iloc[train_index]['TRTMNT_SUCCESS_0'], early_stopping_rounds=10, eval_metric="auc", \
            eval_set=[(X.iloc[test_index], Y.iloc[test_index]['TRTMNT_SUCCESS_0'])], verbose=0)
        Y_testing_score = model.predict_proba(X.iloc[test_index])[:,1]
        roc_auc_testing = roc_auc_score(Y.iloc[test_index]['TRTMNT_SUCCESS_0'], Y_testing_score)
        auc_ave += roc_auc_testing/cvSplit
        Y_validation_score += model.predict_proba(X_validation)[:,1]/cvSplit
        Y_training_score += model.predict_proba(X)[:,1]/cvSplit
        model_list.append(model)

    if auc_ave > best_auc:
        coef_best = params_temp
        best_auc = auc_ave
        Y_validation_score_best = copy.copy(Y_validation_score)
        Y_training_score_best = copy.copy(Y_training_score) 
        validation_auc_best = roc_auc_score(Y_validation['TRTMNT_SUCCESS_0'], Y_validation_score_best)
        training_auc_best = roc_auc_score(Y['TRTMNT_SUCCESS_0'], Y_training_score_best)
        model_best = copy.deepcopy(model_list)
        print('{}, Testing AUC: {}, Training AUC: {}, Validation AUC: {}'.format(params_temp, auc_ave, training_auc_best, validation_auc_best))
        
