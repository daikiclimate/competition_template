import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import pickle

import utils

def data_loader():
    train = pd.read_csv("data/")
    test = pd.read_csv("data/")
    train2 = utils.reduce_mem_usage(train)
    train2.to_csv("data/train2.csv")
    test2 = utils.reduce_mem_usage(test)
    test2.to_csv("data/test2.csv")
    exit()
    return train, test

def data_loader2():
    train = pd.read_csv("data/train2.csv", index_col = 0)
    test = pd.read_csv("data/test2.csv", index_col = 0)
    return train, test


def main():
    train, test = data_loader()
    # train, test = data_loader2()

    feature_cols = [
            "",
    ]
    target_col = "y"
    
    param_search = False
    if param_search:
        grid_param ={'n_estimators':[200,300,1000],#2000
                     'max_depth':[4,6,8],#8, 16
                     'num_leaves':[10,7,3],
                     'learning_rate':[0.05]}#0.1, 0.05
        
        X_train, X_test = utils.make_feature(train.copy(), test.copy())

        clf = lgb.LGBMClassifier()
        gscv = GridSearchCV(clf, grid_param, cv=4, verbose=3)
        gscv.fit(tr[feature_cols], train[target_col])

        # スコアの一覧を取得
        gs_result = pd.DataFrame.from_dict(gscv.cv_results_)
        #gs_result.to_csv('gs_result.csv')
        # 最高性能のモデルを取得し、テストデータを分類
        best = gscv.best_estimator_
        print(best)
    else:
        params ={
            'n_estimators':200,#2000
                     'max_depth':6,#8, 16
                     'num_leaves':30,
                     'learning_rate':0.05}
    print("training feature")

    # kf=KFold(n_splits=5, random_state = 0)
    kf=StratifiedKFold(n_splits=5, random_state = 14)
    score = 0
    counter = 1

    path = "data/featured_data.pkl"
    a = True
    a = False
    if a:
        train, test = utils.make_feature(train, test)
        print(train.head(2))
        with open(path, 'wb') as f:
            pickle.dump([train, test],f )
            print("finish make dataset")
            exit()
    else:
        with open(path, 'rb') as f:
            train, test = pickle.load(f)
 
    models = []
    for train_index, valid_index in kf.split(train, train[target_col]):
            # break
        
            train_X,valid_X = train.loc[train_index,:].copy()  , train.loc[valid_index,:].copy()
            tr, te = utils.make_feature(train_X, valid_X)

            t4 = time.time()
            
            X_train, X_valid = tr[feature_cols] , te[feature_cols]
            y_train, y_valid = tr[target_col], te[target_col]

            print(X_train.shape)
            
            clf = lgb.LGBMClassifier()
            # clf = lgb.LGBMClassifier(**params)

            clf.fit(X_train,y_train)
            
            preds = clf.predict(X_valid[feature_cols])

            #evaluation
            print(len(y_valid[y_valid == 1]), "/", len(preds[preds == 1]))
            
            acc_score = acc(y_valid,preds)
            print(f"fold{counter} score is :{acc_score} :", acc_score)
            score += acc_score
            counter += 1
            t5 = time.time()
            print("learning:",round(t5-t4,1))

            models.append(clf)

    print("average : ",round(score/5,5), ":", round(score2/5,5))
    y_pred = [model.predict_proba(test[feature_cols].values) for model in models]
    y_pred = np.array(y_pred)
    y_pred = np.mean(y_pred, axis=0)

    #提出用　全データ
    # tr, te = utils.make_feature(train, test)
    #
    # X_train, X_valid = tr[feature_cols], te[feature_cols]
    # y_train, y_valid = tr[target_col], te[target_col]
    # print(X_train.shape)
    #
    # clf = lgb.LGBMClassifier().fit(X_train[feature_cols].fillna(0),y_train)
    #
    # pred_test = clf.predict(X_valid)


    #make submit
    #よう調整
    pd.DataFrame({"id": range(len(pred_test)), target_col: pred_test }).to_csv("submission.csv", index=False)

    importance = pd.DataFrame(clf.booster_.feature_importance(importance_type='gain'), index=feature_cols, columns = ["f"])
    print(importance.sort_values("f", ascending = False).head(15))
    print(importance.sort_values("f", ascending = False).tail(15))

def make_weight(target):
    rt = np.ones(len(target))
    rt[target == 4] = 0.1
    return rt

if __name__ == "__main__":
    main()
