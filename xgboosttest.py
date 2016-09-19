# -*- coding: cp1252 -*-

import time
start_time = time.time()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
random.seed(100)
import xgboost as xgb

if __name__ == '__main__':
    rfc=RandomForestClassifier(n_estimators=50)
    fd=open("class5.csv","r")
    contents=fd.readlines()

    X=[]
    y=[]

    for i in contents:
        a=i.split(",")
        X.append(a[:len(a)-2])
        y.append(a[len(a)-1])
    #print(X[0:2])
    X=np.array(X).astype(np.float32)
    y=np.array(y).astype(np.float32)
    skf = StratifiedKFold(y, n_folds=10)
    #skf=KFold(n=4, n_folds=2, shuffle=False,random_state=None)
    avgprecision=0
    avgrecall=0
    avgfscore=0
    #print(len(skf))
    for train_index, test_index in skf:
        #print(train_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        gbm = xgb.XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05).fit(X_train, y_train)
        ypred = gbm.predict(X_test)
        #rfc.fit(X_train,y_train)
        #ypred=rfc.predict(X_test)
        avgprecision+=precision_score(y_test, ypred)
        avgrecall+=recall_score(y_test, ypred)
        avgfscore+=f1_score(y_test,ypred)

        #print classification_report(y_test,ypred)
    print(avgprecision/len(skf))
    print(avgrecall/len(skf))
    print(avgfscore/len(skf))


# #!/usr/bin/python
# import numpy as np
# import xgboost as xgb
# import pickle
# import xgboost as xgb
#
# import numpy as np
# from sklearn.cross_validation import KFold, train_test_split
# from sklearn.metrics import confusion_matrix, mean_squared_error
# from sklearn.grid_search import GridSearchCV
# from sklearn.datasets import load_iris, load_digits, load_boston
#
# ###
# # advanced: customized loss function
# #
# print ('start running example to used customized objective function')
#
# dtrain = xgb.DMatrix('agaricus.txt.train.txt')
# dtest = xgb.DMatrix('agaricus.txt.test.txt')
#
# # note: for customized objective function, we leave objective as default
# # note: what we are getting is margin value in prediction
# # you must know what you are doing
# param = {'max_depth': 2, 'eta': 1, 'silent': 1}
# watchlist = [(dtest, 'eval'), (dtrain, 'train')]
# num_round = 2
#
# # user define objective function, given prediction, return gradient and second order gradient
# # this is log likelihood loss
# def mani( labels,preds):
#     #print "mani"
#     label = labels
#     preds = 1.0 / (1.0 + np.exp(-preds))
#     grad = preds - labels
#     hess = preds * (1.0-preds)
#     return grad, hess
# #
# # def logregobj(preds, dtrain):
# #     label = dtrain.get_label()
# #     weight = dtrain.get_weight()
# #     weight = weight * float(test_size) / len(label)#refactorize with 55k test sample for CV
# #     s = sum( weight[i] for i in range(len(label)) if label[i] == 1.0 )
# #     b = sum( weight[i] for i in range(len(label)) if label[i] == 0.0 )
# #     ams = AMS(s,b)
# #     ds=(np.log(s/(b+10.)+1))/ams;
# #     db=(((b+10)*np.log(s/(b+10.)+1)-s)/(b+10.))/ams;
# #     preds = 1.0 / (1.0 + np.exp(-preds))#sigmod it
# #     grad = ds*(preds-label)+db*(1-(preds-label))
# #     hess = np.ones(preds.shape)/300. #constant
# #     return grad, hess
#
# # user defined evaluation function, return a pair metric_name, result
# # NOTE: when you do customized loss function, the default prediction value is margin
# # this may make buildin evalution metric not function properly
# # for example, we are doing logistic loss, the prediction is score before logistic transformation
# # the buildin evaluation error assumes input is after logistic transformation
# # Take this in mind when you use the customization, and maybe you need write customized evaluation function
# # def evalerror(preds, dtrain):
# #     labels = dtrain.get_label()
# #     # return a pair metric_name, result
# #     # since preds are margin(before logistic transformation, cutoff at 0)
# #     return 'error', float(sum(labels != (preds > 0.0))) / len(labels)
# #
# # # training with customized objective, we can also do step by step training
# # # simply look at xgboost.py's implementation of train
# # bst = xgb.train(param, dtrain, num_round, watchlist, mani, evalerror)
# #
#
#
# rng = np.random.RandomState(31337)
#
# def me(preds,dtrain):
#     print type(preds),type(dtrain)
#     print "me"
#     print dtrain.get_weight()
#     labels = dtrain.get_label()
#
#     # return a pair metric_name, result
#     # since preds are margin(before logistic transformation, cutoff at 0)
#     return 'error', float(sum(labels != (preds > 0.0))) / len(labels)
#
#
# print("Zeros and Ones from the Digits dataset: binary classification")
# digits = load_digits(2)
# y = digits['target']
# X = digits['data']
# kf = KFold(y.shape[0], n_folds=2, shuffle=True, random_state=rng)
# for train_index, test_index in kf:
#     Xtrain = X[train_index]
#     ytrain = y[train_index]
#     Xtest = X[test_index]
#     ytest = y[test_index]
#     xgb_model = xgb.XGBClassifier().fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)], eval_metric=me)
#     predictions = xgb_model.predict(X[test_index])
#     actuals = y[test_index]
#     print(confusion_matrix(actuals, predictions))
