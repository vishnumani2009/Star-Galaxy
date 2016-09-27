# -*- coding: cp1252 -*-

import time
start_time = time.time()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import *
import os
from sklearn.cross_validation import *
random.seed(100)
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
ff=open("results6.csv","w+")
dirs=os.listdir("..\\class6")


def main():
    ii=0
    tp = 0
    tr = 0
    tf = 0
    for files in dirs:
        print(files,ii)
        rfc=RandomForestClassifier(n_estimators=50)
        ii=ii+1
        fd=open("..\\class6\\"+files,"r")
        contents=fd.readlines()
        X = []
        y = []

        for i in contents:
            a = i.split(",")
            X.append(a[:len(a) - 2])
            y.append(a[len(a) - 1])
        # print(X[0:2])

        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)
       # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
        #skf = StratifiedKFold(y, n_folds=2)
        avgprecision = 0
        avgrecall = 0
        avgfscore = 0
        skf=KFold(n=len(X), n_folds=2, shuffle=False,random_state=None)
        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            gbm = GradientBoostingClassifier().fit(X_train,y_train)

            #gbm = xgb.XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05).fit(X_train, y_train)
            ypred = gbm.predict(X_test)
            avgprecision += precision_score(y_test, ypred)
            avgrecall += recall_score(y_test, ypred)
            avgfscore += f1_score(y_test, ypred)
        print avgprecision/len(skf),avgrecall/len(skf),avgfscore/len(skf)
        print>> ff,avgprecision/len(skf),avgrecall/len(skf),avgfscore/len(skf)

                # print classification_report(y_test,ypred)

        #print>>ff,str(recall_score(testy,ypred,average=None))+","+str(precision_score(testy,ypred,average=None))+","+str(f1_score(testy,ypred,average=None))

main()
