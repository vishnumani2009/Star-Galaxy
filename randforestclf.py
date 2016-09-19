# -*- coding: cp1252 -*-

import time
start_time = time.time()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.calibration import calibration_curve
random.seed(100)

from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    rfc=RandomForestClassifier(n_estimators=50)


    skf=KFold(n=4, n_folds=2, shuffle=False,random_state=None)
    avgprecision=0
    avgrecall=0
    avgfscore=0
    #print(len(skf))
    ###############################################################################
    # Plot calibration plots


    fd = open("class1.csv", "r")
    contents = fd.readlines()

    X = []
    y = []

    for i in contents:
            a = i.split(",")
            X.append(a[:len(a) - 2])
            y.append(a[len(a) - 1])
        # print(X[0:2])
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)

    for train_index, test_index in skf:
        #print(train_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rfc.fit(X_train,y_train)
        ypred=rfc.predict(X_test)
        avgprecision+=precision_score(y_test, ypred)
        avgrecall+=recall_score(y_test, ypred)
        avgfscore+=f1_score(y_test,ypred)

        print classification_report(y_test,ypred)
    print(avgprecision/len(skf))
    print(avgrecall/len(skf))
    print(avgfscore/len(skf))