# -*- coding: cp1252 -*-

import time
start_time = time.time()
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
import xgboost as xgb
from sklearn.metrics import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.calibration import calibration_curve
random.seed(100)
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
def test(rfc):
    #rfc=RandomForestClassifier(n_estimators=50)
    #rfc=GradientBoostingClassifier()
    #rfc=xgb.XGBClassifier()

    #print(len(skf))
    ###############################################################################
    # Plot calibration plots



    fd = open("class8\\class8-000.csv", "r")
    contents = fd.readlines()

    X = []
    y = []

    for i in contents:
            a = i.split(",")
            X.append(a[:len(a) - 2])
            y.append(a[len(a) - 1])
        # print(X[0:2])
    X = np.array(X).astype(np.float32)
    print len(X)

    y = np.array(y).astype(np.float32)
    skf = KFold(n=len(X), n_folds=2, shuffle=False, random_state=None)
    #y = label_binarize(y, classes=[1, 2, 3, 4])
    #n_classes = y.shape[1]
    avgprecision = 0
    avgrecall = 0
    avgfscore = 0
    for train_index, test_index in skf:
        #print(train_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rfc.fit(X_train,y_train)
        ypred=rfc.predict(X_test)
        avgprecision+=precision_score(y_test, ypred)
        avgrecall+=recall_score(y_test, ypred)
        avgfscore+=f1_score(y_test,ypred)

        # classifier = OneVsRestClassifier(rfc)
        # y_score = classifier.fit(X_train, y_train).predict(X_test)
        # print y_score
        # # Compute ROC curve and ROC area for each class
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        #
        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #
        # ##############################################################################
        # # Plot of a ROC curve for a specific class
        #
        #
        # ##############################################################################
        # # Plot ROC curves for the multiclass problem
        #
        # # Compute macro-average ROC curve and ROC area
        #
        # # First aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        #
        # # Then interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(n_classes):
        #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        #
        # # Finally average it and compute AUC
        # mean_tpr /= n_classes
        #
        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        #
        # # Plot all ROC curves
        # plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          linewidth=2)
        #
        # plt.plot(fpr["macro"], tpr["macro"],
        #          label='macro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["macro"]),
        #          linewidth=2)
        #
        # for i in range(n_classes):
        #     plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
        #                                    ''.format(i, roc_auc[i]))
        #
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic to POS 2 INFRARED')
        # plt.legend(loc="lower right")
        # plt.show()

        print classification_report(y_test,ypred)
    print(avgprecision/len(skf))
    print(avgrecall/len(skf))
    print(avgfscore/len(skf))

test(GradientBoostingClassifier())
test(xgb.XGBClassifier())