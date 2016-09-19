# -*- coding: cp1252 -*-

import time
start_time = time.time()
import numpy as np,pandas
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
random.seed(100)

from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    rfc=RandomForestClassifier(n_estimators=50)


    #skf=KFold(n=4, n_folds=2, shuffle=False,random_state=None)
    avgprecision=0
    avgrecall=0
    avgfscore=0
    #print(len(skf))
    ###############################################################################
    # Plot c   alibration plots

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for clf,  name ,files in [(rfc, 'UKST red',"class1.csv"),
                      (rfc, 'UKST infrared',"class2.csv"),
                      (rfc, 'ESO RED',"class3.csv"),
                      (rfc, 'POS 1 RED',"class4.csv"),(rfc, 'POS 1 RED(+2.5)',"class5.csv"),(rfc, 'POS 2 BLUE',"class6.csv"),(rfc, 'POS 2 RED',"class7.csv"),(rfc, 'POS 2 INFRARED',"class8.csv")]:

        if files in ["class6.csv","class7.csv","class8.csv"]:
            n = sum(1 for line in open(files)) - 1  # number of records in file (excludes header)
            s = 36496  # desired sample size
            skip = sorted(
                random.sample(xrange(1, n + 1), n - s))  # the 0-indexed header will not be included in the skip list
            df = pandas.read_csv(files, skiprows=skip)
            contents=df.values.tolist()
            flag=1
        else:
            fd = open(files, "r")
            contents = fd.readlines()
            flag=0

        X = []
        y = []

        for i in contents:
            if flag:
               a=i
            else:
                a=i.split(",")
            X.append(a[:len(a) - 2])
            y.append(a[len(a) - 1])
        # print(X[0:2])
        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)
        lb = preprocessing.LabelBinarizer()
        #y=lb.fit_transform(y)
        y=np.array(y).astype(np.float64)
        y[y!=4]=0

        X_train, X_test, y_train, y_test = train_test_split(np.array(X).astype(np.float64), np.array(y).astype(np.float64), test_size=0.20, random_state=42)
        clf.fit((X_train), (y_train))
        y_pred=clf.predict(X_test)
        print log_loss(y_test, y_pred)#print classification_report(y_test,y_pred)
    #     if hasattr(clf, "predict_proba"):
    #         prob_pos = np.array(clf.predict_proba(np.array(X_test).astype(np.float64))).astype(np.float64)[:,1]
    #     else:  # use decision function
    #         prob_pos = clf.decision_function(np.array(X_test).astype(np.float64))
    #         prob_pos = \
    #             (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    #     print "passed"
    #     fraction_of_positives, mean_predicted_value = \
    #         calibration_curve(y_test, prob_pos, n_bins=10)
    #
    #     ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
    #              label="%s" % (name,))
    #
    #     ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
    #              histtype="step", lw=2)
    #
    # ax1.set_ylabel("Fraction of positives")
    # ax1.set_ylim([-0.05, 1.05])
    # ax1.legend(loc="lower right")
    # ax1.set_title('Calibration plots  (reliability curve) Class 4')
    #
    # ax2.set_xlabel("Mean predicted value")
    # ax2.set_ylabel("Count")
    # ax2.legend(loc="upper center", ncol=2)
    # plt.show()
    # for train_index, test_index in skf:
    #     #print(train_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     rfc.fit(X_train,y_train)
    #     ypred=rfc.predict(X_test)
    #     avgprecision+=precision_score(y_test, ypred)
    #     avgrecall+=recall_score(y_test, ypred)
    #     avgfscore+=f1_score(y_test,ypred)
    #
    #     #print classification_report(y_test,ypred)
    # print(avgprecision/len(skf))
    # print(avgrecall/len(skf))
    # print(avgfscore/len(skf))