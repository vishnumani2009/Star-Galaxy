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
import matplotlib.pyplot as plt


def show_confusion_matrix(C, class_labels=['0', '1',"2","3"]):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert C.shape == (2, 2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    tn = C[0, 0];
    fp = C[0, 1];
    fn = C[1, 0];
    tp = C[1, 1];

    NP = fn + tp  # Num positive examples
    NN = tn + fp  # Num negative examples
    N = NP + NN

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.plot([-0.5, 2.5], [0.5, 0.5], '-k', lw=2)
    ax.plot([-0.5, 2.5], [1.5, 1.5], '-k', lw=2)
    ax.plot([0.5, 0.5], [-0.5, 2.5], '-k', lw=2)
    ax.plot([1.5, 1.5], [-0.5, 2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34, 1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''], rotation=90)
    ax.set_yticks([0, 1, 2])
    ax.yaxis.set_label_coords(-0.09, 0.65)

    # Fill in initial metrics: tp, tn, etc...
    ax.text(0, 0,
            'True Neg: %d\n(Num Neg: %d)' % (tn, NN),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 1,
            'False Neg: %d' % fn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 0,
            'False Pos: %d' % fp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 1,
            'True Pos: %d\n(Num Pos: %d)' % (tp, NP),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2, 0,
            'False Pos Rate: %.2f' % (fp / (fp + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 1,
            'True Pos Rate: %.2f' % (tp / (tp + fn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 2,
            'Accuracy: %.2f' % ((tp + tn + 0.) / N),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 2,
            'Neg Pre Val: %.2f' % (1 - fn / (fn + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 2,
            'Pos Pred Val: %.2f' % (tp / (tp + fp + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, filename,title='Confusion matrix', cmap=plt.cm.winter):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len( ["class0","class1","class2","class3"]))
    plt.xticks(tick_marks, ["class0","class1","class2","class3"], rotation=45)
    plt.yticks(tick_marks,  ["class0","class1","class2","class3"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("G:\\SG Classifier\\traindata\\Star-Galaxy\\CONFUSION MATRIX\\GBDTCS\\"+filename.replace("csv","png"))
    #plt.show()

def testmain():
    #rfc=RandomForestClassifier(n_estimators=50)
    rfc=GradientBoostingClassifier()
    #rfc=xgb.XGBClassifier()

    #print(len(skf))
    ###############################################################################
    # Plot calibration plots


    #class8\\class8-000.csv
    for file in ["class1.csv","class2.csv","class3.csv","class4.csv","class5.csv","class6\\class6-000.csv","class7\\class7-000.csv","class8\\class8-000.csv"]:
        print file
        fd = open(file, "r")
        contents = fd.readlines()

        X = []
        y = []

        for i in contents:
                a = i.split(",")
                X.append(a[:len(a) - 2])
                y.append(a[len(a) - 1])
            # print(X[0:2])
        X = np.array(X).astype(np.float32)
        #print len(X)

        y = np.array(y).astype(np.float32)
        skf = KFold(n=len(X), n_folds=2, shuffle=False, random_state=None)
        # y = label_binarize(y, classes=[1, 2, 3, 4])
        n_classes = 4
        avgprecision = 0
        avgrecall = 0
        avgfscore = 0
        for train_index, test_index in skf:
            #print(train_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            we=[]
            c=0.7
            for i in range(len(X_train)):
                if y_train[i]==1 or y_train[i]==2 :
                    we.append(1)
                else:
                    we.append(c)
            rfc.fit(X_train,y_train,sample_weight=we)
            ypred=rfc.predict(X_test)
            avgprecision+=precision_score(y_test, ypred)
            avgrecall+=recall_score(y_test, ypred)
            avgfscore+=f1_score(y_test,ypred)
            plt.figure()
            cm=confusion_matrix(y_test,ypred)
            plot_confusion_matrix(cm,filename=file)



            #print classification_report(y_test,ypred)
        print(avgprecision/len(skf))
        print(avgrecall/len(skf))
        print(avgfscore/len(skf))




testmain()



















#calibration curve codes
# # -*- coding: cp1252 -*-
#
# import time
# start_time = time.time()
# import numpy as np,pandas
# from sklearn.ensemble import RandomForestClassifier
# import random
# from sklearn.metrics import *
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.cross_validation import KFold
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from sklearn.calibration import calibration_curve
# from sklearn.metrics import mean_squared_error
# random.seed(100)
#
# from sklearn.cross_validation import train_test_split
#
# if __name__ == '__main__':
#     rfc=RandomForestClassifier(n_estimators=50)
#
#
#     #skf=KFold(n=4, n_folds=2, shuffle=False,random_state=None)
#     avgprecision=0
#     avgrecall=0
#     avgfscore=0
#     #print(len(skf))
#     ###############################################################################
#     # Plot c   alibration plots
#
#     plt.figure(figsize=(10, 10))
#     ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#     ax2 = plt.subplot2grid((3, 1), (2, 0))
#     ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#
#     for clf,  name ,files in [(rfc, 'UKST red',"class1.csv"),
#                       (rfc, 'UKST infrared',"class2.csv"),
#                       (rfc, 'ESO RED',"class3.csv"),
#                       (rfc, 'POS 1 RED',"class4.csv"),(rfc, 'POS 1 RED(+2.5)',"class5.csv"),(rfc, 'POS 2 BLUE',"class6.csv"),(rfc, 'POS 2 RED',"class7.csv"),(rfc, 'POS 2 INFRARED',"class8.csv")]:
#
#         if files in ["class6.csv","class7.csv","class8.csv"]:
#             n = sum(1 for line in open(files)) - 1  # number of records in file (excludes header)
#             s = 36496  # desired sample size
#             skip = sorted(
#                 random.sample(xrange(1, n + 1), n - s))  # the 0-indexed header will not be included in the skip list
#             df = pandas.read_csv(files, skiprows=skip)
#             contents=df.values.tolist()
#             flag=1
#         else:
#             fd = open(files, "r")
#             contents = fd.readlines()
#             flag=0
#
#         X = []
#         y = []
#
#         for i in contents:
#             if flag:
#                a=i
#             else:
#                 a=i.split(",")
#             X.append(a[:len(a) - 2])
#             y.append(a[len(a) - 1])
#         # print(X[0:2])
#         X = np.array(X).astype(np.float32)
#         y = np.array(y).astype(np.float32)
#         lb = preprocessing.LabelBinarizer()
#         #y=lb.fit_transform(y)
#         y=np.array(y).astype(np.float64)
#         y[y!=4]=0
#
#         X_train, X_test, y_train, y_test = train_test_split(np.array(X).astype(np.float64), np.array(y).astype(np.float64), test_size=0.20, random_state=42)
#         clf.fit((X_train), (y_train))
#         y_pred=clf.predict(X_test)
#         print log_loss(y_test, y_pred)#print classification_report(y_test,y_pred)
#     #     if hasattr(clf, "predict_proba"):
#     #         prob_pos = np.array(clf.predict_proba(np.array(X_test).astype(np.float64))).astype(np.float64)[:,1]
#     #     else:  # use decision function
#     #         prob_pos = clf.decision_function(np.array(X_test).astype(np.float64))
#     #         prob_pos = \
#     #             (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
#     #     print "passed"
#     #     fraction_of_positives, mean_predicted_value = \
#     #         calibration_curve(y_test, prob_pos, n_bins=10)
#     #
#     #     ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
#     #              label="%s" % (name,))
#     #
#     #     ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
#     #              histtype="step", lw=2)
#     #
#     # ax1.set_ylabel("Fraction of positives")
#     # ax1.set_ylim([-0.05, 1.05])
#     # ax1.legend(loc="lower right")
#     # ax1.set_title('Calibration plots  (reliability curve) Class 4')
#     #
#     # ax2.set_xlabel("Mean predicted value")
#     # ax2.set_ylabel("Count")
#     # ax2.legend(loc="upper center", ncol=2)
#     # plt.show()
#     # for train_index, test_index in skf:
#     #     #print(train_index)
#     #     X_train, X_test = X[train_index], X[test_index]
#     #     y_train, y_test = y[train_index], y[test_index]
#     #
#     #     rfc.fit(X_train,y_train)
#     #     ypred=rfc.predict(X_test)
#     #     avgprecision+=precision_score(y_test, ypred)
#     #     avgrecall+=recall_score(y_test, ypred)
#     #     avgfscore+=f1_score(y_test,ypred)
#     #
#     #     #print classification_report(y_test,ypred)
#     # print(avgprecision/len(skf))
#     # print(avgrecall/len(skf))
#     # print(avgfscore/len(skf))