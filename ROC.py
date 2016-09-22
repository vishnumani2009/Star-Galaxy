# -*- coding: cp1252 -*-

import time
start_time = time.time()
import numpy as np
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
import random
from sklearn.metrics import *
import matplotlib.pyplot as pl
from sklearn.preprocessing import label_binarize
import pandas as pd
import csv
random.seed(100)
sum_f1=0
sum_acc=0
sum_prec=0

sum_recall=0

if __name__ == '__main__':
    classifier=RandomForestClassifier(n_estimators=50)
    file2=open('resultAggr.csv','w')
    for i in range(1):
        fd=open("class1.csv" ,"r")

        contents=fd.readlines()
        train=random.sample(contents,int(len(contents)*0.7))
        test=list(set(contents)-set(train))
        trainx=[]
        trainy=[]

        testx=[]
        testy=[]
        for i in train:
            a=i.split(",")
            trainx.append(a[:len(a)-2])
            trainy.append(a[len(a)-1])
        print len(trainx),len(trainy)

        trainx=np.array(trainx).astype(np.float)
        trainy=np.array(trainy).astype(np.float)




        for i in test:
            a=i.split(",")
            testx.append(a[:len(a)-2])
            testy.append(a[len(a)-1])

        print len(testx),len(testy)
        testx=np.array(testx).astype(np.float)
        testy=np.array(testy).astype(np.float)


        classifier.fit(trainx,trainy)
        ypred=classifier.predict(testx)
        ytest1=np.zeros(len(testy))
        ypred1=np.zeros(len(testy))
        ytest2=np.zeros(len(testy))
        ypred2=np.zeros(len(testy))
        ytest3=np.zeros(len(testy))
        ypred3=np.zeros(len(testy))
        ytest4=np.zeros(len(testy))
        ypred4=np.zeros(len(testy))
        for i in range(len(testy)):
            if(ypred[i]==1):
                ytest1[i]=(testy[i])
                ypred1[i]=(ypred[i])
            elif(ypred[i]==2):
                # ytest2.append(testy[i])
                ypred2[i]=(ypred[i])
            elif(ypred[i]==3):
                # ytest3.append(testy[i])
                ypred3[i]=(ypred[i])
            elif(ypred[i]==4):
                # ytest4.append(testy[i])
                ypred4=(ypred[i])






        accSc = accuracy_score(testy,ypred)

        f1 = f1_score(testy,ypred)
        preSc = precision_score(testy,ypred)
        rec=recall_score(testy,ypred)
        l1=[accSc,f1,preSc,rec]
        sum_acc=sum_acc+accSc
        sum_prec=sum_prec+preSc
        sum_f1=sum_f1+f1
        sum_recall=sum_recall+rec
        print classification_report(testy,ypred)
    print testy,ypred
fpr1=[]
fpr2=[]
fpr3=[]
fpr4=[]
tpr1=[]
tpr2=[]
tpr3=[]
tpr4=[]
roc_auc=[[]*4]
k=[1,2,3,4]
for i in (k):
    if i==1:
        fpr1,tpr1,_=roc_curve(testy,ypred1,pos_label=[1.0])
    elif i==2:
        fpr2,tpr2,_=roc_curve(testy,ypred2,pos_label=[2.0])
    elif i==3:
        fpr3,tpr3,_=roc_curve(testy,ypred3,pos_label=[3.0])
    # else:
    #     fpr4,tpr4,_=roc_curve(testy,ypred4,pos_label=[4.0])

a1=auc(fpr1,tpr1)
a2=auc(fpr2,tpr2)
a3=auc(fpr3,tpr3)
pl.figure()
line1,=pl.plot(fpr1,tpr1,label='galaxy ')
line2,=pl.plot(fpr2,tpr2,label="star auc")
line3,=pl.plot(fpr3,tpr3,label="unclassifiable")
first_legend = pl.legend(handles=[line1,line2,line3], loc=1)
# second_legend = pl.legend(handles=[line2], loc=1)
# third_legend = pl.legend(handles=[line3], loc=1)

pl.xlabel('false positive ratio')
pl.ylabel('True positive ratio')

pl.show()
