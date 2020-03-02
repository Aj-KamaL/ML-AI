#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.image as mpimg
import numpy as np
import sys
import os
import gzip
import math
import matplotlib.pyplot as plt
import random
import copy
import csv
import pandas as pd
import seaborn as sn
import cv2
import glob
import pickle
import scipy
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import random
import pickle
from string import ascii_uppercase
from collections import OrderedDict


# In[2]:


data=np.array(pd.read_csv("letter-recognition.data"))
print(len(data))
images=data[:,1:16]
label=data[:,0]
# print(images)
NoofClasses=len(np.unique(label))
# print(NoofClasses)
# OrderedDict((k, i+1) for i, k in enumerate(ascii_uppercase))


# In[3]:


x=np.array(images)
mix = list(zip(x, label))
random.shuffle(mix)
data,labels = zip(*mix)

train=[]
test=[]
trainl=[]
testl=[]
for i in range(0,len(labels)):
    if(i<(.7*len(labels))):
        train.append(data[i])
        trainl.append(labels[i])
    else:
        test.append(data[i])
        testl.append(labels[i])
train=np.array(train)
test=np.array(test)


# In[4]:


def bagging(arr1,arr2,arr3,arr4,bags,elmnts):
    probs=np.zeros((len(arr3),NoofClasses))
    probs2=np.zeros((len(arr2),NoofClasses))
    votes=np.zeros((len(arr3),NoofClasses))
    votes2=np.zeros((len(arr2),NoofClasses))
    for i in range(0,bags):
        ttrain=[]
        ttrainl=[]
        for j in range (0,elmnts):
            indx=random.randint(0,len(arr1)-1)
            ttrain.append(arr1[indx])
            ttrainl.append(arr2[indx])
        clf = DecisionTreeClassifier(max_depth=2,max_leaf_nodes=5)
        clf = clf.fit(ttrain,ttrainl)
        y_pred = clf.predict(arr3)
        acc = (y_pred==arr4).sum()/len(arr4)
        print(acc)
        tprob1=clf.predict_proba(arr3)
        tprob2=clf.predict_proba(arr1)
        for j in range(0,len(tprob1)):
            predl=np.argmax(tprob1[j])
            votes[j][predl]+=1 
        for j in range(0,len(tprob2)):
            predl2=np.argmax(tprob2[j])
            votes2[j][predl2]+=1
    print("Bagging Test")   
    crrct=0
    for i in range(0,len(votes)):
        predl3=np.argmax(votes[i])
        if (predl3+1)==ord(arr4[i])-64:
            crrct=crrct+1
    print(crrct/len(arr4)*100)
    print("Bagging Train")   
    crrct2=0
    for i in range(0,len(votes2)):
        predl4=np.argmax(votes2[i])
        if (predl4+1)==ord(arr2[i])-64:
            crrct2=crrct2+1
    print(crrct2/len(arr2)*100)


# In[5]:


def zscore(arr1):
    for i in range(0,len(arr1)):
        v = arr1[i]
        arr1[i] = (v - np.mean(v)) / np.std(v)
    return arr1
    
    
def minmax(arr2):
    for i in range(0,len(arr2)):
        v = arr2[i]
        arr2[i] = (v - v.min()) / (v.max() - v.min())
    return arr2
    
def tanh(arr3):
    for i in range(0,len(arr3)):
        v = arr3[i]
        arr3[i] = np.tanh(v)
    return arr3    


# In[6]:


def bagging_norm_tanh(arr1,arr2,arr3,arr4,bags,elmnts):
    probs=np.zeros((len(arr3),NoofClasses))
    probs2=np.zeros((len(arr2),NoofClasses))
    votes=np.zeros((len(arr3),NoofClasses))
    votes2=np.zeros((len(arr2),NoofClasses))

    for i in range(0,bags):
        ttrain=[]
        ttrainl=[]
        for j in range (0,elmnts):
            indx=random.randint(0,len(arr1)-1)
            ttrain.append(arr1[indx])
            ttrainl.append(arr2[indx])
#         Model.append(ttrain)
#         Modell.append(ttrainl)
        clf = DecisionTreeClassifier(max_depth=2,max_leaf_nodes=5)
        clf = clf.fit(ttrain,ttrainl)
        y_pred = clf.predict(arr3)
        acc = (y_pred==arr4).sum()/len(arr4)
        print(acc)
        tprob1=clf.predict_proba(arr3)
        tprob2=clf.predict_proba(arr1)
        tprob1=tanh(tprob1)
        tprob2=tanh(tprob2)
        for j in range(0,len(tprob1)):
            predl=np.argmax(tprob1[j])
            votes[j][predl]+=1 
        for j in range(0,len(tprob2)):
            predl2=np.argmax(tprob2[j])
            votes2[j][predl2]+=1
        probs=probs+tprob1
        probs2=probs2+tprob2

    
    print("Bagging Test")   
#     print(probs)
    crrct=0
    for i in range(0,len(probs)):
        predl=np.argmax(probs[i])
        if (predl+1)==ord(arr4[i])-64:
            crrct=crrct+1
    print("Label")
    
    print(crrct/len(arr4)*100)
    print("Bagging Train")   
    crrct2=0
    for i in range(0,len(probs2)):
        predl2=np.argmax(probs2[i])
        if (predl2+1)==ord(arr2[i])-64:
            crrct2=crrct2+1
    print(crrct2/len(arr2)*100)


# In[7]:


def bagging_norm_zscore(arr1,arr2,arr3,arr4,bags,elmnts):
    probs=np.zeros((len(arr3),NoofClasses))
    probs2=np.zeros((len(arr2),NoofClasses))

    for i in range(0,bags):
        ttrain=[]
        ttrainl=[]
        for j in range (0,elmnts):
            indx=random.randint(0,len(arr1)-1)
            ttrain.append(arr1[indx])
            ttrainl.append(arr2[indx])
#         Model.append(ttrain)
#         Modell.append(ttrainl)
        clf = DecisionTreeClassifier(max_depth=2,max_leaf_nodes=5)
        clf = clf.fit(ttrain,ttrainl)
        y_pred = clf.predict(arr3)
        acc = (y_pred==arr4).sum()/len(arr4)
        print(acc)
        tprob1=clf.predict_proba(arr3)
        tprob2=clf.predict_proba(arr1)
        tprob1=zscore(tprob1)
        tprob2=zscore(tprob2) 
        probs=probs+tprob1
        probs2=probs2+tprob2

    
    print("Bagging Test")   
#     print(probs)
    crrct=0
    for i in range(0,len(probs)):
        predl=np.argmax(probs[i])
        if (predl+1)==ord(arr4[i])-64:
            crrct=crrct+1
    print("Label")
    
    print(crrct/len(arr4)*100)
    print("Bagging Train")   
    crrct2=0
    for i in range(0,len(probs2)):
        predl2=np.argmax(probs2[i])
        if (predl2+1)==ord(arr2[i])-64:
            crrct2=crrct2+1
    print(crrct2/len(arr2)*100)


# In[8]:


def bagging_norm_minmax(arr1,arr2,arr3,arr4,bags,elmnts):
    probs=np.zeros((len(arr3),NoofClasses))
    probs2=np.zeros((len(arr2),NoofClasses))

    for i in range(0,bags):
        ttrain=[]
        ttrainl=[]
        for j in range (0,elmnts):
            indx=random.randint(0,len(arr1)-1)
            ttrain.append(arr1[indx])
            ttrainl.append(arr2[indx])
#         Model.append(ttrain)
#         Modell.append(ttrainl)
        clf = DecisionTreeClassifier(max_depth=2,max_leaf_nodes=5)
        clf = clf.fit(ttrain,ttrainl)
        y_pred = clf.predict(arr3)
        acc = (y_pred==arr4).sum()/len(arr4)
        print(acc)
        tprob1=clf.predict_proba(arr3)
        tprob2=clf.predict_proba(arr1)
        tprob1=minmax(tprob1)
        tprob2=minmax(tprob2)        
        probs=probs+tprob1
        probs2=probs2+tprob2

    
    print("Bagging Test")   
#     print(probs)
    crrct=0
    for i in range(0,len(probs)):
        predl=np.argmax(probs[i])
        if (predl+1)==ord(arr4[i])-64:
            crrct=crrct+1
    print("Label")
    
    print(crrct/len(arr4)*100)
    print("Bagging Train")   
    crrct2=0
    for i in range(0,len(probs2)):
        predl2=np.argmax(probs2[i])
        if (predl2+1)==ord(arr2[i])-64:
            crrct2=crrct2+1
    print(crrct2/len(arr2)*100)


# In[9]:


def retrnindx(ar1,elmnt):
#     print("******************************************************************")
#     print(elmnt)
#     print(arr1)
    for k in range(0,len(ar1)):
        if ar1[k]>=elmnt:
            return k
#     return k
def retnorm(ar2):
    summ=np.sum(ar2)
    for k in range(0,len(ar2)):
#         print("summ "+str(summ))
        ar2[k]=ar2[k]/summ
    return ar2


# In[18]:


def boosting(arrr1,arr2,arrr3,arr4,bags):
    erate=[]
    arr1 = copy.deepcopy(arrr1)
    arr3 = copy.deepcopy(arrr3)
    for i in range(0,len(arr1)):
        arr1[i]=ord(arr1[i])-64
    for i in range(0,len(arr3)):
        arr3[i]=ord(arr3[i])-64        
    TrPred, TePred = [np.zeros(len(arr2)), np.zeros(len(arr4))]
    probs = np.zeros((len(arr4),len(np.unique(arr1))))
    Tprod = np.ones(len(arr2)) /len(arr2)
    probs2=np.zeros((len(arr2),len(np.unique(arr1))))   
    Talpha = 1
    for i in range(bags):
        Tclf = DecisionTreeClassifier(max_depth = 2, max_leaf_nodes = 5)
        Tclf.fit(arr2,arr1, sample_weight = Tprod)        
        TrPred = Tclf.predict(arr2)
        TePred = Tclf.predict(arr4)
        probs+=Tclf.predict_proba(arr4)*Talpha
        probs2+=Tclf.predict_proba(arr2)*Talpha
        miss = [int(x) for x in (TrPred!=arr1)]        
        miss2 = [x if x==1 else -1 for x in miss] 
        
        Terr = np.dot(Tprod,miss)/sum(Tprod) + 0.045
        erate.append(Terr)
        Talpha = 0.5*np.log((1-Terr)/(float(Terr)))
        Tprod = np.multiply(Tprod, np.exp([float(x)*-Talpha for x in miss2]))
        Tprod = Tprod/np.sum(Tprod)
    crr1 = 0
    crr2 = 0
    for i in range(len(arr3)):
        PrCl1 = np.argmax(probs[i])+1
        crr1 += arr3[i]==PrCl1
    for i in range(len(arr1)):
        PrCl2 = np.argmax(probs2[i])+1
        crr2 += arr1[i]==PrCl2
    print("Boosting Test")
#     Validation and Training
    print(crr1/len(arr3))
    print("Boosting_Train")
    print(crr2/len(arr1))
    print(np.mean(np.array(erate)))
    return crr1/len(arr3)


# In[11]:


def crssvald_boost(dta,length,tesst,lentst,lbels,tlbels,CK2):
    CVA=[]
    CVC=[]

    CVacc=[]
    for i in range(0,5):
        ttrain=[]
        ttest=[]
        ttrainl=[]
        ttestl=[]       
        for j in range(0,len(dta)):
            if j>=(i*(0.2*length)) and j<((i+1)*(0.2*length)):
                ttest.append(dta[j])
                ttestl.append(lbels[j])
            else:
                ttrain.append(dta[j])
                ttrainl.append(lbels[j])
        train_2=np.array(ttrain)
        test_2=np.array(ttest)
        
        CVA.append(train_2)
        CVC.append(ttrainl)
        CViprob=[]
        for i in range(0,len(train_2)):
            CViprob.append(1/len(train_2))
        # print(len(iprob))
        CViprob=np.cumsum(CViprob)
        CVacc.append(boosting(ttrainl,train_2,ttestl,test_2,CK2))
    CVacc=np.array(CVacc)
    print(np.mean(CVacc),np.std(CVacc))
    print("Accuracy on test from best Cross Validation Model")
    return CVA[np.argmax(CVacc)],CVC[np.argmax(CVacc)]


# In[12]:


# def boosting(arr1,arr2,arr3,arr4,bags,elmnts,i_prob):
#     probs=np.zeros((len(arr4),NoofClasses))
#     probs2=np.zeros((len(arr2),NoofClasses))
#     for i in range(0,bags):
#         ttrain=[]
#         ttrainl=[]
#         for j in range (0,elmnts):
#             prob1=random.random()
#             indx=retrnindx(i_prob,prob1)
# #             print(prob1,indx)
#             ttrain.append(arr1[indx])
#             ttrainl.append(arr2[indx])
#         clf = DecisionTreeClassifier(max_depth=2,max_leaf_nodes=5)
# #         clf = DecisionTreeClassifier()

#         clf = clf.fit(ttrain,ttrainl)
#         y_pred = clf.predict(ttrain)
#         acc = (y_pred==ttrainl).sum()/len(ttrainl)
#         terr=(y_pred!=ttrainl).sum()/len(ttrainl)+0.9
# #         print("terr "+str(acc))
#         alpha=0.5*np.log((1-terr/100)/(terr/100))
# #         print("alp "+str(alpha))
#         for j in range(0,len(ttrainl)):
#             if(ttrainl[j]!=y_pred[j]):
#                 i_prob[j]=i_prob[j]*np.exp(alpha)
#             else:
#                 i_prob[j]=i_prob[j]*np.exp(-alpha)
# #         print(i_prob)
#         i_prob=retnorm(i_prob)
#         i_prob=np.cumsum(i_prob)
        
#         y_pred2 = clf.predict(arr3)
#         acc2 = (y_pred2==arr4).sum()/len(arr4)
# #         print(acc2)
        
#         probs=probs+(clf.predict_proba(arr3))
#         probs2=probs2+(clf.predict_proba(arr1))
#     print("Boosting Test")  
#     crrct=0
#     for i in range(0,len(probs)):
#         predl=np.argmax(probs[i])
# #         print(predl+1,testl[i])
#         if (predl+1)==ord(arr4[i])-64:
#             crrct=crrct+1
#     RVal=crrct/len(arr4)*100
#     print(RVal)
#     print("Boosting_Train")   
#     crrct2=0
#     for i in range(0,len(probs2)):
#         predl2=np.argmax(probs2[i])
# #         print(predl+1,testl[i])
#         if (predl2+1)==ord(arr2[i])-64:
#             crrct2=crrct2+1
#     print(crrct2/len(arr2)*100)
#     return(RVal)


# In[13]:


size=int(np.round(.7*len(train)))
bagging(train,trainl,test,testl,10,size)


    


# In[14]:


bagging_norm_minmax(train,trainl,test,testl,10,size)


# In[15]:


bagging_norm_zscore(train,trainl,test,testl,10,size)


# In[16]:


bagging_norm_tanh(train,trainl,test,testl,10,size)


# In[ ]:


iprob=[]
for i in range(0,len(train)):
    iprob.append(1/len(train))
# print(len(iprob))
iprob=np.cumsum(iprob)
# print(testl)
boosting(trainl,train,testl,test,100)


# In[ ]:


RA,RC=crssvald_boost(train,len(trainl),test,len(testl),trainl,testl,500)
Riprob=[]
for i in range(0,len(RA)):
    Riprob.append(1/len(RA))
# print(len(iprob))
Riprob=np.cumsum(Riprob)
boosting(RC,RA,testl,test,10)


# In[ ]:


# np.mean([0.13818969828304717,
# 0.15302550425070846,
# 0.1611935322553759,
# 0.15769294882480414,
# 0.15002500416736123,
# 0.15769294882480414,
# 0.14902483747291215,
# 0.15769294882480414,
# 0.14352392065344224,
# 0.15769294882480414])


# In[ ]:




