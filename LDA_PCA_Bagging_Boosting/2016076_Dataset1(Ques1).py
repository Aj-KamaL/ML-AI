#!/usr/bin/env python
# coding: utf-8

# In[39]:


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


# In[40]:


# Change a little Bit
def split(Adata,Alabel,ratio):
    Atemp=np.arange(0,Adata.shape[0])
    np.random.shuffle(Atemp)
    
    An_train=int(ratio*Adata.shape[0])
    An_test=Adata.shape[0] - An_train
    
    Atrain_index=Atemp[:An_train]
    Atest_index=Atemp[An_train:]
    Atrain_x=[]
    Atrain_y=[]
    Atest_x=[]
    Atest_y=[]
    for i in Atrain_index:
        Atrain_x.append(Adata[i])
        Atrain_y.append(Alabel[i])
    for i in Atest_index:
        Atest_x.append(Adata[i])
        Atest_y.append(Alabel[i])
        
    Atrain_x=np.array(Atrain_x)
    Atrain_y=np.array(Atrain_y)
    Atest_x=np.array(Atest_x)
    Atest_y=np.array(Atest_y)
    return Atrain_x,Atrain_y,Atest_x,Atest_y


# In[41]:


def PM(BTrain,BTrainl):
    BClass1=[]
    BClass2=[]
    BClass3=[]
    BClass4=[]
    BClass5=[]
    BClass6=[]
    BClass7=[]
    BClass8=[]
    BClass9=[]
    BClass10=[]
    BClass11=[]
    for i in range(0,len(BTrain)):
        if(BTrainl[i]==1):
            BClass1.append(BTrain[i])
        elif(BTrainl[i]==2):
            BClass2.append(BTrain[i])
        elif(BTrainl[i]==3):
            BClass3.append(BTrain[i])
        elif(BTrainl[i]==4):
            BClass4.append(BTrain[i])
        elif(BTrainl[i]==5):
            BClass5.append(BTrain[i])
        elif(BTrainl[i]==6):
            BClass6.append(BTrain[i])
        elif(BTrainl[i]==7):
            BClass7.append(BTrain[i])
        elif(BTrainl[i]==8):
            BClass8.append(BTrain[i])
        elif(BTrainl[i]==9):
            BClass9.append(BTrain[i])
        elif(BTrainl[i]==10):
            BClass10.append(BTrain[i])
        elif(BTrainl[i]==11):
            BClass11.append(BTrain[i])
            
    Bx1 = np.array(BClass1)
    Bx2 = np.array(BClass2)
    Bx3 = np.array(BClass3)
    Bx4 = np.array(BClass4)
    Bx5 = np.array(BClass5)
    Bx6 = np.array(BClass6)
    Bx7 = np.array(BClass7)
    Bx8 = np.array(BClass8)
    Bx9 = np.array(BClass9)
    Bx10 = np.array(BClass10)
    Bx11 = np.array(BClass11)
    Btl=[Bx1,Bx2,Bx3,Bx4,Bx5,Bx6,Bx7,Bx8,Bx9,Bx10,Bx11]
#     c1=len(x1)*np.cov(x1.T)
#     c2=len(x2)*np.cov(x2.T)
#     c3=len(x3)*np.cov(x3.T)
#     c4=len(x4)*np.cov(x4.T)
#     c5=len(x5)*np.cov(x5.T)
#     c6=len(x6)*np.cov(x6.T)
#     c7=len(x7)*np.cov(x7.T)
#     c8=len(x8)*np.cov(x8.T)
#     c9=len(x9)*np.cov(x9.T)
#     c10=len(x10)*np.cov(x10.T)
#     c11=len(x11)*np.cov(x11.T)
    Bc1=len(Bx1-1)*np.cov(Bx1.T)
    Bc2=len(Bx2-1)*np.cov(Bx2.T)
    Bc3=len(Bx3-1)*np.cov(Bx3.T)
    Bc4=len(Bx4-1)*np.cov(Bx4.T)
    Bc5=len(Bx5-1)*np.cov(Bx5.T)
    Bc6=len(Bx6-1)*np.cov(Bx6.T)
    Bc7=len(Bx7-1)*np.cov(Bx7.T)
    Bc8=len(Bx8-1)*np.cov(Bx8.T)
    Bc9=len(Bx9-1)*np.cov(Bx9.T)
    Bc10=len(Bx10-1)*np.cov(Bx10.T)
    Bc11=len(Bx11-1)*np.cov(Bx11.T)

    BSW=Bc1+Bc2+Bc3+Bc4+Bc5+Bc6+Bc7+Bc8+Bc9+Bc10+Bc11

    Bm=[] 
    for i in range(0,11):
        Btmpm=[]
        for j in range(0,len(BTrain[0])):
            Btmpm.append(np.average(Btl[i][:,j]))
        Bm.append(Btmpm)

    Bm=np.array(Bm)

    Bum=[]
    for j in range(0,len(BTrain[0])):
        Bum.append(np.average(Bm[:,j]))
    Bum=np.array(Bum)
    BSB=np.zeros((len(BTrain[0]),len(BTrain[0])))

    for i in range(0,11):
        Btmdff=Bm[i]-Bum
        Btmdff=Btmdff.reshape(len(BTrain[0]),1)
        Btmdffb=Btmdff.reshape(1,len(BTrain[0]))
        Btmpm=np.dot(Btmdff,Btmdffb)
        BSB=BSB+(Btmpm*len(Btl[i]))

    BSS=np.dot(np.linalg.inv(BSW),BSB)
    return BSS


# In[42]:


# Change a little Bit
def lda(Carr1,CTrain,CTest,Ck1):
    LEV, LET = np.linalg.eig(Carr1)
    LEV = np.abs(LEV)
    eig_pair = []
    for i in range(len(LEV)):
        eig_pair.append([(np.abs(LEV[i]), LET[:,i])])
    eig_pair.sort(key=lambda x: x[0][0], reverse=True)
    var_exp = []
    tot = sum(LEV)
    for i in sorted(LEV, reverse = True):
        var_exp.append(100*i/tot)
    energy = np.cumsum(var_exp)
    topvec = Ck1
    project = np.zeros([topvec, len(CTrain[0])])
    for i in range(topvec):
        project[i] = eig_pair[i][0][1].reshape(len(CTrain[0]))
    project = project.T
    Ctt1=np.dot(CTrain,project)
    Ctt2=np.dot(CTest,project)    
    return Ctt1,Ctt2,topvec


# In[43]:


def pca(arr1,t1,t2,k1):   
    LEV, LET = np.linalg.eig(arr1)
    LEV = np.abs(LEV)
    eig_pair = []    
    for i in range(len(LEV)):
        eig_pair.append([(np.abs(LEV[i]), LET[:,i])])
    eig_pair.sort(key=lambda x: x[0][0], reverse=True)
    var_exp = []
    tot = sum(LEV)
    for i in sorted(LEV, reverse = True):
        var_exp.append(100*i/tot)
    energy = np.cumsum(var_exp)
    topvec = len(energy)
    for i in range(len(energy)):
        if(energy[i] >= 100*k1):
            topvec = i
            break
#     print("bvbvbvb")
#     print(len(t1[0]))
    project = np.zeros([topvec,len(t1[0])])
    for i in range(topvec):
        project[i] = eig_pair[i][0][1].reshape(len(t1[0]))
    project = project.T
    Ctt1=np.dot(t1,project)
    Ctt2=np.dot(t2,project)    
    return Ctt1,Ctt2,topvec,project


# In[44]:


def crssvald_pca(dta,length,tesst,lentst,lbels,tlbels):
    accur=[]
    classifier=[]
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
        
#         Pcvr=np.cov(train_2.T)
#         tmtrain,tmtest,idx=pca(Pcvr,train_2,test_2,0.95)
#         tmtrain=tmtrain.real
#         tmtest=tmtest.real
#         tmtestl = np.array(ttestl)
#         tmtrainl = np.array(ttrainl)
#         ss=[]
#         GaussianNB(priors=None, var_smoothing=1e-09)
#         for j in range(0,len(testl)):
#             ss.append(clf.predict(np.reshape(test[j],(1,idx))))
#         print(y_pred)
#         crt=0
#         print(ss)
#         for j in range(0,88):
        #     print(label2[i],ss[i])
#             if testl[j]==ss[j]:
#                 crt=crt+1

        clf = GaussianNB()
        clf.fit(train_2,ttrainl)
        classifier.append(clf)
        y_pred = clf.predict(test_2)
        acc = (y_pred==ttestl).sum()/len(ttestl)
        accur.append(acc)
        print(acc)
    accur=np.array(accur)
    return classifier[np.argmax(accur)]
        


# In[45]:


def ROC_Con(arr1,arr2):
    class_images=arr1.T
    acc=[]

    for i in range (0,len(class_images)):
        labels=[]
        threshold=np.arange(0, 1, 0.005)
        for j in range (0,len(threshold)):
            label=[]
            for k in range (0,len(class_images[0])):
                if class_images[i][k]>=threshold[j]:
                    label.append(i)
                else:
                    label.append(-100)
            labels.append(label)
        TPP=[]
        FPP=[]
        for j in range (0,len(labels)):
            T_TP=0
            T_NP=0
            F_FP=0
            F_NP=0    
            for k in range(0,len(arr2)):
                if arr2[k]==i+1:
                    if labels[j][k]==i:
                        T_TP+=1
                    else:
                        F_NP+=1
                elif arr2[k]!=i+1:
                    if labels[j][k]==-100:
                        T_NP+=1
                    else:
                        F_FP+=1                        
            TPP.append(T_TP/(T_TP+F_NP))
            FPP.append(F_FP/(T_NP+F_FP))
            acc.append(TPP)
    #     print(TPP)
    #     print(FPP)
        rdff='Class: '+str(i+1)
    #     print(rdff)
        plt.plot(FPP,TPP,label=rdff)
    #     plt.plot([0,1],[0,1],'b--')
    plt.xlabel('FPR')

    plt.ylabel('TPR')
    # plt.ylim(ymin=0)
    plt.legend(loc='lower right')
    plt.show()


# In[46]:


def make_Con(arr1,arr2):
    NoofC=len(np.unique(arr2))
    con_matrix=np.zeros((NoofC,NoofC))
    for i in range(0,len(arr1)):
        con_matrix[int(arr1[i])-1][int(arr2[i])-1]+=1
        
    df_cm = pd.DataFrame(con_matrix, index = [i for i in "ABCDEFGHIJK"],columns = [i for i in "ABCDEFGHIJK"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
#     plt.matshow(con_matrix)


# In[47]:


def crssvald_lda(Edta,Elength,tesst,lentst,Elbels,tlbels):
    accur=[]
    classifier=[]
    for i in range(0,5):
        ETrain=[]
        ETest=[]
        ETrainl=[]
        ETestl=[]
        Eptr=int(len(Edta)/5)
#         print(Eptr)
        for j in range(0,len(Edta)):
            if j>=(i*Eptr) and j<((i+1)*Eptr):
                ETest.append(Edta[j])
                ETestl.append(Elbels[j])
            else:
                ETrain.append(Edta[j])
                ETrainl.append(Elbels[j])
        ETrain=np.array(ETrain)
        ETrainl=np.array(ETrainl)
        ETestl=np.array(ETestl)
        ETest=np.array(ETest)
#         EMass=PM(ETrain,ETrainl)
#         Etrain,Etest,Eidx=lda(EMass,ETrain,ETest,0.99)
#         Etrain=Etrain.real
#         Etest=Etest.real
#         TTrain=np.array(TTrain)
#         TTest=np.array(TTest)
#         TTrainl=np.array(TTrainl)
#         TTestl=np.array(TTestl)
#         train,test,idx=do_lda(TTrain,TTest,TTrainl,TTestl,99)
        Eclf = GaussianNB()
        Eclf.fit(ETrain,ETrainl)
        Ey_pred = Eclf.predict(ETest)
        ECr=0
        for j in range(0,len(ETestl)):
            if Ey_pred[j]==ETestl[j]:
                ECr+=1
        acc = ECr/len(ETestl)
        print(acc)
        classifier.append(Eclf)
        accur.append(acc)
    accur=np.array(accur)
    return classifier[np.argmax(accur)]


# In[48]:


scx=0.1
scy=0.1
images=[]
label=[]
for i in range(0,11):
    address="Face_data/"+str(i+1)+"/*.pgm"
    files=glob.glob(address)
    for x in files:
        small = cv2.imread(x,-1)
        small = cv2.resize(small, (0,0), fx=scx, fy=scy)
        small=np.reshape(small,small.shape[0]*small.shape[1])
        images.append(small)
        label.append(i+1)
    
print(np.unique(label))


# In[49]:


x=np.array(images)
y=np.array(label)
Train,Trainl,Test,Testl=split(x,y,0.7)
NoofClasses=len(np.unique(Trainl))
fin_train = copy.deepcopy(Train)


# In[50]:


clf = GaussianNB()
clf.fit(Train,Trainl)
ss=[]
GaussianNB(priors=None, var_smoothing=1e-09)
for i in range(0,len(Testl)):
    ss.append(clf.predict(np.reshape(Test[i],(1,len(Train[0])))))

crt=0

for i in range(0,len(Test)):
#     print(label2[i],ss[i])
    if Testl[i]==ss[i]:
        crt=crt+1
print(crt/len(Test)*100)


# In[38]:


cvr=np.cov(Train.T)
np.reshape(Train,(len(Train),len(Train[0])))
np.reshape(Test,(len(Test),len(Train[0])))
# print(train.shape,test.shape)


# In[14]:


PTrain,PTest,Pidx,Pprojection=pca(cvr,Train,Test,0.95)
PTest=PTest.real
PTrain=PTrain.real
print(Pprojection.shape)

np.reshape(PTrain,(len(PTrain),Pidx))
np.reshape(PTest,(len(PTest),Pidx))

for i in range(Pprojection.shape[1]):
    plt.subplot(6, 5, i+1)
    plt.imshow(Pprojection[:, i].reshape((19, 17)),  cmap= plt.get_cmap('gray'))
plt.show()


# Pprojection=Pprojection.T
# Pvectors=Pprojection[:,:2]
# for i in range(0,len(Pvectors)):
#     tmppa=Pvectors[i][0]
#     tmppb=Pvectors[i][1]
#     Pvectors[i][0]=tmppa/(np.sqrt((tmppa*tmppa)+(tmppb*tmppb)))
#     Pvectors[i][1]=tmppb/(np.sqrt((tmppa*tmppa)+(tmppb*tmppb)))
#     rdff='Eigenvector: '+str(i+1)
#     plt.plot([0,Pvectors[i][0]],[0,Pvectors[i][1]])
# plt.xlabel('Feature1')
# plt.ylabel('Feature2')



# plt.legend(loc='lower right')
# print(Pvectors)


# In[15]:


clf = GaussianNB()
clf.fit(PTrain,Trainl)
Pss=[]
GaussianNB(priors=None, var_smoothing=1e-09)
for i in range(0,len(Testl)):
    Pss.append(clf.predict(np.reshape(PTest[i],(1,Pidx))))
crt=0
for i in range(0,len(Testl)):
#     print(label2[i],ss[i])
    if Testl[i]==Pss[i]:
        crt=crt+1
print(crt/len(Testl)*100)
Pprobs=clf.predict_proba(PTest)
print(Pprobs.shape)


# In[16]:


print(Pprobs.shape)
ROC_Con(Pprobs,Testl)


# In[17]:


Ptemp_ss=[]
for i in range(len(Pss)):
    Ptemp_ss.append(Pss[i][0])

make_Con(Ptemp_ss,Testl)
# print(type(ss))
# print(ss)


# In[18]:


Ptemp_clf=crssvald_pca(PTrain,len(PTrain),PTest,len(PTest),Trainl,Testl)
PA=Ptemp_clf.predict(PTest)
PAA=Ptemp_clf.predict(PTrain)

PB=Ptemp_clf.predict_proba(PTest)
ROC_Con(PB,Testl)
make_Con(PA,Testl)
Pacc = (PA==Testl).sum()/len(Testl)
Paccc = (PAA==Trainl).sum()/len(Trainl)
print("Accuracy from Best Model on Test Set")
print(Pacc)
print("Accuracy from Best Model on Train Set")
print(Paccc)


# In[19]:


Mass=PM(Train,Trainl)
DTrain,DTest,Didx=lda(Mass,Train,Test,NoofClasses-1)
print(Didx)
# features=len(Train[0])
# Strain,Stest,idx=do_lda(Train,Test,Trainl,Testl,99)

DTrain=DTrain.real
# Strain=Strain.real
# train=np.round(train)
DTest=DTest.real
# Stest=Stest.real
# test=np.round(test)
DTrain=np.reshape(DTrain,(len(Trainl),Didx))
DTest=np.reshape(DTest,(len(Testl),Didx))
# print(DTrain)


# In[20]:


plt.scatter(DTrain[:,0], DTrain[:,1], c = Trainl)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
# plt.legend(loc='lower right')
plt.show()
plt.scatter(DTest[:,0], DTest[:,1], c = Testl)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
# plt.legend(loc='lower right')
plt.show()


# In[21]:


Dclf = GaussianNB()
Dclf.fit(DTrain,Trainl)
Dss=[]
GaussianNB(priors=None, var_smoothing=1e-09)
for i in range(0,len(Testl)):
    Dss.append(Dclf.predict(np.reshape(DTest[i],(1,Didx))))
crt=0
for i in range(0,len(Testl)):
    if Testl[i]==Dss[i]:
        crt=crt+1
print(crt/len(Testl)*100)
Dprobs=Dclf.predict_proba(DTest)
# print(Dprobs.shape)


# In[22]:


ROC_Con(Dprobs,Testl)
# plot_multiclass_roc(Dprobs,Testl-1,"ROC Curve","ROC")


# In[23]:


Dtemp_ss=[]
for i in range(len(Dss)):
    Dtemp_ss.append(Dss[i][0])

make_Con(Dtemp_ss,Testl)


# In[24]:


Dtemp_clf=crssvald_lda(DTrain,len(DTrain),DTest,len(DTest),Trainl,Testl)

DA=Dtemp_clf.predict(DTest)
DAA=Dtemp_clf.predict(DTrain)

DB=Dtemp_clf.predict_proba(DTest)
ROC_Con(DB,Testl)
make_Con(DA,Testl)
Dacc = (DA==Testl).sum()/len(Testl)
Daccc = (DAA==Trainl).sum()/len(Trainl)
print("Accuracy from Best Model on Test Set")
print(Dacc)
print("Accuracy from Best Model on Train Set")
print(Daccc)



# crssvald_lda(DTrain,len(DTrain),Trainl)


# In[25]:


# c=[0.8865979381443299,
# 0.8541666666666666,
# 0.7916666666666666,
# 0.7916666666666666,
# 0.8125]
# c=np.array(c)
# d=np.std(c)
# print(d)


# In[ ]:




