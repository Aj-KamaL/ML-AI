#!/usr/bin/env python
# coding: utf-8

# In[25]:


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


# In[26]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


# In[27]:


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
    for i in range(0,len(BTrain)):
        if(BTrainl[i]==0):
            BClass1.append(BTrain[i])
        elif(BTrainl[i]==1):
            BClass2.append(BTrain[i])
        elif(BTrainl[i]==2):
            BClass3.append(BTrain[i])
        elif(BTrainl[i]==3):
            BClass4.append(BTrain[i])
        elif(BTrainl[i]==4):
            BClass5.append(BTrain[i])
        elif(BTrainl[i]==5):
            BClass6.append(BTrain[i])
        elif(BTrainl[i]==6):
            BClass7.append(BTrain[i])
        elif(BTrainl[i]==7):
            BClass8.append(BTrain[i])
        elif(BTrainl[i]==8):
            BClass9.append(BTrain[i])
        elif(BTrainl[i]==9):
            BClass10.append(BTrain[i])

            
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
    Btl=[Bx1,Bx2,Bx3,Bx4,Bx5,Bx6,Bx7,Bx8,Bx9,Bx10]

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

    BSW=Bc1+Bc2+Bc3+Bc4+Bc5+Bc6+Bc7+Bc8+Bc9+Bc10

    Bm=[] 
    for i in range(0,10):
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

    for i in range(0,10):
        Btmdff=Bm[i]-Bum
        Btmdff=Btmdff.reshape(len(BTrain[0]),1)
        Btmdffb=Btmdff.reshape(1,len(BTrain[0]))
        Btmpm=np.dot(Btmdff,Btmdffb)
        BSB=BSB+(Btmpm*len(Btl[i]))

    BSS=np.dot(np.linalg.inv(BSW),BSB)
    return BSS


# In[28]:


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


# In[29]:


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


# In[30]:


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
        


# In[31]:


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
                if arr2[k]==i:
                    if labels[j][k]==i:
                        T_TP+=1
                    else:
                        F_NP+=1
                elif arr2[k]!=i:
                    if labels[j][k]==-100:
                        T_NP+=1
                    else:
                        F_FP+=1  
            if (T_TP+F_NP==0) or (T_NP+F_FP==0):
                TPP.append(0)
                FPP.append(1)
                acc.append(TPP)
            elif (T_TP+F_NP!=0) and (T_NP+F_FP!=0):
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


# In[32]:


def make_Con(arr1,arr2):
    NoofC=len(np.unique(arr2))
    con_matrix=np.zeros((NoofC,NoofC))
    for i in range(0,len(arr1)):
        con_matrix[int(arr1[i])][int(arr2[i])]+=1
        
    df_cm = pd.DataFrame(con_matrix, index = [i for i in "ABCDEFGHIJ"],columns = [i for i in "ABCDEFGHIJ"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
#     plt.matshow(con_matrix)


# In[33]:


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


# In[34]:


diction = []
cross_section=1024
for i in range(0, 5):
    addr = 'cifar-10-batches-py/data_batch_' + str(i+1)
    diction.append(unpickle(addr))
TRBI = []
TRLI = []
TEBI = []
TELI = []

for i in range(0,5):
    for j in range(0,len(diction[i]['labels'])):
        red = diction[i]['data'][j][0:cross_section].reshape(32, 32)
        green = diction[i]['data'][j][cross_section:cross_section*2].reshape(32, 32)
        blue = diction[i]['data'][j][cross_section*2:3*cross_section].reshape(32, 32)
        newimage = np.zeros([32, 32, 3],dtype=np.uint8)
        
        newimage[:,:,2] = red
        newimage[:,:,1] = green
        newimage[:,:,0] = blue
        
        gray = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(cross_section)
        TRBI.append(gray)
        TRLI.append(diction[i]['labels'][j])  
        
        
diction = unpickle('cifar-10-batches-py/test_batch')
for j in range(0,len(diction['labels'])):
    red = diction['data'][j][0:cross_section].reshape(32, 32)
    green = diction['data'][j][cross_section:2*cross_section].reshape(32, 32)
    blue = diction['data'][j][2*cross_section:3*cross_section].reshape(32, 32)
    newimage = np.zeros([32, 32, 3],dtype=np.uint8)
    
    newimage[:,:,2] = red
    newimage[:,:,1] = green
    newimage[:,:,0] = blue


    gray = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY)
    gray = gray.reshape(cross_section)
    TEBI.append(gray)
    TELI .append(diction['labels'][j])

    
images = TRBI + TEBI
label=TRLI+TELI 

Train = np.array(TRBI)
Trainl= np.array(TRLI)
Test = np.array(TEBI)
Testl = np.array(TELI )
NoofClasses=len(np.unique(Trainl))
label=np.array(label)
print(np.unique(label))


# In[35]:


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


# In[36]:


cvr=np.cov(Train.T)
np.reshape(Train,(len(Train),len(Train[0])))
np.reshape(Test,(len(Test),len(Train[0])))
# print(train.shape,test.shape)


# In[37]:


PTrain,PTest,Pidx,Pprojection=pca(cvr,Train,Test,0.95)
PTest=PTest.real
PTrain=PTrain.real
print(Pprojection.shape)
np.reshape(PTrain,(len(PTrain),Pidx))
np.reshape(PTest,(len(PTest),Pidx))


# In[38]:


# plt.subplots()

# for i in range(30):
#     fig = plt.subplot(6,5, i+1)
#     plt.imshow(Pprojection[:, i].reshape((32,32)),  cmap= plt.get_cmap('gray'))
# plt.show()



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


# In[39]:


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


# In[40]:


print(Pprobs.shape)
ROC_Con(Pprobs,Testl)


# In[41]:


Ptemp_ss=[]
for i in range(len(Pss)):
    Ptemp_ss.append(Pss[i][0])

make_Con(Ptemp_ss,Testl)


# In[42]:


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


# In[43]:


Mass=PM(Train,Trainl)
DTrain,DTest,Didx=lda(Mass,Train,Test,NoofClasses-1)
print(Didx)

DTrain=DTrain.real
# Strain=Strain.real
# train=np.round(train)
DTest=DTest.real
# Stest=Stest.real
# test=np.round(test)
DTrain=np.reshape(DTrain,(len(Trainl),Didx))
DTest=np.reshape(DTest,(len(Testl),Didx))
# print(DTrain)


# In[44]:


plt.scatter(DTrain[:,0], DTrain[:,1], c = Trainl)
plt.show()
plt.scatter(DTest[:,0], DTest[:,1], c = Testl)
plt.show()


# In[45]:


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


# In[46]:


# print(Dprobs.shape)
ROC_Con(Dprobs,Testl)


# In[47]:


Dtemp_ss=[]
for i in range(len(Dss)):
    Dtemp_ss.append(Dss[i][0])

make_Con(Dtemp_ss,Testl)


# In[48]:


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


# In[49]:


# out1=[1,
# 2,
# -3,
# 4,
# 5
# ]
# out1=np.array(out1)
# out2=np.tanh(out1)
# print(out2)
# # c=np.array(c)
# # eee=np.mean(c)
# # d=np.std(c)
# # print(d,eee)


# In[50]:


xcv=[0.3544,
0.3415,
0.3569,
0.3428,
0.3468]
print(np.mean(xcv))
print(np.std(xcv))


# In[ ]:





# In[ ]:




