#!/usr/bin/env python
# coding: utf-8

# In[173]:


import numpy as np
import sys
import os
import gzip
import math
import matplotlib.pyplot as plt
import random
from copy import copy, deepcopy
import csv
import pandas as pd
import seaborn as sn


# In[174]:


roc_train=[]
roc_test=[]
roc_miss=[]


# In[175]:


# ROC, Accuracy, Confusion Matrix
def model_mes(ckx1,ckx2,ckmx1,ckmx2,ckmx3,ckmx4,ckcovar1,ckcovar2,ia_test,ckp1,ckp2):
    threshold=[0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
#     Calculate probabilty (like*prior)
    prob_aa=calculate_p(ckx1,ckx2,ckmx1,ckmx2,ckcovar1,ckp1,1)
    prob_bb=calculate_p(ckx1,ckx2,ckmx3,ckmx4,ckcovar2,ckp2,1)
#     Normalize (divide by evidence)
    prob_t=list(abs(np.array(prob_aa) + np.array(prob_bb)))
    prob_a= np.divide(np.array(prob_aa),np.array(prob_t)) 
    prob_b= np.divide(np.array(prob_bb),np.array(prob_t))
#     count=0
#     for i in range(0,len(prob_a)):
#         if(prob_a[i]>=prob_b[i]):
#             if ia_test[i]==0:
#                 count+=1
#         elif(prob_a[i]<=prob_b[i]):
#             if ia_test[i]==1:
#                 count+=1
#     print("Accuracy")
#     print(count/350*100)
    t_label=[]    
    for i in range (0,len(threshold)):
        t_label2=[]
        for j in range (0,len(prob_a)):
            if prob_a[j]>=threshold[i]:
                t_label2.append(0)
            else:
                t_label2.append(1)
        t_label.append(t_label2)
    TPP=[]
    FPP=[]
    FNN=[]
    TNN=[]
    con_matr=[]
    for j in range (0,len(t_label)):
        T_TP=0
        F_FP=0
        T_NP=0
        F_NP=0
        for k in range(0,len(ia_test)):
            if ia_test[k]== 0:
                if t_label[j][k]==0:
                    T_TP+=1
                else:
                    F_NP+=1
            elif ia_test[k]== 1:
                if t_label[j][k]==1:
                    T_NP+=1
                else:
                    F_FP+=1
        if(j==8):       
            c_matrix=np.zeros((2,2))
            c_matrix[0][0]=T_TP
            c_matrix[0][1]=F_FP
            c_matrix[1][0]=F_NP
            c_matrix[1][1]=T_NP
            print((T_TP+T_NP)/(T_TP+F_NP+T_NP+F_FP))
            con_matr=c_matrix
        TPP.append(T_TP/(T_TP+F_NP))
        FPP.append(F_FP/(F_FP+T_NP))
        FNN.append(F_NP/(F_NP+T_TP))
        TNN.append(T_NP/(T_NP+F_FP))
    return TPP,FPP,con_matr    


# In[176]:


# calculate gaussian probility
def calculate_p(ckx1,ckx2,ckmx1,ckmx2,ckcovar,ckp,ckconst):
    prob=[]
    mean=[ckmx1,ckmx2]
    mean=np.array(mean)
    for i in range(len(ckx1)):
        x=[ckx1[i],ckx2[i]]
        x=np.array(x)
        prob.append(ckconst*ckp*np.exp(-0.5*(np.matmul((x-mean).T,np.matmul(np.linalg.inv(ckcovar),(x-mean)))))/np.sqrt(np.linalg.det(ckcovar)))
#     for ii in range (0,len(ckx1)):
#         a=ckx1[ii]
#         b=ckx2[ii]
#         c=a-ckmx1
#         d=b-ckmx2
#         e=np.zeros((1,2))
#         e[0][0]=c
#         e[0][1]=d
#         f=e.transpose()
#         g=np.zeros((2,2))
#         g[1][1]=ckcovar[0][0]
#         g[0][0]=ckcovar[1][1]
#         g[0][1]=-ckcovar[0][1]
#         g[1][0]=-ckcovar[1][0]
#         # print(g)
#         h=np.zeros((1,2))
#         h=np.dot(e,g)
        
# #         for i in range(len(e)): 
# #             for j in range(len(g[0])): 
# #                 for k in range(len(g)): 
# #                     h[i][j] += e[i][k] * g[k][j]
#         # print(h)
#         # print(f)
#         l=np.zeros((1,1))
#         l=np.dot(h,f)
# #         for i in range(len(h)):
# #             for j in range(len(f[0])):
# #                 for k in range(len(f)):
# #                     l[i][j] += h[i][k] * f[k][j]
#         # print(l)
#         m=math.exp(-0.5*l[0][0])
#         # print(m)
#         n=math.pi*2*math.sqrt((ckcovar[0][0]*ckcovar[1][1])-(ckcovar[0][1]*ckcovar[1][0]))
#         o=1/n
#         prob.append(ckconst*o*m*ckp)
    return prob


# In[177]:


def decboun(ckx1,ckx2,ckmx1,ckmx2,ckcovar,ckg1,ckp):
    for ii in range (0,len(ckx1)):
        a=ckx1[ii]
        b=ckx2[ii]
        c=a-ckmx1
        d=b-ckmx2
        e=np.zeros((1,2))
        e[0][0]=c
        e[0][1]=d
        f=e.transpose()
        g=np.zeros((2,2))
        g[1][1]=ckcovar[0][0]
        g[0][0]=ckcovar[1][1]
        g[0][1]=-ckcovar[0][1]
        g[1][0]=-ckcovar[1][0]
        # print(g)
        h=np.zeros((1,2))
        for i in range(len(e)): 
            for j in range(len(g[0])): 
                for k in range(len(g)): 
                    h[i][j] += e[i][k] * g[k][j]
        # print(h)
        # print(f)
        l=np.zeros((1,1))
        for i in range(len(h)):
            for j in range(len(f[0])):
                for k in range(len(f)):
                    l[i][j] += h[i][k] * f[k][j]
        # print(l)
        m=-0.5*l[0][0]
        n=-0.5*math.log(math.sqrt(ckcovar[0][0]*ckcovar[1][1]-ckcovar[0][1]*ckcovar[1][0]))
        o=math.log(ckp)
        q=m+n+o
        ckg1.append(q)
    return ckg1


# In[178]:


def risk_plot(cklmb11,cklmb12,cklmb21,cklmb22,ckdiff,ckindx):
#      Calculating and storing the probabilities of the test dtaset
    gg1=calculate_p_risk(x1,x2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb11,cklmb12)
    gg2=calculate_p_risk(x1,x2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb21,cklmb22)
    
#     Calculating and storing the probabilities of the test dtaset
    gg3=calculate_p_risk(y1,y2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb11,cklmb12)
    gg4=calculate_p_risk(y1,y2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb21,cklmb22)
    
#   Find the diff on training set    
    gg_f=list(abs(np.array(gg1) - np.array(gg2)))

#   Risk Decision Boundary 
    ppl1=[]
    ppl2=[]
    for i in range(0,len(gg_f)):
        if gg_f[i]<ckdiff:
            ppl1.append(x1[i])
            ppl2.append(x2[i])
            
    extnd_x = np.arange(-10, 10, 0.05)
    extnd_y = np.arange(-10, 10, 0.05)
    exx, eyy = np.meshgrid(extnd_x,extnd_y)
    flst1=[]
    flst2=[]
    flst1=calculate_p_risk(exx.ravel(), eyy.ravel(),mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb11,cklmb12)
    flst2=calculate_p_risk(exx.ravel(), eyy.ravel(),mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb21,cklmb22)
    diffe=list(np.array(flst1) - np.array(flst2))
    diffz=[]
    for i in range(len(diffe)):
        if diffe[i]<=0:
            diffz.append(1)
        else:
            diffz.append(0)
    diffz=np.array(diffz)
    diffz=diffz.reshape(exx.shape)  
    
#   Plot Risk Decision Boundary on Training Dataset
    plt.contourf(exx, eyy,diffz,cmap=plt.cm.bone)
    plt.scatter(X10,X11)
    plt.scatter(X20,X21)

    plt.show()

    pyax,pxax,pcax=model_mes_risk(gg1,gg2,label,ckindx)
    df_cm = pd.DataFrame(pcax, index = [i for i in "AB"],columns = [i for i in "AB"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
#     plt.plot(pxax,pyax,'k')
    roc_train.append(pxax)
    roc_train.append(pyax)
    plt.show()
    
    
    
    
    
    
#     Plot Risk Decision Boundary on Test Dataset
    plt.contourf(exx, eyy,diffz,cmap=plt.cm.bone)
    plt.scatter(Y10,Y11)
    plt.scatter(Y20,Y21)
    plt.show()

    pyax,pxax,pcax=model_mes_risk(gg3,gg4,label2,ckindx)
    df_cm = pd.DataFrame(pcax, index = [i for i in "AB"],columns = [i for i in "AB"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
#     plt.plot(pxax,pyax,'k')
    roc_test.append(pxax)
    roc_test.append(pyax)
    plt.show()


# In[179]:


# ROC, Accuracy, Confusion Matrix including risk
def model_mes_risk(prob_aa,prob_bb,ia_test,indx):
    threshold=[0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
    prob_t=list(abs(np.array(prob_aa) + np.array(prob_bb)))
    prob_a= np.divide(np.array(prob_aa),np.array(prob_t)) 
    prob_b= np.divide(np.array(prob_bb),np.array(prob_t))
    t_label=[]    
    for i in range (0,len(threshold)):
        t_label2=[]
        for j in range (0,len(prob_a)):
            if prob_a[j]<=threshold[i]:
                t_label2.append(0)
            else:
                t_label2.append(1)
        t_label.append(t_label2)
    TPP=[]
    FPP=[]
    FNN=[]
    TNN=[]
    con_matr=[]
    for j in range (0,len(t_label)):
        T_TP=0
        F_FP=0
        T_NP=0
        F_NP=0
        for k in range(0,len(ia_test)):
            if ia_test[k]== 0:
                if t_label[j][k]==0:
                    T_TP+=1
                else:
                    F_NP+=1
            elif ia_test[k]== 1:
                if t_label[j][k]==1:
                    T_NP+=1
                else:
                    F_FP+=1
        if(j==indx):       
            c_matrix=np.zeros((2,2))
            c_matrix[0][0]=T_TP
            c_matrix[0][1]=F_FP
            c_matrix[1][0]=F_NP
            c_matrix[1][1]=T_NP
            print((T_TP+T_NP)/(T_TP+F_NP+T_NP+F_FP))
            con_matr=c_matrix
        TPP.append(T_TP/(T_TP+F_NP))
        FPP.append(F_FP/(F_FP+T_NP))
        FNN.append(F_NP/(F_NP+T_TP))
        TNN.append(T_NP/(T_NP+F_FP))
    return TPP,FPP,con_matr    


# In[180]:


# calculate gaussian prob with risk
def calculate_p_risk(ckx1,ckx2,ckmx1,ckmx2,ckmx3,ckmx4,ckcovar1,ckcovar2,ckp1,ckp2,cklm1,cklm2):
    ta=calculate_p(ckx1,ckx2,ckmx1,ckmx2,ckcovar1,ckp1,cklm1)
    tb=calculate_p(ckx1,ckx2,ckmx3,ckmx4,ckcovar2,ckp2,cklm2)
    tc=list(abs(np.array(ta) + np.array(tb)))
    return tc


# In[181]:


xo1=[]
xo2=[]
label=[]
crs = open("train.txt", "r")
for columns in ( raw.strip().split() for raw in crs ):
    xo1.append(float(columns[0].split(',')[0]))
    xo2.append(float(columns[0].split(',')[1]))
    label.append(float(columns[0].split(',')[2]))
yo1=[]
yo2=[]
label2=[]
crs2 = open("test_all.txt", "r")
for columns in ( raw.strip().split() for raw in crs2 ):
    yo1.append(float(columns[0].split(',')[0]))
    yo2.append(float(columns[0].split(',')[1]))
    label2.append(float(columns[0].split(',')[2]))


# In[182]:


dX10=[]
dX11=[]
dX20=[]
dX21=[]

for i in range(0,len(label)):
    if label[i]==0:
        dX10.append(xo1[i])
        dX11.append(xo2[i])
    else:
        dX20.append(xo1[i])
        dX21.append(xo2[i])

# plt.plot(X10,X11)
# plt.plot(X20,X21)

dY10=[]
dY11=[]
dY20=[]
dY21=[]

for i in range(0,len(label2)):
    if label2[i]==0:
        dY10.append(yo1[i])
        dY11.append(yo2[i])
    else:
        dY20.append(yo1[i])
        dY21.append(yo2[i])


# In[183]:


mox11=np.mean(np.array(dX10))
mox12=np.mean(np.array(dX11))
ocovar1=np.cov(np.array(dX10),np.array(dX11))
# print(ocovar1)
ocorr1=ocovar1[0][1]/(math.sqrt(ocovar1[0][0])*math.sqrt(ocovar1[1][1]))
# print(ocorr1)


# In[184]:


mox21=np.mean(np.array(dX20))
mox22=np.mean(np.array(dX21))
ocovar2=np.cov(np.array(dX20),np.array(dX21))
# print(ocovar2)
ocorr2=ocovar2[0][1]/(math.sqrt(ocovar2[0][0])*math.sqrt(ocovar2[1][1]))
# print(ocorr2)


# In[185]:


d, V = np.linalg.eig(ocovar1)
D = np.diag(1. / np.sqrt(d+1E-18))
W1 = (np.dot(V, D))
# print(W1)
# X_white = np.dot(X, W)


# In[186]:


d, V = np.linalg.eig(ocovar2)
D = np.diag(1. / np.sqrt(d+1E-18))
W2 = (np.dot(V, D))
# print(W2)


# In[187]:


# x10=[]
# x11=[]
# x20=[]
# x21=[]
# y10=[]
# y11=[]
# y20=[]
# y21=[]
# for i in range (0,len(dX10)):
#     x10.append(W1[0][0]*dX10[i]+W1[0][1]*dX11[i])
#     x11.append(W1[1][0]*dX10[i]+W1[1][1]*dX11[i])
# for i in range (0,len(dY10)):
#     y10.append(W1[0][0]*dY10[i]+W1[0][1]*dY11[i])
#     y11.append(W1[1][0]*dY10[i]+W1[1][1]*dY11[i])
    
# for i in range (0,len(dX20)):
#     x20.append(W2[0][0]*dX20[i]+W2[0][1]*dX21[i])
#     x21.append(W2[1][0]*dX20[i]+W2[1][1]*dX21[i])
# for i in range (0,len(dY20)):
#     y20.append(W2[0][0]*dY20[i]+W2[0][1]*dY21[i])
#     y21.append(W2[1][0]*dY20[i]+W2[1][1]*dY21[i])


# In[188]:


x_train_cl1 = np.array([dX10, dX11])
final_temp1 = np.matmul(x_train_cl1.T, W1)
x_train_cl2 = np.array([dX20, dX21])
final_temp2 = np.matmul(x_train_cl2.T, W2)
y_train_cl1 = np.array([dY10, dY11])
final_temp3 = np.matmul(y_train_cl1.T, W1)
y_train_cl2 = np.array([dY20, dY21])
final_temp4 = np.matmul(y_train_cl2.T, W2)


# In[189]:


x1=list(final_temp1[:,0])+list(final_temp2[:,0])
# print(x1)
# print(len(x1))
x2=list(final_temp1[:,1])+list(final_temp2[:,1])
# print(list(final_temp2[:,1]))
y1=list(final_temp3[:,0])+list(final_temp4[:,0])
y2=list(final_temp3[:,1])+list(final_temp4[:,1])
tl1=[0]
tl2=[1]
tl1=tl1*150
tl2=tl2*200
label=tl1+tl2
tt1=[0]
tt2=[1]
tt1=tt1*45
tt2=tt2*35
label2=tt1+tt2


# In[190]:


mx1=np.mean(np.array(x1))
mx2=np.mean(np.array(x2))
# print(mx1)
# print(mx2)
covar=np.cov(np.array(x1),np.array(x2))
# print(covar)
corr=covar[0][1]/(math.sqrt(covar[0][0])*math.sqrt(covar[1][1]))
# print(corr)


# In[191]:


#Separate data point class wise
X10=[]
X11=[]
X20=[]
X21=[]

for i in range(0,len(label)):
    if label[i]==0:
        X10.append(x1[i])
        X11.append(x2[i])
    else:
        X20.append(x1[i])
        X21.append(x2[i])

# plt.plot(X10,X11)
# plt.plot(X20,X21)
Y10=[]
Y11=[]
Y20=[]
Y21=[]

for i in range(0,len(label2)):
    if label2[i]==0:
        Y10.append(y1[i])
        Y11.append(y2[i])
    else:
        Y20.append(y1[i])
        Y21.append(y2[i])


# In[192]:


# Class Mean and Covariance
mX10=np.mean(np.array(X10))
mX11=np.mean(np.array(X11))
covar1=np.cov(np.array(X10),np.array(X11))
print(covar1)
mX20=np.mean(np.array(X20))
mX21=np.mean(np.array(X21))
covar2=np.cov(np.array(X20),np.array(X21))
print(covar2)


# In[209]:


plt.scatter(X10,X11,label="Class 0")
plt.scatter(X20,X21,label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
plt.scatter(Y10,Y11,label="Class 0")
plt.scatter(Y20,Y21,label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# In[193]:


# g1=[]
# g2=[]
# decboun(x1,x2,mX10,mX11,covar1,g1,0.43)
# decboun(x1,x2,mX20,mX21,covar2,g2,0.57)
# # print(g1)
# # print(g2)
# g_f=list(abs(np.array(g1) - np.array(g2)))
# # print(g_f)
# # print(len(g_f))
# pl1=[]
# pl2=[]
# for i in range(0,len(g_f)):
#     if g_f[i]<=0.06:
#         pl1.append(x1[i])
#         pl2.append(x2[i])
# extnd_x = np.arange(-10, 10, 0.05)
# extnd_y = np.arange(-10, 10, 0.05)
# exx, eyy = np.meshgrid(extnd_x,extnd_y)
# flst1=[]
# flst2=[]
# decboun(exx.ravel(), eyy.ravel(),mX10,mX11,covar1,flst1,0.43)
# decboun(exx.ravel(), eyy.ravel(),mX20,mX21,covar2,flst2,0.57)
# diffe=list(np.array(flst1) - np.array(flst2))
# diffz=[]
# for i in range(len(diffe)):
#     if diffe[i]<=0:
#         diffz.append(1)
#     else:
#         diffz.append(0)
# diffz=np.array(diffz)
# diffz=diffz.reshape(exx.shape)

# #  Plot Decision Boundary on Training Dataset
# plt.contourf(exx, eyy,diffz)
# plt.scatter(X10,X11)
# plt.scatter(X20,X21)
# # plt.plot(pl1,pl2,'k')
# plt.show()

# yax,xax,cax=model_mes(x1,x2,mX10,mX11,mX20,mX21,covar1,covar2,label,0.43,0.57)
# df_cm = pd.DataFrame(cax, index = [i for i in "AB"],columns = [i for i in "AB"])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()
# plt.plot(xax,yax,'k')
# plt.show()

# # Plot Decision Boundary on Test Dataset
# plt.contourf(exx, eyy,diffz)
# plt.scatter(Y10,Y11)
# plt.scatter(Y20,Y21)
# # plt.plot(pl1,pl2,'k')
# plt.show()

# yax,xax,cax=model_mes(y1,y2,mX10,mX11,mX20,mX21,covar1,covar2,label2,0.43,0.57)

# df_cm = pd.DataFrame(cax, index = [i for i in "AB"],columns = [i for i in "AB"])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()
# plt.plot(xax,yax,'k')
# plt.show()


# In[194]:


risk_plot(0,1,1,0,0.0009,8)


# In[195]:


risk_plot(1,0,0,1,0.0009,8)


# In[196]:


risk_plot(1000,1000,0,0,10,8)


# In[197]:


risk_plot(0,0,1000,1000,10,8)


# In[198]:


risk_plot(0,100,150,0,0.07,8)


# In[199]:


risk_plot(0,150,100,0,0.32,8)


# In[200]:


plt.figure()
indx=1
for i in range(0,len(roc_train)-1,2):
    plt.plot(roc_train[i],roc_train[i+1],label="risk"+str(indx))
    plt.legend(loc="center")
    indx=indx+1


# In[201]:


plt.figure()
indx=1
for i in range(0,len(roc_test)-1,2):
    plt.plot(roc_test[i],roc_test[i+1],label="risk"+str(indx))
    indx=indx+1
    plt.legend(loc="center")


# In[ ]:




