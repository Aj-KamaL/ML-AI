#!/usr/bin/env python
# coding: utf-8

# In[263]:


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


# In[264]:


roc_train=[]
roc_test=[]
roc_miss=[]


# In[265]:


# ROC, Accuracy, Confusion Matrix
def model_mes(ckx1,ckx2,ckmx1,ckmx2,ckmx3,ckmx4,ckcovar1,ckcovar2,ia_test,ckp1,ckp2):
    threshold=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
#     Calculate probabilty (like*prior)
    prob_aa=calculate_p(ckx1,ckx2,ckmx1,ckmx2,ckcovar1,ckp1,1)
    prob_bb=calculate_p(ckx1,ckx2,ckmx3,ckmx4,ckcovar2,ckp2,1)
#     Normalize (divide by evidence)
    prob_t=list(abs(np.array(prob_aa) + np.array(prob_bb)))
    prob_a= np.divide(np.array(prob_aa),np.array(prob_t)) 
    prob_b= np.divide(np.array(prob_bb),np.array(prob_t))
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


# In[266]:


# ROC, Accuracy, Confusion Matrix including risk
def model_mes_risk(prob_aa,prob_bb,ia_test,indx):
    threshold=[0, 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95, 1]
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


# In[267]:


# calculate gaussian probility
def calculate_p(ckx1,ckx2,ckmx1,ckmx2,ckcovar,ckp,ckconst):
    prob=[]
#     mean=[ckmx1,ckmx2]
#     mean=np.array(mean)
#     for i in range(len(ckx1)):
#         x=[ckx1[i],ckx2[i]]
#         x=np.array(x)
#         prob.append(ckp*np.exp(-0.5*(np.matmul((x-mean).T,np.matmul(np.linalg.inv(ckcovar),(x-mean)))))/np.sqrt(np.linalg.det(ckcovar)))
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
        h=np.dot(e,g)
        
#         for i in range(len(e)): 
#             for j in range(len(g[0])): 
#                 for k in range(len(g)): 
#                     h[i][j] += e[i][k] * g[k][j]
        # print(h)
        # print(f)
        l=np.zeros((1,1))
        l=np.dot(h,f)
#         for i in range(len(h)):
#             for j in range(len(f[0])):
#                 for k in range(len(f)):
#                     l[i][j] += h[i][k] * f[k][j]
        # print(l)
        m=math.exp(-0.5*l[0][0])
        # print(m)
        n=math.pi*2*math.sqrt((ckcovar[0][0]*ckcovar[1][1])-(ckcovar[0][1]*ckcovar[1][0]))
        o=1/n
        prob.append(ckconst*o*m*ckp)
    return prob


# In[268]:


# calculate gaussian prob with risk
def calculate_p_risk(ckx1,ckx2,ckmx1,ckmx2,ckmx3,ckmx4,ckcovar1,ckcovar2,ckp1,ckp2,cklm1,cklm2):
    ta=calculate_p(ckx1,ckx2,ckmx1,ckmx2,ckcovar1,ckp1,cklm1)
    tb=calculate_p(ckx1,ckx2,ckmx3,ckmx4,ckcovar2,ckp2,cklm2)
    tc=list(abs(np.array(ta) + np.array(tb)))
    return tc


# In[269]:


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


# In[270]:



def risk_plot(cklmb11,cklmb12,cklmb21,cklmb22,ckdiff,ckindx):
#      Calculating and storing the probabilities of the test dtaset
    gg1=calculate_p_risk(x1,x2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb11,cklmb12)
    gg2=calculate_p_risk(x1,x2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb21,cklmb22)
    
#     Calculating and storing the probabilities of the test dtaset
    gg3=calculate_p_risk(y1,y2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb11,cklmb12)
    gg4=calculate_p_risk(y1,y2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb21,cklmb22)
    
    gg5=calculate_p_missing_risk(z1,z2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb11,cklmb12)
    gg6=calculate_p_missing_risk(z1,z2,mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb21,cklmb22)
    
#   Find the diff on training set    
    gg_f=list(abs(np.array(gg1) - np.array(gg2)))

#   Risk Decision Boundary 
    ppl1=[]
    ppl2=[]
    for i in range(0,len(gg_f)):
        if gg_f[i]<ckdiff:
            ppl1.append(x1[i])
            ppl2.append(x2[i])
            
    extnd_x = np.arange(-3, 12, 0.1)
    extnd_y = np.arange(-3, 12, 0.1)
    exx, eyy = np.meshgrid(extnd_x,extnd_y)
    flst1=[]
    flst2=[]
    flst1=calculate_p_risk(exx.ravel(), eyy.ravel(),mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb11,cklmb12)
    flst2=calculate_p_risk(exx.ravel(), eyy.ravel(),mX10,mX11,mX20,mX21,covar1,covar2,0.43,0.57,cklmb21,cklmb22)
    
    diffe=list(np.array(flst1) - np.array(flst2))
    diffz=[]
    for i in range(len(diffe)):
        if diffe[i]<=0:
            diffz.append(0)
        else:
            diffz.append(1)
    diffz=np.array(diffz)
    diffz=diffz.reshape(exx.shape)       
            
            
            
#   Plot Risk Decision Boundary on Training Dataset
    lbael_clas=[0,1]
    plt.contourf(exx, eyy,diffz,cmap=plt.cm.bone)
#     CS = plt.contour(CS2, cmap=plt.cm.YlGn_r)
#     plt.clabel(CS, fontsize=10, colors=plt.cm.Reds(CS.norm(CS.levels)))
    plt.scatter(X10,X11)
    plt.scatter(X20,X21)
#     plt.plot(ppl1,ppl2,'k')
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
#     plt.plot(ppl1,ppl2,'k')
    # print(len(pl1))
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
#     Plot Risk Decision Boundary on Missing Test Dataset
    plt.contourf(exx, eyy,diffz,cmap=plt.cm.bone)
    plt.scatter(Z10,Z11)
    plt.scatter(Z20,Z21)
#     plt.plot(ppl1,ppl2,'k')
    plt.show()
    
    pyax,pxax,pcax=model_mes_risk_missing(gg5,gg6,label3,ckindx)
    df_cm = pd.DataFrame(pcax, index = [i for i in "AB"],columns = [i for i in "AB"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
#     plt.plot(pxax,pyax,'k')
    roc_miss.append(pxax)
    roc_miss.append(pyax)
    plt.show()


# In[271]:


def calculte_integration(a,b,p,q,r,s,t):
    sqrk=s
    untl=a*q+a*r-2*b*s
    zrm=b*b*s+a*a*p-a*b*q-a*b*r
    it1=math.sqrt(math.pi)
    it2=math.exp((untl*untl)/((4*sqrk)-zrm))
    it3=math.erf(((2*t*sqrk)-untl)/(2*math.sqrt(sqrk)))
    it4=math.erf(((2*t*sqrk)+untl)/(2*math.sqrt(sqrk)))
    it5=2*math.sqrt(sqrk)
    it6=it1*it2*(it3+it4)
    it7=it6/it5
    return it7


# In[272]:


# calculate gaussian probility
def calculate_p_missing(ckx1,ckx2,ckmx1,ckmx2,ckcovar,ckp,ckconst):
    prob=[]
    for ii in range (0,len(ckx1)):
        if ckx2[ii]==-1000:
            mintg=calculte_integration(ckx1[ii]-ckmx1,ckmx2,ckcovar[0][0],ckcovar[0][1],ckcovar[1][0],ckcovar[1][1],10)
            noff=math.pi*2*math.sqrt((ckcovar[0][0]*ckcovar[1][1])-(ckcovar[0][1]*ckcovar[1][0]))
            off=1/noff
            prob.append(ckconst*off*mintg*ckp)
        else:
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
            h=np.dot(e,g)
            l=np.zeros((1,1))
            l=np.dot(h,f)
            m=math.exp(-0.5*l[0][0])
            n=math.pi*2*math.sqrt((ckcovar[0][0]*ckcovar[1][1])-(ckcovar[0][1]*ckcovar[1][0]))
            o=1/n
            prob.append(ckconst*o*m*ckp)      
    return prob


# In[273]:


# ROC, Accuracy, Confusion Matrix
def model_mes_missing(ckx1,ckx2,ckmx1,ckmx2,ckmx3,ckmx4,ckcovar1,ckcovar2,ckia_test,ckp1,ckp2):
    threshold=[0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
#     Calculate probabilty (like*prior)
    prob_aa=calculate_p_missing(ckx1,ckx2,ckmx1,ckmx2,ckcovar1,ckp1,1)
    prob_aa = [x+0.5 for x in prob_aa]
    prob_bb=calculate_p_missing(ckx1,ckx2,ckmx3,ckmx4,ckcovar2,ckp2,1)
#     Normalize (divide by evidence)
    prob_t=list(abs(np.array(prob_aa) + np.array(prob_bb)))
    prob_a= np.divide(np.array(prob_aa),np.array(prob_t)) 
    prob_b= np.divide(np.array(prob_bb),np.array(prob_t))
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
        for k in range(0,len(ckia_test)):
            if ckia_test[k]== 0:
                if t_label[j][k]==0:
                    T_TP+=1
                else:
                    F_NP+=1
            elif ckia_test[k]== 1:
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


# In[274]:


def calculate_p_missing_risk(ckx1,ckx2,ckmx1,ckmx2,ckmx3,ckmx4,ckcovar1,ckcovar2,ckp1,ckp2,cklm1,cklm2):
    ta=calculate_p_missing(ckx1,ckx2,ckmx1,ckmx2,ckcovar1,ckp1,cklm1)
    tb=calculate_p_missing(ckx1,ckx2,ckmx3,ckmx4,ckcovar2,ckp2,cklm2)
    tc=list(abs(np.array(ta) + np.array(tb)))
    return tc    


# In[275]:


def model_mes_risk_missing(prob_aa,prob_bb,ia_test,indx):
    threshold=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
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


# In[276]:


x1=[]
x2=[]
label=[]
crs = open("train.txt", "r")
for columns in ( raw.strip().split() for raw in crs ):
    x1.append(float(columns[0].split(',')[0]))
    x2.append(float(columns[0].split(',')[1]))
    label.append(float(columns[0].split(',')[2]))
y1=[]
y2=[]
label2=[]
crs2 = open("test_all.txt", "r")
for columns in ( raw.strip().split() for raw in crs2 ):
    y1.append(float(columns[0].split(',')[0]))
    y2.append(float(columns[0].split(',')[1]))
    label2.append(float(columns[0].split(',')[2]))
z1=[]
z2=[]
label3=[]
crs3 = open("test_missing.txt", "r")
for columns in ( raw.strip().split() for raw in crs3 ):
    a_tp=float(columns[0].split(',')[0])
    b_tp=(columns[0].split(',')[1])
    z1.append(a_tp)
    if(b_tp=='NA'):
        z2.append(-1000)
    else:
        z2.append(float(b_tp))
    label3.append(float(columns[0].split(',')[2]))


# In[277]:


mx1=np.mean(np.array(x1))
mx2=np.mean(np.array(x2))
covar=np.cov(np.array(x1),np.array(x2))
print(covar)
corr=covar[0][1]/(math.sqrt(covar[0][0])*math.sqrt(covar[1][1]))
print(corr)


# In[278]:


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
Z10=[]
Z11=[]
Z20=[]
Z21=[]
for i in range(0,len(label3)):
    if label3[i]==0:
        if z2[i]!=-1000:
            Z10.append(z1[i])
            Z11.append(z2[i])
    else:
        if z2[i]!=-1000:
            Z20.append(z1[i])
            Z21.append(z2[i])
# plt.plot(X10,X11)
# plt.plot(X20,X21)
# plt.show()
# print(label.count(1))
# print(len(X2))
# 150
# 200


# In[279]:


# Class Mean and Covariance
mX10=np.mean(np.array(X10))
mX11=np.mean(np.array(X11))
covar1=np.cov(np.array(X10),np.array(X11))
print(covar1)
mX20=np.mean(np.array(X20))
mX21=np.mean(np.array(X21))
covar2=np.cov(np.array(X20),np.array(X21))
print(covar2)


# In[295]:


plt.scatter(X10,X11,label="Class 0")
plt.scatter(X20,X21,label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
plt.scatter(Y10,Y11,label="Class 0")
plt.scatter(Y20,Y21,label="Class 1")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
plt.scatter(Z10,Z11,label="Class 0")
plt.scatter(Z20,Z21,label="Class 1")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# In[280]:


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
#     if g_f[i]<=0.2:
#         pl1.append(x1[i])
#         pl2.append(x2[i])

# extnd_x = np.arange(-3, 13, 0.11)
# extnd_y = np.arange(-3, 13, 0.1)
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




# plt.contourf(exx, eyy,diffz)
# plt.scatter(X10,X11)
# plt.scatter(X20,X21)
# plt.show()

# yax,xax,cax=model_mes(x1,x2,mX10,mX11,mX20,mX21,covar1,covar2,label,0.43,0.57)
# df_cm = pd.DataFrame(cax, index = [i for i in "AB"],columns = [i for i in "AB"])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()
# plt.plot(xax,yax,'k')
# # roc_train.append(xax,yax)
# plt.show()

# # Plot Decision Boundary on Test Dataset
# plt.contourf(exx, eyy,diffz)
# plt.scatter(Y10,Y11)
# plt.scatter(Y20,Y21)
# plt.show()

# yax,xax,cax=model_mes(y1,y2,mX10,mX11,mX20,mX21,covar1,covar2,label2,0.43,0.57)
# df_cm = pd.DataFrame(cax, index = [i for i in "AB"],columns = [i for i in "AB"])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()

# plt.plot(xax,yax,'k')

# plt.show()


# # Plot Decision Boundary on Missing Test Dataset
# plt.contourf(exx, eyy,diffz)
# plt.scatter(Z10,Z11)
# plt.scatter(Z20,Z21)
# plt.show()

# yyax,xxax,ccax=model_mes_missing(z1,z2,mX10,mX11,mX20,mX21,covar1,covar2,label3,0.43,0.57)
# df_cm = pd.DataFrame(ccax, index = [i for i in "AB"],columns = [i for i in "AB"])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()

# plt.plot(xxax,yyax,'k')

# plt.show()


# In[281]:


risk_plot(0,1,1,0,0.0009,8)


# In[282]:


risk_plot(1,0,0,1,0.0009,8)


# In[283]:


risk_plot(1000,1000,0,0,1.2,8)


# In[284]:


risk_plot(0,0,1000,1000,1.2,8)


# In[285]:


risk_plot(0,100,150,0,0.07,8)


# In[286]:


risk_plot(0,150,100,0,0.12389,8)


# In[287]:



# e1=[]
# e2=[]
# xx, yy = np.meshgrid(x, y)
# decboun(xx.ravel(), yy.ravel(),mX10,mX11,covar1,e1,0.43)
# decboun(xx.ravel(), yy.ravel(),mX20,mX21,covar2,e2,0.57)
# g_f=list(np.array(e1) - np.array(e2))
# zz=[]
# for i in range(len(g_f)):
#     if g_f[i]<=0:
#         zz.append(1)
#     else:
#         zz.append(0)
# zz=np.array(zz)
# zz=zz.reshape(xx.shape)
# plt.contourf(xx, yy,zz)
# plt.scatter(X10,X11)
# plt.scatter(X20,X21)


# In[288]:


plt.figure()
indx=1
for i in range(0,len(roc_train)-1,2):
    plt.plot(roc_train[i],roc_train[i+1],label="risk"+str(indx))
    plt.legend(loc="center")
    indx=indx+1


# In[289]:


plt.figure()
indx=1
for i in range(0,len(roc_test)-1,2):
    plt.plot(roc_test[i],roc_test[i+1],label="risk"+str(indx))
    indx=indx+1
    plt.legend(loc="center")


# In[290]:


plt.figure()
indx=1
for i in range(0,len(roc_miss)-1,2):
    plt.plot(roc_miss[i],roc_miss[i+1],label="risk"+str(indx))
    plt.legend(loc="center")
    indx=indx+1


# In[ ]:




