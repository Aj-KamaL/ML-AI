#!/usr/bin/env python
# coding: utf-8

# In[215]:


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


# In[216]:


def calculate_pmc(ckx1,ckx2,ckx3,ckmx1,ckmx2,ckmx3,ckcovar,ckp,ckconst):
    prob=[]
    mean=[ckmx1,ckmx2,ckmx3]
#     print(mean)
    mean=np.array(mean)
    for i in range(len(ckx1)):
        x=[ckx1[i],ckx2[i],ckx3[i]]
        x=np.array(x)
        prob.append(ckp*np.exp(-0.5*(np.matmul((x-mean).T,np.matmul(np.linalg.inv(ckcovar),(x-mean)))))/np.sqrt(np.linalg.det(ckcovar)))
    return prob


# In[217]:


def calculate_pm(ckx1,ckx2,ckmx1,ckmx2,ckcovar,ckp,ckconst):
    prob=[]
#     mean=[mx1,mx2]
#     mean=np.array(mean)
#     for i in range(len(x1)):
#         x=[x1[i],x2[i]]
#         x=np.array(x)
#         prob.append(p*np.exp(-0.5*(np.matmul((x-mean).T,np.matmul(np.linalg.inv(covar),(x-mean)))))/np.sqrt(np.linalg.det(covar)))
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


# In[218]:


def bhatta(mu1,mu2,cvr1,cvr2,i):
    d=(1/8)*((np.subtract(mu1,mu2)).transpose())
    e=(np.subtract(mu1,mu2))
    if(i==0):
        l=1/((cvr1+cvr2)*0.5)
    else:
        l=np.linalg.inv((np.add(cvr1,cvr2)*0.5))
    if(i==0):
        g=((cvr1+cvr2)*0.5)
    else:
        g=np.linalg.det(np.add(cvr1,cvr2)*0.5)
    if(i==0):
        h=math.sqrt(cvr1*cvr2)
    else:
        h=math.sqrt(np.linalg.det(cvr1)*np.linalg.det(cvr2))
    ii=g/h
    f=0.5*math.log(ii)
    if i==0:
        k=d*l
        q=k*e
    else:
        k=np.matmul(d,l)
        q=np.matmul(k,e)
    r=q+f
    return(r)
    
    
    


# In[219]:


def calculate_p(ckx1,ckmx1,ckcovar,ckp,ckconst):
    prob=[]
    for ii in range (0,len(ckx1)):
        n=math.sqrt(math.pi*2)*math.sqrt(ckcovar)
        m=math.exp(-0.5*((ckx1[ii]-ckmx1)*(ckx1[ii]-ckmx1))/(ckcovar))
        o=1/n             
        prob.append(ckconst*o*m*ckp)
    return prob


# In[220]:


w1x1=[-5.01,-5.43,1.08,0.86,-2.67,4.94,-2.51,-2.25,5.56,1.03]
w1x2=[-8.12, -3.48, -5.52, -3.78, 0.63, 3.29, 2.09, -2.13, 2.86, -3.33]
w1x3=[-3.68, -3.54, 1.66, -4.11, 7.39, 2.08, -2.59, -6.94, -2.26, 4.33]

w2x1=[-0.91, 1.30, -7.75, -5.47, 6.14, 3.60, 5.37, 7.18, -7.39, -7.50]
w2x2=[-0.18, -2.06, -4.54, 0.50, 5.72, 1.26, -4.63, 1.46, 1.17, -6.32]
w2x3=[-0.05, -3.53, -0.95, 3.92, -4.85, 4.36, -3.65, -6.66, 6.30, -0.31]

wx1=w1x1+w2x1
wx2=w1x2+w2x2
wx3=w1x3+w2x3

mw1x1=np.mean(w1x1)
mw1x2=np.mean(w1x2)
mw1x3=np.mean(w1x3)

cw1x1=np.var(w1x1)
cw1x2=np.var(w1x2)
cw1x3=np.var(w1x3)

cow0=np.cov(w1x1,w1x2)
xx = np.vstack([w1x1,w1x2,w1x3])
cow1 = np.cov(xx)
# cow1=np.cov(w1x1,w1x2,w1x3)
# print(cow1)


mw2x1=np.mean(w2x1)
mw2x2=np.mean(w2x2)
mw2x3=np.mean(w2x3)

cw2x1=np.var(w2x1)
cw2x2=np.var(w2x2)
cw2x3=np.var(w2x3)

cow2=np.cov(w2x1,w2x2)
xx = np.vstack([w2x1,w2x2,w2x3])
cow3 = np.cov(xx)
# cow3=np.cov(w2x1,w2x2,w2x3)


# In[221]:


c1=calculate_p(wx1,mw1x1,cw1x1,0.5,1)
c2=calculate_p(wx1,mw2x1,cw2x1,0.5,1)

prob_t=list(abs(np.array(c1) + np.array(c2)))
prob_a= np.divide(np.array(c1),np.array(prob_t)) 
prob_b= np.divide(np.array(c2),np.array(prob_t))
g_f=list((np.array(prob_a) - np.array(prob_b)))
true_count=0
te1=0
te2=0
for i in range(0,len(g_f)):
    if(i<10):
        if(g_f[i]>=0):
            true_count+=1
        else:
            te1+=1
    else:
        if(g_f[i]<0):
            true_count+=1
        else:
            te2+=1
# print(true_count/20)
print("Total Error")
print((1-(true_count/20)))
print("ClassWise Error")
print(te1/10)
print(te2/10)
print("Bhatt Bound")
c=math.sqrt(0.5*0.5)
d=math.exp(-1*bhatta(mw1x1,mw2x1,cw1x1,cw2x1,0))
print(c*d)


# In[222]:


c1=calculate_pm(wx1,wx2,mw1x1,mw1x2,cow0,0.5,1)
c2=calculate_pm(wx1,wx2,mw2x1,mw2x2,cow2,0.5,1)
prob_t=list(abs(np.array(c1) + np.array(c2)))
prob_a= np.divide(np.array(c1),np.array(prob_t)) 
prob_b= np.divide(np.array(c2),np.array(prob_t))
g_f=list((np.array(prob_a) - np.array(prob_b)))
true_count=0
te1=0
te2=0
for i in range(0,len(g_f)):
    if(i<10):
        if(g_f[i]>=0):
            true_count+=1
        else:
            te1+=1
    else:
        if(g_f[i]<0):
            true_count+=1
        else:
            te2+=1
# print(true_count/20)
print("Total Error")
print((1-(true_count/20)))
print("ClassWise Error")
print(te1/10)
print(te2/10)
print("Bhatt Bound")
c=math.sqrt(0.5*0.5)
mean1=[mw1x1,mw1x2]
mean2=[mw2x1,mw2x2]
d=math.exp(-1*bhatta(mean1,mean2,cow0,cow2,1))
print(c*d)


# In[223]:


c1=calculate_pmc(wx1,wx2,wx3,mw1x1,mw1x2,mw1x3,cow1,0.5,1)
c2=calculate_pmc(wx1,wx2,wx3,mw2x1,mw2x2,mw2x3,cow3,0.5,1)
prob_t=list(abs(np.array(c1) + np.array(c2)))
prob_a= np.divide(np.array(c1),np.array(prob_t)) 
prob_b= np.divide(np.array(c2),np.array(prob_t))
g_f=list((np.array(prob_a) - np.array(prob_b)))
true_count=0
te1=0
te2=0
for i in range(0,len(g_f)):
    if(i<10):
        if(g_f[i]>=0):
            true_count+=1
        else:
            te1+=1
    else:
        if(g_f[i]<0):
            true_count+=1
        else:
            te2+=1
# print(true_count/20)
print("Total Error")
print((1-(true_count/20)))
print("ClassWise Error")
print(te1/10)
print(te2/10)
print("Bhatt Bound")
c=math.sqrt(0.5*0.5)
mean1=[mw1x1,mw1x2,mw1x3]
mean2=[mw2x1,mw2x2,mw2x3]
d=math.exp(-1*bhatta(mean1,mean2,cow1,cow3,1))
print(c*d)


# In[ ]:





# In[ ]:





# In[ ]:




