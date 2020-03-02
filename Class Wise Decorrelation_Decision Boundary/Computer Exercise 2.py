#!/usr/bin/env python
# coding: utf-8

# In[284]:


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


# In[267]:


def calculate_p(ckx1,ckmx1,ckcovar,ckp,ckconst):
    prob=[]
    for ii in range (0,len(ckx1)):
        n=math.sqrt(math.pi*2)*math.sqrt(ckcovar)
        m=math.exp(-0.5*((ckx1[ii]-ckmx1)*(ckx1[ii]-ckmx1))/(ckcovar))
        o=1/n             
        prob.append(ckconst*o*m*ckp)
    return prob


# In[290]:


def emperr(N,m1,m2,s1,s2,p1,p2):
    wx11=np.random.normal(m1,s1,N)
    wx21=np.random.normal(m2,s2,N)
    
    print(round(givdb(np.mean(wx11),np.mean(wx21),np.std(wx11),np.std(wx21),p1,p2),2))
    wx1=list(wx11)+ list(wx21)
#     wx1=wx11.extend(wx21)
#     wx1=np.concatenate(wx11,wx21)
#     print(len(wx1))
#     print("hello world")
    c1=calculate_p(wx1,m1,s1,p1,1)
    c2=calculate_p(wx1,m2,s2,p2,1)

    prob_t=list(abs(np.array(c1) + np.array(c2)))
    prob_a= np.divide(np.array(c1),np.array(prob_t)) 
    prob_b= np.divide(np.array(c2),np.array(prob_t))
    g_f=list((np.array(prob_a) - np.array(prob_b)))
    true_count=0
    for i in range(0,len(g_f)):
        if(i<N):
            if(g_f[i]>=0):
                true_count+=1
        else:
            if(g_f[i]<0):
                true_count+=1
#     print(true_count)
    return ((1-(true_count/(2*N))))


# In[291]:


def bhatta(mu1,mu2,cvr1,cvr2,i):
    d=(1/8)*((np.subtract(mu2,mu1)).transpose())
    e=(np.subtract(mu2,mu1))
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


# In[292]:


def givdb(ca,cb,cc,cd,cp1,cp2):
    if cp1==cp2:
        db=(ca*cc+cb*cd)/(cb+cd)
    else:
        db=math.log(2)*2
    return db
    


# In[293]:


def ter(a,b,c,d,p1,p2):
    print("Decision Boundary: db")
    db=givdb(a,b,c,d,p1,p2)
    print(db)
    print("True Error from Region 2")
    print("(0.5 + 0.5*erf(db-m2)/(math.sqrt(2)*sig2))*P(x2)")
    print("True Error from Region 1")
    print("(0.5 - 0.5*erf(db-m1)/(math.sqrt(2)*sig1))*P(x1)")
    return((0.5 + 0.5*np.math.erf(db-b)/(math.sqrt(2)*d))*p2+(0.5 - 0.5*np.math.erf(db-a)/(math.sqrt(2)*c))*p1)
   


# In[294]:


me1=-.5
me2=.5
var1=1
var2=1
pr1=0.5
pr2=0.5
print("Battacharya Bound")
bb=bhatta(me1,me2,var1,var2,0)
bbc=math.sqrt(pr1*pr2)*math.exp(-bb)
print(bbc)
std1=float(math.sqrt(var1))
std2=float(math.sqrt(var2))
vc1=ter(me1,me2,std1,std2,pr1,pr2)
print("True Error")
print(vc1)
Nu=[10,50,100,200,500,1000]
EER=[]
for i in range(0,len(Nu)):
    EER.append(emperr(Nu[i],me1,me2,var1,var2,pr1,pr2))
plt.plot(Nu,EER,label="Emperical Error")
bbl=[bbc,bbc,bbc,bbc,bbc,bbc]
bbl2=[vc1,vc1,vc1,vc1,vc1,vc1]

plt.plot(Nu,bbl,label="Bhattacharya Bound")
plt.plot(Nu,bbl2,label="True Error Rate")
plt.xlabel('Number of Points')
plt.ylabel('Error Rate')
plt.legend()
plt.show()


# In[317]:


me1=-.5
me2=.5
var1=2
var2=2
pr1=0.67
pr2=0.33
print("Battacharya Bound")
bb=bhatta(me1,me2,var1,var2,0)
bbc=math.sqrt(pr1*pr2)*math.exp(-bb)
print(bbc)
std1=float(math.sqrt(var1))
std2=float(math.sqrt(var2))
vc1=ter(me1,me2,std1,std2,pr1,pr2)
print("Empirical Error")
print(vc1)
Nu=[10,50,100,200,500,1000]
EER=[]
for i in range(0,len(Nu)):
    EER.append(emperr(Nu[i],me1,me2,var1,var2,pr1,pr2))
plt.plot(Nu,EER,label="Emperical Error")
bbl=[bbc,bbc,bbc,bbc,bbc,bbc]
bbl2=[vc1,vc1,vc1,vc1,vc1,vc1]

plt.plot(Nu,bbl,label="Bhattacharya Bound")
plt.plot(Nu,bbl2,label="True Error Rate")
plt.xlabel('Number of Points')
plt.ylabel('Error Rate')
plt.legend()
plt.show()


# In[330]:


me1=-.5
me2=.5
var1=2
var2=2
pr1=0.5
pr2=0.5
print("Battacharya Bound")
bb=bhatta(me1,me2,var1,var2,0)
bbc=math.sqrt(pr1*pr2)*math.exp(-bb)
print(bbc)
std1=float(math.sqrt(var1))
std2=float(math.sqrt(var2))
vc1=ter(me1,me2,std1,std2,pr1,pr2)
print("Empirical Error")
print(vc1)
Nu=[10,50,100,200,500,1000]
EER=[]
for i in range(0,len(Nu)):
    EER.append(emperr(Nu[i],me1,me2,var1,var2,pr1,pr2))

bbl=[bbc,bbc,bbc,bbc,bbc,bbc]
bbl2=[vc1,vc1,vc1,vc1,vc1,vc1]

plt.plot(Nu,EER,label="Emperical Error")
plt.plot(Nu,bbl,label="Bhattacharya Bound")
plt.plot(Nu,bbl2,label="True Error Rate")
plt.xlabel('Number of Points')
plt.ylabel('Error Rate')
plt.legend(loc="lower right")
plt.show()


# In[332]:


me1=-.5
me2=.5
var1=3
var2=1
pr1=0.5
pr2=0.5
print("Battacharya Bound")
bb=bhatta(me1,me2,var1,var2,0)
bbc=math.sqrt(pr1*pr2)*math.exp(-bb)
print(bbc)
std1=float(math.sqrt(var1))
std2=float(math.sqrt(var2))
vc1=ter(me1,me2,std1,std2,pr1,pr2)
print("Empirical Error")
print(vc1)
Nu=[10,50,100,200,500,1000]
EER=[]
for i in range(0,len(Nu)):
    EER.append(emperr(Nu[i],me1,me2,var1,var2,pr1,pr2))
bbl=[bbc,bbc,bbc,bbc,bbc,bbc]
bbl2=[vc1,vc1,vc1,vc1,vc1,vc1]

plt.plot(Nu,EER,label="Emperical Error")
plt.plot(Nu,bbl,label="Bhattacharya Bound")
plt.plot(Nu,bbl2,label="True Error Rate")
plt.xlabel('Number of Points')
plt.ylabel('Error Rate')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




