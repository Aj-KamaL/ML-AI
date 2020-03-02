#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os
import gzip
import math
from copy import copy, deepcopy
import matplotlib.pyplot as plt
fmnistdata="/home/ajkamal/Desktop/SML_Assgn_1/FA/data/FMNIST"
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
if_train, il_train = load_mnist(fmnistdata, kind='train')
if_test, il_test = load_mnist(fmnistdata, kind='t10k')


# In[2]:


ff_train = deepcopy(if_train)
ff_train[ff_train<=127] = 0
ff_train[ff_train>127] = 1

ff_test = deepcopy(if_test)
ff_test[ff_test<=127] = 0
ff_test[ff_test>127] = 1

s1=list(il_train).count(0)
s2=list(il_train).count(1)
s3=list(il_train).count(2)
s4=list(il_train).count(3)
s5=list(il_train).count(4)
s6=list(il_train).count(5)
s7=list(il_train).count(6)
s8=list(il_train).count(7)
s9=list(il_train).count(8)
s10=list(il_train).count(9)
s21=len(il_train)
s11=s1/s21
s12=s2/s21
s13=s3/s21
s14=s4/s21
s15=s5/s21
s16=s6/s21
s17=s7/s21
s18=s8/s21
s19=s9/s21
s20=s10/s21 


# In[3]:


one_class=np.zeros((784,10))
zero_class=np.zeros((784,10))

for i in range(0,len(ff_train[0])):
    for j in range (0,len(ff_train)):
      t_label=il_train[j]
      if ff_train[j][i]==1:      
        one_class[i][t_label]+=1
      elif ff_train[j][i]==0:
        zero_class[i][t_label]+=1
for i in range(0,784):
    for j in range (0,10):
      one_class[i][j]+=1
      one_class[i][j]=one_class[i][j]/100
      if j==0:
        one_class[i][j]=(one_class[i][j]/s1)
      elif j==1:
        one_class[i][j]=(one_class[i][j]/s2)
      elif j==2:
        one_class[i][j]=(one_class[i][j]/s3)
      elif j==3:
        one_class[i][j]=(one_class[i][j]/s4)
      elif j==4:
        one_class[i][j]=(one_class[i][j]/s5)
      elif j==5:
        one_class[i][j]=(one_class[i][j]/s6)
      elif j==6:
        one_class[i][j]=(one_class[i][j]/s7)
      elif j==7:
        one_class[i][j]=(one_class[i][j]/s8)
      elif j==8:
        one_class[i][j]=(one_class[i][j]/s9)
      elif j==9:
        one_class[i][j]=(one_class[i][j]/s10)
for i in range(0,784):
    for j in range (0,10):
      zero_class[i][j]+=1
      zero_class[i][j]=zero_class[i][j]/100
      if j==0:
        zero_class[i][j]=(zero_class[i][j]/s1)
      elif j==1:
        zero_class[i][j]=(zero_class[i][j]/s2)
      elif j==2:
        zero_class[i][j]=(zero_class[i][j]/s3)
      elif j==3:
        zero_class[i][j]=(zero_class[i][j]/s4)
      elif j==4:
        zero_class[i][j]=(zero_class[i][j]/s5)
      elif j==5:
        zero_class[i][j]=(zero_class[i][j]/s6)
      elif j==6:
        zero_class[i][j]=(zero_class[i][j]/s7)
      elif j==7:
        zero_class[i][j]=(zero_class[i][j]/s8)
      elif j==8:
        zero_class[i][j]=(zero_class[i][j]/s9)
      elif j==9:
        zero_class[i][j]=(zero_class[i][j]/s10)


# In[4]:


class_images=np.zeros((10,len(ff_test)))
for i in range(0,10):
    for j in range(0,len(ff_test)):
      p_t=0
      for k in range (0,784): 
        if ff_test[j][k]==1:
          p_t+=math.log(one_class[k][i])
        elif ff_test[j][k]==0:
          p_t+=math.log(zero_class[k][i])
      if i==0:
        p_t=p_t *s11
      elif i==1:
        p_t=p_t *s12
      elif i==2:
        p_t=p_t *s13
      elif i==3:
        p_t=p_t *s14
      elif i==4:
        p_t=p_t *s15
      elif i==5:
        p_t=p_t *s16
      elif i==6:
        p_t=p_t *s17
      elif i==7:
        p_t=p_t *s18
      elif i==8:
        p_t=p_t *s19
      elif i==9:
        p_t=p_t *s20
      class_images[i][j]=p_t
clms_sum=np.sum(class_images,axis=0)
for i in range(0,len(class_images[0])):
    for j in range(0,len(class_images)):
        class_images[j][i]=class_images[j][i]/clms_sum[i]
# print(class_images)
# print(np.sum(class_images,axis=0))


# In[5]:


plt.figure()
acc=[]
for i in range (0,len(class_images)):
    labels=[]
    threshold=[0.08,0.085,0.09,0.092,0.095,0.097,0.099,0.1,0.12,0.15,0.5]
#     threshold=[0.05,0.5]
    
    for j in range (0,len(threshold)):
        label=[]
        for k in range (0,len(class_images[0])):
            if class_images[i][k]<=threshold[j]:
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
        for k in range(0,len(il_test)):
          if il_test[k]==i:
            if labels[j][k]==i:
                T_TP+=+1
            else:
                F_NP+=1
          elif il_test[k]!=i:
            if labels[j][k]==-100:
                T_NP+=1
            else:
                F_FP+=1 
        
        if(j==len(threshold)-4):
            print("Class : ",str(i))
            precision=(T_TP/(T_TP+F_FP))
            recall=(T_TP/(T_TP+F_NP))
            print("Precision ",str(precision))
            print("Recall ",str(recall))        
            c_matrix=np.zeros((2,2))
            c_matrix[0][0]=T_TP
            c_matrix[0][1]=F_FP
            c_matrix[1][0]=F_NP
            c_matrix[1][1]=T_NP
            print(c_matrix)
        TPP.append(T_TP/(T_TP+F_NP))
        FPP.append(F_FP/(T_NP+F_FP))
        acc.append(TPP)
#     print(TPP)
#     print(FPP)
    rdff='Class: '+str(i)
#     print(rdff)
    plt.plot(FPP,TPP,label=rdff)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.show()


# In[117]:


ans_class=np.zeros((10))
for i in range(0,len(ff_test)):
    t_lbl=il_test[i]
    t_op=class_images[t_lbl][i]
    t_ary=class_images[:,i]
    t_ary.sort()
    cv=list(t_ary).index(t_op)
    ans_class[cv]+=1
cdf_rank=[]
for i in range(0,len(ans_class)):
    cdf_rank.append(i)
    if i!=0:
        ans_class[i]+=ans_class[i-1]
for i in range(0,len(ans_class)):
    ans_class[i]=ans_class[i]/10000
plt.xlabel('Rank')
plt.ylabel('Identification Accuracy')
plt.plot(cdf_rank,ans_class)
plt.show
    
    
#         plt.plot(np.sum(acc[i]),i)
# plt.plot(axz,azx)
# plt.xlabel('Rank')
# plt.ylabel('Identification Rate')
# plt.show()  


# In[ ]:




