#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os
import gzip
import math
import matplotlib.pyplot as plt
import random
from copy import copy, deepcopy
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

if_train, ill_train = load_mnist(fmnistdata, kind='train')
if_test, ill_test = load_mnist(fmnistdata, kind='t10k')


# In[2]:


aff_train = deepcopy(if_train)
aff_train[aff_train<=127] = 0
aff_train[aff_train>127] = 1
aff_test = deepcopy(if_test)
aff_test[aff_test<=127] = 0
aff_test[aff_test>127] = 1
ff_train=[]
ff_test=[]
il_train=[]
il_test=[]

for i in range (0,len(ill_train)):
    if ill_train[i]==1 or ill_train[i]==2:
      ff_train.append(aff_train[i])
      atp=ill_train[i]-1
      il_train.append(atp)


for i in range (0,len(ill_test)):
    if ill_test[i]==1 or ill_test[i]==2:
      ff_test.append(aff_test[i])
      atp=ill_test[i]-1
      il_test.append(atp)

s1=il_train.count(0)
s2=il_train.count(1)
s3=s1/(s1+s2)
s4=s2/(s1+s2)

one_class=np.zeros((784,2))
zero_class=np.zeros((784,2))

for i in range(0,len(ff_train[0])):
    for j in range (0,len(ff_train)):
      t_label=il_train[j]
      if ff_train[j][i]==1:      
        one_class[i][t_label]+=1
      elif ff_train[j][i]==0:
        zero_class[i][t_label]+=1
    
for i in range(0,784):
    for j in range (0,2):
      one_class[i][j]+=1
      one_class[i][j]=one_class[i][j]/100
      if j==0:
        one_class[i][j]=one_class[i][j]/s1      
      else:
        one_class[i][j]=one_class[i][j]/s2


for i in range(0,784):
    for j in range (0,2):
      zero_class[i][j]+=1
      zero_class[i][j]=zero_class[i][j]/100
      if j==0:
        zero_class[i][j]=zero_class[i][j]/s1
      else:
        zero_class[i][j]=zero_class[i][j]/s2


# In[3]:


class_images=np.zeros((2,len(ff_test)))
for i in range(0,2):
    for j in range(0,len(ff_test)):
      p_t=0
      for k in range (0,784):
        if ff_test[j][k]==1:
          # print(p_t * one_class[k][i])
          p_t+=math.log(one_class[k][i])

        elif ff_test[j][k]==0:
          # print(p_t * zero_class[k][i])          
          p_t+=math.log(zero_class[k][i])

      # divide each by its count
      if i==0:
        p_t=p_t *s3
      else:
        p_t=p_t *s4
      class_images[i][j]=p_t

clms_sum=np.sum(class_images,axis=0)
for i in range(0,len(class_images[0])):
    for j in range(0,len(class_images)):
        class_images[j][i]=class_images[j][i]/clms_sum[i]

    
print(class_images)


# In[8]:


labels=[]
labels2=[]
threshold=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in range (0,len(threshold)):
    label=[]
    label2=[]
    for j in range (0,len(class_images[0])):
        if class_images[0][j]<=threshold[i]:
            label.append(0)
        else:
            label.append(1)
    labels.append(label)
for i in range (0,len(threshold)):
    label=[]
    for j in range (0,len(class_images[0])):
        if class_images[1][j]<=threshold[i]:
            label.append(1)
        else:
            label.append(0)
    labels2.append(label)

            


# In[9]:


TPP=[]
FPP=[]
for j in range (0,len(labels)):
    T_TP=0
    T_NP=0
    F_FP=0
    F_NP=0    
    for i in range(0,len(il_test)):
      if il_test[i]==labels[j][i]:
        if il_test[i]==0:
          T_TP+=+1
        else:
          T_NP+=1          
      else:
        if il_test[i]==1:
          F_FP+=1
        else:
          F_NP+=1
    if(j==4):
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
TPP2=[]
FPP2=[]
for j in range (0,len(labels)):
    T_TP=0
    T_NP=0
    F_FP=0
    F_NP=0    
    for i in range(0,len(il_test)):
      if il_test[i]==labels2[j][i]:
        if il_test[i]==1:
          T_TP+=+1
        else:
          T_NP+=1          
      else:
        if il_test[i]==0:
          F_FP+=1
        else:
          F_NP+=1
    if(j==4):
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
    TPP2.append(T_TP/(T_TP+F_NP))
    FPP2.append(F_FP/(T_NP+F_FP))


# In[11]:


plt.plot(FPP,TPP,label='1 as positive class')
plt.plot(FPP2,TPP2,label='2 as positive class')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




