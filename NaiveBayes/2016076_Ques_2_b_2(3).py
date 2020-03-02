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
fmnistdata="/home/ajkamal/Desktop/SML_Assgn_1/FA/data/MNIST"
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

fa_train=[]
fa_test=[]
ia_train=[]
ia_test=[]

for i in range (0,len(ill_train)):
    if ill_train[i]==3 or ill_train[i]==8:
      fa_train.append(aff_train[i])
      ia_train.append(ill_train[i])


for i in range (0,len(ill_test)):
    if ill_test[i]==3 or ill_test[i]==8:
      fa_test.append(aff_test[i])
      ia_test.append(ill_test[i])
# 6742
# 5851
#7:6


# In[4]:


#Straitified K-fold
one_f=[]
one_l=[]
et_f=[]
et_l=[]

for i in range (0,len(ia_train)):
    if ia_train[i]==3:
        one_f.append(fa_train[i])
        one_l.append(ia_train[i])
    elif ia_train[i]==8:
        et_f.append(fa_train[i])
        et_l.append(ia_train[i])

cpp1 = list(zip(one_f,one_l))
random.shuffle(cpp1)
one_f,  one_l = zip(*cpp1)

cpp2 = list(zip(et_f,et_l))
random.shuffle(cpp2)
et_f, et_l = zip(*cpp2)

folds_f=[]
fold_l=[]
for i in range (0,5):
    aa=[]
    ab=[]
    if(i<4):
        for j in range (1198*i,1198*(i+1)):
            aa.append(one_f[j])
            ab.append(one_l[j])
        for j in range (1198*i,1198*(i+1)):
            aa.append(et_f[j])
            ab.append(et_l[j])
        folds_f.append(aa)
        fold_l.append(ab)
    else:
        for j in range (1198*i,len(one_l)):
            aa.append(one_f[j])
            ab.append(one_l[j])
        for j in range (1198*i,len(et_l)):
            aa.append(et_f[j])
            ab.append(et_l[j])
        folds_f.append(aa)
        fold_l.append(ab)
f_ff_train=[]
f_il_train=[]
Accuracy=[]
Accuracy2=[]
for i in range (0,5):
    ff_train=[]
    ff_test=[]
    il_train=[]
    il_test=[]
    ff_test=folds_f[i]
    il_test=fold_l[i]
    for j in range (0,5):
        if(j!=i):
            ff_train=folds_f[j]
            il_train=fold_l[j]
    print(i)
    f_ff_train.append(ff_train)
    f_il_train.append(il_train)
    s1=il_train.count(3)
    s2=il_train.count(8)
    s3=s1/(s1+s2)
    s4=s2/(s1+s2)
    one_class=np.zeros((784,2))
    zero_class=np.zeros((784,2))
    for j in range(0,len(ff_train[0])):
        for k in range (0,len(ff_train)):
          t_label=il_train[k]
          if ff_train[k][j]==1:
            if t_label==3:
                one_class[j][0]+=1
            elif t_label==8:
                one_class[j][1]+=1
          elif ff_train[k][j]==0:
            if t_label==3:
                zero_class[j][0]+=1
            elif t_label==8:
                zero_class[j][1]+=1           

    for j in range(0,784):
        for k in range (0,2):
          one_class[j][k]+=1
          one_class[j][k]=one_class[j][k]/100
          if j==0:
            one_class[j][k]=one_class[j][k]/s1      
          else:
            one_class[j][k]=one_class[j][k]/s2


    for j in range(0,784):
        for k in range (0,2):
          zero_class[j][k]+=1
          zero_class[j][k]=zero_class[j][k]/100
          if j==0:
            zero_class[j][k]=zero_class[j][k]/s1
          else:
            zero_class[j][k]=zero_class[j][k]/s2
    
    class_images=np.zeros((2,len(ff_test)))
#     print(class_images.shape)
    for j in range(0,2):
        for k in range(0,len(ff_test)):
          p_t=0
          for l in range (0,784):
            if ff_test[k][l]==1:
              # print(p_t * one_class[k][i])
              p_t+=math.log(one_class[l][j])

            elif ff_test[k][l]==0:
              # print(p_t * zero_class[k][i])          
              p_t+=math.log(zero_class[l][j])

          # divide each by its count
          if j==0:
            p_t=p_t *s3
          else:
            p_t=p_t *s4
          class_images[j][k]=p_t
#     print(class_images.shape)
    clms_sum=np.sum(class_images,axis=0)
    for j in range(0,len(class_images[0])):
        for k in range(0,len(class_images)):
            class_images[k][j]=class_images[k][j]/clms_sum[j]
#     print(class_images.shape)
#     print(class_images)
    Y_test=np.argmax(class_images, axis=0)
    listw=[]
    for j in Y_test:
        listw.append(j)
    TP=0
    FP=0
    TN=0
    FN=0
    for j in range(0,len(il_test)):
        if il_test[j]== 3:
            if listw[j]==0:
                TP+=1
            else:
                FN+=1
        elif il_test[j]== 8:
            if listw[j]==1:
                TN+=1
            else:
                FP+=1
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    print(accuracy)
    Accuracy.append(accuracy)
    
    class_images2=np.zeros((2,len(ff_train)))    
#     print(class_images.shape)
    for j in range(0,2):
        for k in range(0,len(ff_train)):
          p_t=0
          for l in range (0,784):
            if ff_train[k][l]==1:
              # print(p_t * one_class[k][i])
              p_t+=math.log(one_class[l][j])

            elif ff_train[k][l]==0:
              # print(p_t * zero_class[k][i])          
              p_t+=math.log(zero_class[l][j])

          # divide each by its count
          if j==0:
            p_t=p_t *s3
          else:
            p_t=p_t *s4
          class_images2[j][k]=p_t
        
    clms_sum2=np.sum(class_images2,axis=0)
    for j in range(0,len(class_images2[0])):
        for k in range(0,len(class_images2)):
            class_images2[k][j]=class_images2[k][j]/clms_sum2[j]
    Y_test2=np.argmax(class_images2, axis=0)
    listw2=[]
    for j in Y_test2:
        listw2.append(j)
    TP2=0
    FP2=0
    TN2=0
    FN2=0
    for j in range(0,len(il_train)):
        if il_train[j]== 3:
            if listw2[j]==0:
                TP2+=1
            else:
                FN2+=1
        elif il_train[j]== 8:
            if listw2[j]==1:
                TN2+=1
            else:
                FP2+=1
    accuracy2=(TP2+TN2)/(TP2+TN2+FP2+FN2)
    print(accuracy2)
    Accuracy2.append(accuracy2) 
    
    
print("Validation")
print(np.mean(Accuracy))
print(np.var(Accuracy))
print("Training")
print(np.mean(Accuracy2))
print(np.var(Accuracy2))


# In[5]:


ff_train=f_ff_train[0]
il_train=f_il_train[0]

s1=il_train.count(3)
s2=il_train.count(8)
s3=s1/(s1+s2)
s4=s2/(s1+s2)

one_class=np.zeros((784,2))
zero_class=np.zeros((784,2))
for j in range(0,len(ff_train[0])):
    for k in range (0,len(ff_train)):
      t_label=il_train[k]
      if ff_train[k][j]==1:
        if t_label==3:
            one_class[j][0]+=1
        elif t_label==8:
            one_class[j][1]+=1
      elif ff_train[k][j]==0:
        if t_label==3:
            zero_class[j][0]+=1
        elif t_label==8:
            zero_class[j][1]+=1           

for j in range(0,784):
    for k in range (0,2):
      one_class[j][k]+=1
      one_class[j][k]=one_class[j][k]/100
      if j==0:
        one_class[j][k]=one_class[j][k]/s1      
      else:
        one_class[j][k]=one_class[j][k]/s2


for j in range(0,784):
    for k in range (0,2):
      zero_class[j][k]+=1
      zero_class[j][k]=zero_class[j][k]/100
      if j==0:
        zero_class[j][k]=zero_class[j][k]/s1
      else:
        zero_class[j][k]=zero_class[j][k]/s2


# In[6]:


class_images=np.zeros((2,len(fa_test)))    
#     print(class_images.shape)
for j in range(0,2):
    for k in range(0,len(fa_test)):
      p_t=0
      for l in range (0,784):
        if fa_test[k][l]==1:
          # print(p_t * one_class[k][i])
          p_t+=math.log(one_class[l][j])

        elif fa_test[k][l]==0:
          # print(p_t * zero_class[k][i])          
          p_t+=math.log(zero_class[l][j])

      # divide each by its count
      if j==0:
        p_t=p_t *s3
      else:
        p_t=p_t *s4
      class_images[j][k]=p_t
#     print(class_images.shape)
clms_sum=np.sum(class_images,axis=0)
for j in range(0,len(class_images[0])):
    for k in range(0,len(class_images)):
        class_images[k][j]=class_images[k][j]/clms_sum[j]
# print(class_images)
# print(min(class_images[0]))
# print(min(class_images[1]))
# print(max(class_images[0]))
# print(max(class_images[1]))

# print(min(np.diff(class_images,axis=0)))


# In[7]:


class_images2=np.zeros((2,len(fa_train)))    
for j in range(0,2):
    for k in range(0,len(fa_train)):
      p_t=0
      for l in range (0,784):
        if fa_train[k][l]==1:
          # print(p_t * one_class[k][i])
          p_t+=math.log(one_class[l][j])

        elif fa_train[k][l]==0:
          # print(p_t * zero_class[k][i])          
          p_t+=math.log(zero_class[l][j])

      # divide each by its count
      if j==0:
        p_t=p_t *s3
      else:
        p_t=p_t *s4
      class_images2[j][k]=p_t
#     print(class_images.shape)
clms_sum2=np.sum(class_images2,axis=0)
for j in range(0,len(class_images2[0])):
    for k in range(0,len(class_images2)):
        class_images2[k][j]=class_images2[k][j]/clms_sum2[j]


# In[8]:


labels=[]

# threshold=[0.3,0.35,0.4,0.45,0.5,0.55]
threshold=[0.08,0.085,0.09,0.095,0.1,0.12,0.15,0.2,0.25,0.3,0.35,0.41,0.42,0.43,0.45,0.475,0.48,0.49,0.5,0.52,0.55,0.56,0.6,0.7]
for i in range (0,len(threshold)):
    label=[]
    for j in range (0,len(class_images[0])):
        if class_images[0][j]<=threshold[i]:
            label.append(0)
        else:
            label.append(1)
    labels.append(label)


# In[9]:


labels2=[]
# threshold=[0.3,0.35,0.4,0.45,0.5,0.55]
for i in range (0,len(threshold)):
    label=[]
    for j in range (0,len(class_images2[0])):
        if class_images2[0][j]<=threshold[i]:
            label.append(0)
        else:
            label.append(1)
    labels2.append(label)


# In[21]:


TPP=[]
FPP=[]
FNN=[]
TNN=[]
for j in range (0,len(labels)):
    T_TP=0
    F_FP=0
    T_NP=0
    F_NP=0
    for k in range(0,len(ia_test)):
        if ia_test[k]== 3:
            if labels[j][k]==0:
                T_TP+=1
            else:
                F_NP+=1
        elif ia_test[k]== 8:
            if labels[j][k]==1:
                T_NP+=1
            else:
                F_FP+=1
    if(j==20):       
        c_matrix=np.zeros((2,2))
        c_matrix[0][0]=T_TP
        c_matrix[0][1]=F_FP
        c_matrix[1][0]=F_NP
        c_matrix[1][1]=T_NP
        print(c_matrix)
    TPP.append(T_TP/(T_TP+F_NP))
    FPP.append(F_FP/(F_FP+T_NP))
    FNN.append(F_NP/(F_NP+T_TP))
    TNN.append(T_NP/(T_NP+F_FP))


# In[22]:


TPP2=[]
FPP2=[]
FNN2=[]
TNN2=[]
for j in range (0,len(labels2)):
    T_TP=0
    F_FP=0
    T_NP=0
    F_NP=0
    for k in range(0,len(ia_train)):
        if ia_train[k]== 3:
            if labels2[j][k]==0:
                T_TP+=1
            else:
                F_NP+=1
        elif ia_train[k]== 8:
            if labels2[j][k]==1:
                T_NP+=1
            else:
                F_FP+=1
    if(j==20):       
        c_matrix=np.zeros((2,2))
        c_matrix[0][0]=T_TP
        c_matrix[0][1]=F_FP
        c_matrix[1][0]=F_NP
        c_matrix[1][1]=T_NP
        print(c_matrix)
    TPP2.append(T_TP/(T_TP+F_NP))
    FPP2.append(F_FP/(F_FP+T_NP))
    FNN2.append(F_NP/(F_NP+T_TP))
    TNN2.append(T_NP/(T_NP+F_FP))  


# In[12]:


plt.plot(FPP,TPP,label='Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(FPP2,TPP2,label='Training')
plt.legend(loc='lower right')
plt.show()


# In[13]:


plt.plot(FPP,FNN,label='Test')
diff_rate=[]
diff_rate2=[]
for i in range(0,len(FNN)):
    diff_rate.append(abs(FNN[i]-FPP[i])) 
indice=diff_rate.index(min(diff_rate))
print(threshold[indice])

plt.xlabel('FPR')
plt.ylabel('FNR')
plt.plot(FPP2,FNN2,label='Training')
plt.legend(loc='lower right')
for i in range(0,len(FNN2)):
    diff_rate2.append(abs(FNN2[i]-FPP2[i])) 
indice2=diff_rate2.index(min(diff_rate2))
print(threshold[indice2])
plt.show()


# In[ ]:




