# -*- coding: utf-8 -*-
"""Untitled0.ipynb
Raj Kamal Yadav
2016076
Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zqSK_6X_DWEw8PASUYSey7gp3Xf-N4Wo
"""

import numpy as np
import cmath
import matplotlib.pyplot as plt
import numpy as np
import string
import cv2
import random
from scipy import ndimage
import math

f=[[1,3,4],[2,5,3],[6,8,9]]
W=[[-1,-2,-3],[-4,0,1],[-6,-5,-1]]

f=np.array(f)
W=np.array(W)
# print(f.shape,w.shape)

print(f)
print(W)

fpadx=W.shape[0]-1
fpady=W.shape[1]-1
wpadx=f.shape[0]-1
wpady=f.shape[1]-1
# print(fpadx,fpady)
# print(wpadx,wpady)

a=int((3-1)/2)
b=int((3-1)/2)

# f=np.pad(f, ((int(fpadx/2),int(fpadx-fpadx/2)),(int(fpady/2),int(fpady-fpady/2))), 'constant')
f=np.pad(f,(0,2),'constant')
w=np.pad(W, (0,2+a), 'constant')
w=w[a:,b:]
print(f.shape,w.shape)

print(f)
print(w)

for k in range(-1*a,a+1):
    for l in range(-1*b,b+1):
        m=k
        n=l
        if(k<0):
            m=k %5
        if (l<0):
            n=l%5
        w[m][n]=W[k+a][l+b]

print(w)

from scipy import ndimage
ws= ndimage.convolve(np.array([[1,3,4],[2,5,3],[6,8,9]]),np.array([[-1,-2,-3],[-4,0,1],[-6,-5,-1]]),mode='constant', cval=0.0)
print(ws)

FF=np.fft.fft2(f)
# print(FF)
WW=np.fft.fft2(w)
# print(WW)
HH=np.multiply(FF,WW)
# print(HH)
IHH=np.fft.ifft2(HH)
print(IHH)

# F=np.zeros((f.shape[0],f.shape[1]))
F= [[0.0 for k in range(5)] for l in range(5)]
# F=np.array(F)
W=[[0.0 for k in range(5)] for l in range(5)]

# def DFT2D(image):
#     M=image.shape[0]
#     N=image.shape[1]
#     dft2d_red = [[0.0 for k in range(M)] for l in range(N)] 

#     for k in range(M):
#         for l in range(N):
#             sum_red = 0.0
#             for m in range(M):
#                 for n in range(N):
# #                     (red, alpha) = 
#                     e = np.exp(- 1j * np.pi*2 * (float(k * m) / M + float(l * n) / N))
#                     sum_red += image[m][n] * e
#             dft2d_red[l][k] = sum_red
#     return (dft2d_red)
# print(DFT2D(f))

for i in range(0,5):
    for j in range (0, 5):
        ijsum=0.0
        for k in range(0,f.shape[0]):
            for l in range (0, f.shape[1]):
                e=np.exp(- 1j * np.pi*2 * (float(i * k) / 5 + float(j * l) / 5))
                ijsum+=f[k][l]*e
        F[i][j]=ijsum     
# print(np.array(F))

for i in range(0,5):
    for j in range (0, 5):
        ijsum=0
        for k in range(0,w.shape[0]):
            for l in range (0, w.shape[1]):
                ijsum+=w[k][l]*cmath.exp((-1j*2*cmath.pi*i*k)/w.shape[0])*cmath.exp((-1j*2*cmath.pi*j*l)/w.shape[1])
        W[i][j]=ijsum  
# print(np.array(W))

H=np.multiply(np.array(F),np.array(W))
# print(H)
IH=[[0.0 for k in range(5)] for l in range(5)]

for i in range(0,5):
    for j in range (0, 5):
        ijsum=0
        for k in range(0,H.shape[0]):
            for l in range (0, H.shape[1]):
                ijsum+=H[k][l]*np.exp((1j*2*np.pi*i*k)/H.shape[0])*np.exp((1j*2*np.pi*j*l)/H.shape[1])
        IH[i][j]=(1/(5*5))*ijsum  
print(np.real(IH))

"""# **Ques 4. Unsharp Masking via DFT**"""

import numpy as np
import cmath
import matplotlib.pyplot as plt
import numpy as np
import string
import cv2
import random
from scipy import ndimage
import math

filter=[[1,1,1],[1,1,1],[1,1,1]]
filter=np.array(filter)
W=filter*(1/9)

img = cv2.imread('Chandrayaan2 - Q3a-inputimage.png')
f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
f=f.astype(np.float)

print(f.shape)

f=f[:512,:512]

print(f.shape)

fpadx=W.shape[0]-1
fpady=W.shape[1]-1
wpadx=f.shape[0]-1
wpady=f.shape[1]-1

a=int((3-1)/2)
b=int((3-1)/2)

# f=np.pad(f, ((int(fpadx/2),int(fpadx-fpadx/2)),(int(fpady/2),int(fpady-fpady/2))), 'constant')

f=np.pad(f,(0,514-f.shape[0]),'constant')
w=np.pad(W, (0,511+a), 'constant',constant_values=(0,0))
w=w[a:,b:]
print(f.shape,w.shape)

for k in range(-1*a,a+1):
    for l in range(-1*b,b+1):
        m=k
        n=l
        if(k<0):
            m=k %514
        if (l<0):
            n=l%514
        w[m][n]=W[k+a][l+b]

print(f.shape,w.shape)

FF=np.fft.fft2(f)
WW=np.fft.fft2(w)
HH=np.multiply(FF,WW)
ZF=FF-HH

ZZF=FF+ZF
ZZZF=np.fft.ifft2(ZZF)
FImg=np.real(ZZZF)
Imgplt=FImg[:512,:512]

triop=0
for i in range(0,Imgplt.shape[0]):
    for j in range(0,Imgplt.shape[1]):
        if(Imgplt[i][j]<0):
            triop+=1
            Imgplt[i][j]=0
print(triop)

print(Imgplt.shape)

cv2.imwrite("unsharpmask.png",Imgplt)

y=np.zeros((5,5), dtype=float)

y[0]= np.array([0,1, 0.2, 0.3, 0.4])
y[2,:]=np.array([0,1, 0.2, 0.3, 0.4])
y[4,:]=np.array([0,1, 0.2, 0.3, 0.4])
# y[1,3]=.2
# y[3,2]=.5
y[1][2]=0
y[3][3]=0
# y[1][0]=0.1

print(y)

z=np.fft.fft2(y)
print(np.abs(z))

z[:,0]=0
# z[:, 1:] = 1
print(np.abs(z))

ww=np.fft.ifft2(z)

print(np.abs(ww))

x = np.ones((5,5))
x[:,0] = 0
print(np.fft.ifft2(x))



