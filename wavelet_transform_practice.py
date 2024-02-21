# from typing import Concatenate
import pywt
import numpy as np
import matplotlib.pyplot as plt

t_n = 1
N = 1000
T = t_n / N
f_s = 1/T

# Get x values of the sine wave
frequencies = [4, 30, 90, 300]
xa = np.linspace(start=0, stop=t_n, num=N)

# Amplitude of the sine wave is sine of a variable like time
amplitude= np.sin(2*np.pi*frequencies[0]*xa)*2
amplitude2= np.sin(2*np.pi*frequencies[1]*xa)
amplitude3= np.sin(2*np.pi*frequencies[2]*xa)
amplitude4= np.sin(2*np.pi*frequencies[3]*xa)
mix=amplitude+amplitude2+amplitude3+amplitude4

cA6,cD6,cD5,cD4,cD3,cD2,cD1=pywt.wavedec(mix,'db5',mode='constant',level=6)
ceoffs=[cA6,np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
reverse=pywt.waverec(ceoffs, 'db5')

# cA1, cD1 = pywt.dwt(mix,'db5')
# cA2, cD2 = pywt.dwt(cA1,'db5')
# cA3, cD3 = pywt.dwt(cA2,'db5')
# cA4, cD4 = pywt.dwt(cA3,'db5')

# incA3=pywt.idwt(cA4,np.linspace(start=0, stop=0, num=len(cA4)),'db5')
# incA2=pywt.idwt(incA3,np.linspace(start=0, stop=0, num=len(incA3)),'db5')
# incA1=pywt.idwt(incA2,np.linspace(start=0, stop=0, num=len(incA2)),'db5')
# inverse=pywt.idwt(incA1,np.linspace(start=0, stop=0, num=len(incA1)),'db5')

# A3=pywt.idwt(cA4,cD4,'db5')
# A2=pywt.idwt(cA3,cD3,'db5')
# A1=pywt.idwt(cA2,cD2,'db5')
# A=pywt.idwt(cA1,cD1,'db5')

# print('cA3=',cA3)
# print('cD3=',cD3)
# print('cD2=',cD2)
# print('cD1=',cD1)
# print('Ceoffs=',Ceoffs)
# print('mix=',mix)

plt.subplot(2, 1, 1)
plt.plot(mix,'r')  
plt.title("mix", {'fontsize':10})

plt.subplot(2, 1, 2)
plt.plot(reverse,'k')  
plt.title("reverse", {'fontsize':10})

# plt.subplot(2, 3, 1)
# plt.plot(cD1,'r')  
# plt.title("mix", {'fontsize':10})

# plt.subplot(2, 3, 2)
# plt.plot(cD2,'g')  
# plt.title("inverse", {'fontsize':10})

# plt.subplot(2, 3, 3)
# plt.plot(cD3,'g')  
# plt.title("inverse", {'fontsize':10})

# plt.subplot(2, 3, 4)
# plt.plot(cD4,'g')  
# plt.title("inverse", {'fontsize':10})

# plt.subplot(2, 3, 5)
# plt.plot(cA4,'g')  
# plt.title("inverse", {'fontsize':10})

# plt.subplot(2, 3, 3)
# plt.plot(len(cD3),cD3,'b')  
# plt.title("sin(2*pi*60t)", {'fontsize':10})

# plt.subplot(2, 3, 4)
# plt.plot(len(cA3),cA3,'m')  
# plt.title("mix wave", {'fontsize':10})

# plt.subplot(2, 3, 5)
# plt.plot(xa,mix,'m')  
# plt.title("mix wave", {'fontsize':10})
# plt.subplot(2, 3, 6)
# plt.plot(len(Ceoffs),Ceoffs,'m')  
# plt.title("mix wave", {'fontsize':10})

plt.show()
