import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.restoration import denoise_wavelet

filename='data_djt02_0918.csv'
df=pd.read_csv(filename)#,encoding='big5')
N=df[['N']]
E=df[['E']]
H=df[['H']]
NMEA=df[['NMEA_type']]

N=np.array(N).astype('float')
E=np.array(E).astype('float')
H=np.array(H).astype('float')
NMEA=np.array(NMEA).astype('float')

for i in range(len(NMEA)):
    if NMEA[i]==4:
        temp_N=N[i]
        temp_E=E[i]
        temp_H=H[i]
    if NMEA[i]!=4:
        N[i]=temp_N
        E[i]=temp_E
        H[i]=temp_H
        
N=np.reshape(N,(len(N)))
E=np.reshape(E,(len(E)))
H=np.reshape(H,(len(H)))

plt.plot(N,'b')##054E9F
plt.title("N",{'fontsize':10})
plt.show()

cA5,cD5,cD4,cD3,cD2,cD1=pywt.wavedec(N,'db7',mode='constant',level=5)
cD1 = pywt.threshold(cD1,0.004,mode='hard',substitute=0)
cD2 = pywt.threshold(cD2,0.008,mode='hard',substitute=0)
cD3 = pywt.threshold(cD3,0.01,mode='hard',substitute=0)
cD4 = pywt.threshold(cD4,0.004,mode='hard',substitute=0)
cD5 = pywt.threshold(cD5,0.004,mode='hard',substitute=0)
coeffs=[cA5,cD5,cD4,cD3,cD2,cD1]
de_noise=pywt.waverec(coeffs, 'db7')

# plt.subplot(2, 1, 1)
# plt.plot(E,'g')  
# plt.title("E", {'fontsize':10})

# plt.subplot(2, 1, 2)
plt.plot(de_noise,'b')  
plt.title("de_noise", {'fontsize':10})

plt.show()

cA5,cD5,cD4,cD3,cD2,cD1=pywt.wavedec(de_noise,'db7',mode='constant',level=5)
coeffsA5=[cA5,np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
coeffsD5=[np.zeros(len(cA5)),cD5,np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
coeffsD4=[np.zeros(len(cA5)),np.zeros(len(cD5)),cD4,np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
coeffsD3=[np.zeros(len(cA5)),np.zeros(len(cD5)),np.zeros(len(cD4)),cD3,np.zeros(len(cD2)),np.zeros(len(cD1))]
coeffsD2=[np.zeros(len(cA5)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),cD2,np.zeros(len(cD1))]
coeffsD1=[np.zeros(len(cA5)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),cD1]
coeffsMix=[np.zeros(len(cA5)),cD5,cD4,cD3,cD2,cD1]

A5=pywt.waverec(coeffsA5, 'db7')
D5=pywt.waverec(coeffsD5, 'db7')
D4=pywt.waverec(coeffsD4, 'db7')
D3=pywt.waverec(coeffsD3, 'db7')
D2=pywt.waverec(coeffsD2, 'db7')
D1=pywt.waverec(coeffsD1, 'db7')
mix=pywt.waverec(coeffsMix, 'db7')

plt.subplot(3, 2, 1)
plt.plot(D1,'r')  
plt.title("D1", {'fontsize':10})

plt.subplot(3, 2, 2)
plt.plot(D2,'k')  
plt.title("D2", {'fontsize':10})

plt.subplot(3, 2, 3)
plt.plot(D3,'r')  
plt.title("D3", {'fontsize':10})

plt.subplot(3, 2, 4)
plt.plot(D4,'k')  
plt.title("D4", {'fontsize':10})

plt.subplot(3, 2, 5)
plt.plot(D5,'r')  
plt.title("A5", {'fontsize':10})

plt.subplot(3, 2, 6)
plt.plot(A5,'r')  
plt.title("A5", {'fontsize':10})

plt.show()

plt.plot(mix,'g')##054E9F
plt.title("earthquake_wave",{'fontsize':10})
plt.show()