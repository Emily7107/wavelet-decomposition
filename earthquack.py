import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.restoration import denoise_wavelet

def wavelets(wave,level):
    
    cA5,cD5,cD4,cD3,cD2,cD1=pywt.wavedec(wave,'db7',mode='constant',level=level)
    cD1 = denoise_wavelet(cD1, wavelet='db7', mode='soft', wavelet_levels=5, method='BayesShrink', rescale_sigma='True')
    cD2 = denoise_wavelet(cD2, wavelet='db7', mode='soft', wavelet_levels=5, method='BayesShrink', rescale_sigma='True')
    cD3 = denoise_wavelet(cD3, wavelet='db7', mode='soft', wavelet_levels=5, method='BayesShrink', rescale_sigma='True')
    cD4 = denoise_wavelet(cD4, wavelet='db7', mode='soft', wavelet_levels=5, method='BayesShrink', rescale_sigma='True')
    cD5 = denoise_wavelet(cD5, wavelet='db7', mode='soft', wavelet_levels=5, method='BayesShrink', rescale_sigma='True')
    cA5 = denoise_wavelet(cA5, wavelet='db7', mode='soft', wavelet_levels=5, method='BayesShrink', rescale_sigma='True')
    
    ceoffsA5=[cA5,np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
    ceoffsD5=[np.zeros(len(cA5)),cD5,np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
    ceoffsD4=[np.zeros(len(cA5)),np.zeros(len(cD5)),cD4,np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
    ceoffsD3=[np.zeros(len(cA5)),np.zeros(len(cD5)),np.zeros(len(cD4)),cD3,np.zeros(len(cD2)),np.zeros(len(cD1))]
    ceoffsD2=[np.zeros(len(cA5)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),cD2,np.zeros(len(cD1))]
    ceoffsD1=[np.zeros(len(cA5)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),cD1]
    coeffsMix=[np.zeros(len(cA5)),np.zeros(len(cD5)),cD4,cD3,cD2,np.zeros(len(cD1))]

    A5=pywt.waverec(ceoffsA5, 'db7')
    D5=pywt.waverec(ceoffsD5, 'db7')
    D4=pywt.waverec(ceoffsD4, 'db7')
    D3=pywt.waverec(ceoffsD3, 'db7')
    D2=pywt.waverec(ceoffsD2, 'db7')
    D1=pywt.waverec(ceoffsD1, 'db7')
    mix=pywt.waverec(coeffsMix,'db7')

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

    return D5

# 程式進入點
filename='data_djt03_0918.csv'
df=pd.read_csv(filename)#,encoding='big5')
N=df[['N']]
E=df[['E']]
H=df[['H']]
NMEA=df[['NMEA_type']]
time=df[['Time']]

N=np.array(N).astype('float')
E=np.array(E).astype('float')
H=np.array(H).astype('float')
NMEA=np.array(NMEA).astype('float')
time=np.array(time).astype('str')

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
time=np.reshape(time,(len(time)))

plt.plot(E,'g')##054E9F
plt.title("E",{'fontsize':10})
plt.show()

wave=wavelets(E,4)
# wave=wavelets(wave,4)
# wave=wavelets(wave,4)

plt.figure()
plt.plot(wave,'g')##054E9F
plt.title("E_D4",{'fontsize':10})
plt.show()