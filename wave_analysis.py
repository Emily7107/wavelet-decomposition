import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename='data_0804.csv'
df=pd.read_csv(filename,encoding='big5')
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
    if NMEA[i]==1 or NMEA[i]==2:
        N[i]=temp_N
        E[i]=temp_E
        H[i]=temp_H
        
N=np.reshape(N,(len(N)))
E=np.reshape(E,(len(E)))
H=np.reshape(H,(len(H)))

cA12,cD12,cD11,cD10,cD9,cD8,cD7,cD6,cD5,cD4,cD3,cD2,cD1=pywt.wavedec(H,'db5',mode='constant',level=12)
# ceoffs=[cA12,np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),cD6,cD5,cD4,cD3,cD2,cD1]

ceoffsA12=[cA12,np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD12=[np.zeros(len(cA12)),cD12,np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD11=[np.zeros(len(cA12)),np.zeros(len(cD12)),cD11,np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD10=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),cD10,np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD9=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),cD9,np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD8=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),cD8,np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD7=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),cD7,np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD6=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),cD6,np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD5=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),cD5,np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD4=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),cD4,np.zeros(len(cD3)),np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD3=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),cD3,np.zeros(len(cD2)),np.zeros(len(cD1))]
ceoffsD2=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),cD2,np.zeros(len(cD1))]
ceoffsD1=[np.zeros(len(cA12)),np.zeros(len(cD12)),np.zeros(len(cD11)),np.zeros(len(cD10)),np.zeros(len(cD9)),np.zeros(len(cD8)),np.zeros(len(cD7)),np.zeros(len(cD6)),np.zeros(len(cD5)),np.zeros(len(cD4)),np.zeros(len(cD3)),np.zeros(len(cD2)),cD1]

A12=pywt.waverec(ceoffsA12, 'db5')
D12=pywt.waverec(ceoffsD12, 'db5')
D11=pywt.waverec(ceoffsD11, 'db5')
D10=pywt.waverec(ceoffsD10, 'db5')
D9=pywt.waverec(ceoffsD9, 'db5')
D8=pywt.waverec(ceoffsD8, 'db5')
D7=pywt.waverec(ceoffsD7, 'db5')
D6=pywt.waverec(ceoffsD6, 'db5')
D5=pywt.waverec(ceoffsD5, 'db5')
D4=pywt.waverec(ceoffsD4, 'db5')
D3=pywt.waverec(ceoffsD3, 'db5')
D2=pywt.waverec(ceoffsD2, 'db5')
D1=pywt.waverec(ceoffsD1, 'db5')

plt.subplot(5, 3, 1)
plt.plot(D1,'r')  
plt.title("D1", {'fontsize':10})

plt.subplot(5, 3, 2)
plt.plot(D2,'k')  
plt.title("D2", {'fontsize':10})

plt.subplot(5, 3, 3)
plt.plot(D3,'r')  
plt.title("D3", {'fontsize':10})

plt.subplot(5, 3, 4)
plt.plot(D4,'k')  
plt.title("D4", {'fontsize':10})

plt.subplot(5, 3, 5)
plt.plot(D5,'r')  
plt.title("D5", {'fontsize':10})

plt.subplot(5, 3, 6)
plt.plot(D6,'k')  
plt.title("D6", {'fontsize':10})

plt.subplot(5, 3, 7)
plt.plot(D7,'r')  
plt.title("D7", {'fontsize':10})

plt.subplot(5, 3, 8)
plt.plot(D8,'k')  
plt.title("D8", {'fontsize':10})

plt.subplot(5, 3, 9)
plt.plot(D9,'r')  
plt.title("D9", {'fontsize':10})

plt.subplot(5, 3, 10)
plt.plot(D10,'k')  
plt.title("D10", {'fontsize':10})

plt.subplot(5, 3, 11)
plt.plot(D11,'r')  
plt.title("D11", {'fontsize':10})

plt.subplot(5, 3, 12)
plt.plot(D12,'k')  
plt.title("D12", {'fontsize':10})

plt.subplot(5, 3, 13)
plt.plot(A12,'k')  
plt.title("A12", {'fontsize':10})

plt.show()

plt.figure()
plt.plot(A12,'coral')##054E9F
plt.title("H_A12",{'fontsize':10})
plt.show()


# plt.subplot(2, 1, 1)
# plt.plot(N,'r')  
# plt.title("N", {'fontsize':10})

# plt.subplot(2, 1, 2)
# plt.plot(reverse,'k')  
# plt.title("N", {'fontsize':10})

# plt.subplot(1, 3, 2)
# plt.plot(E,'g')  
# plt.title("E", {'fontsize':10})

# plt.subplot(1, 3, 3)
# plt.plot(H,'b')  
# plt.title("H", {'fontsize':10})

