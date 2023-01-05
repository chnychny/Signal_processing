import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.signal import hilbert

def revNSPmaking(RN,X_in,Y_in,ranE):
    temp_ICRPmtx_out = np.zeros((1,1,RN ,RN ),dtype=np.float)
    for n in range(RN):
        idx_n=np.where((X_in>ranE[n]) &(X_in<=ranE[n+1]))
        for mm in range(len(idx_n[0])):
            y_value=Y_in[idx_n[0][mm]]
            for k in range(RN):
                if (y_value > ranE[k]) and (y_value<=ranE[k+1]):
                    temp_ICRPmtx_out[0,0,n,k]+=1

    temp_ICRPmtx_out[0,0,:,:] = temp_ICRPmtx_out/np.sum(temp_ICRPmtx_out)

    return temp_ICRPmtx_out
'''
# start - mtx에 넣기
signal, RN 변수설정
normalization 방법 알아서
'''
signal_len = 100
signal = np.sin(np.random.rand(signal_len))
RN = 50 # matrix 크기 RN x RN

# Normalization 최소값-1 / 최대값 1
X = minmax_scale(np.real(hilbert(signal)), feature_range=(-1, 1), axis=0,copy=True)
Y = minmax_scale(np.imag(hilbert(signal)), feature_range=(-1, 1), axis=0, copy=True)
#X = np.tanh(np.real(hilbert(signal)))
#Y = np.tanh(np.imag(hilbert(signal)))
ranE = np.linspace(-1, 1, num=RN + 1, dtype=np.float)
tanh_MTX = revNSPmaking(RN, X, Y, ranE)
print(tanh_MTX)