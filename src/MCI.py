import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fftpack import fft 
from scipy.signal import windows
import scipy.signal as sg

def FFT (signal, sr):
    w = np.linspace(0, 2*sr, len(signal))
    f1 = 20*np.log10(np.abs(fft(signal)))
    N = len(signal)
    w1 = windows.hann(N)
    w2 = windows.hamming(N)
    w3 = windows.blackman(N)
    f2 = 20*np.log10(np.abs(fft(signal*w1)))
    f3 = 20*np.log10(np.abs(fft(signal*w2)))
    f4 = 20*np.log10(np.abs(fft(signal*w3)))
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True, constrained_layout=True)
    ax0.plot(w[0:len(signal)//2], f1[0:len(f1)//2]); ax0.set(title='FFT sin ventana', ylabel='dB')
    ax1.plot(w[0:len(signal)//2], f2[0:len(f2)//2]); ax1.set(title='FFT con ventana hann', ylabel='dB')
    ax2.plot(w[0:len(signal)//2], f3[0:len(f3)//2]); ax2.set(title='FFT con ventana hamming', ylabel='dB')
    ax3.plot(w[0:len(signal)//2], f4[0:len(f4)//2]); ax3.set(title='FFT con ventana blackman', ylabel='dB', xlabel='Frecuencia [Hz]')
    return fig

def psd(s, l, t, sr):
    index=np.array([0, l])
    while index[-1]+l<len(s):
        index= np.append(index, np.array([index[-1]-l*(t/100), index[-1]-l*(t/100)+l]) )
    index = np.array(np.sort(np.delete(index, -1))).astype(int)
    epochs = np.array([s[i:i+l] for i in index])
    w = np.linspace(0, 0.5*sr, l)
    mfft = (20*np.log10(np.abs(fft(epochs))))
    mfft = (1/len(epochs[:,0]))*np.sum(mfft, axis=0)
    return mfft, w

def fil(a, b, sr):
    w, h = sg.freqz(b, a)
    fig, ax =plt.subplots()
    ax.plot(w*sr/np.pi, 20*np.log10(np.abs(h))); ax.set(title='Respuesta del filtro', xlabel='Frecuencia (Hz)', ylabel='dB')
    return fig

def media_movil(s, l, t, time):
    index=np.array([0, l])
    while index[-1]+l<len(s):
        index= np.append(index, np.array([index[-1]-l*(t/100), index[-1]-l*(t/100)+l]) )
    index = np.array(np.sort(np.delete(index, -1))).astype(int)

    epochs = np.array([s[i:i+l] for i in index])
    m = np.mean(epochs, axis=1)
    ti = epochs-m[:,np.newaxis]
    filt = ti[0,:]
    for i in range (1, ti.shape[0]):
        filt = np.append(filt, ti[i, int(len(filt)-index[i])::])
    filt = np.append(filt, s[index[-1]+l::]-np.mean(s[index[-1]+l::]))
    return filt

def detrend(s, time, l):
    t= 50
    index=np.array([0, l])
    while index[-1]+l<len(s):
        index= np.append(index, np.array([index[-1]-l*(t/100), index[-1]-l*(t/100)+l]) )
    index = np.array(np.sort(np.delete(index, -1))).astype(int)
    epochs = np.array([s[i:i+l] for i in index])
    for i in range (len(index)):
        x = np.append(time[index[i]:index[i]+l], np.ones(l)).reshape(2,l)
        y = epochs[i,:]
        w = np.dot(np.dot(np.linalg.inv(np.dot(x, x.T)), x), y.T)
        epochs[i,:] = epochs[i,:]-(w[0]*time[index[i]:index[i]+l]+w[1])

    filt = epochs[0,:]
    for i in range (1, len(epochs[:,0])):
        filt = np.append(filt, epochs[i, int(len(filt)-index[i])::])

    x = np.append(time[index[-1]+l::], np.ones(len(time[index[-1]+l::]))).reshape(2,len(time[index[-1]+l::]))
    y = s[index[-1]+l::]
    w = np.dot(np.dot(np.linalg.inv(np.dot(x, x.T)), x), y.T)
    filt = np.append(filt, s[index[-1]+l::]-(w[0]*time[index[-1]+l::]+w[1]))
    return filt

def zscore(s):
    media = (1/s.shape[1])*(np.sum(s, axis=1))
    sigma = np.sqrt((1/(s.shape[1]-1))*(np.sum((s-media[:,np.newaxis])**2, axis=1)))
    z     = (s-media[:,np.newaxis])/sigma[:,np.newaxis]
    return z

def histograma(s,nbins,width=0.5):
    intervals = np.linspace(s.min(),s.max(),nbins+1)
    intervals = intervals[np.newaxis]
    left  = s[:,np.newaxis]>intervals[0,:-1]
    right = s[:,np.newaxis]<intervals[0,1:]
    hist = (left*right).sum(axis=0)
    return intervals[0,:-1],hist

def epochs(signal, segt, sr):
    segt = segt*sr
    i = 0
    epochs = np.array([])
    while(i<len(signal)-segt):
        epochs = np.append(epochs, signal[i:i+segt])
        i = i+segt
    epochs = np.reshape(epochs, (int(len(signal)/segt),segt))
    return epochs

def coef_corr(signal1, signal2):
    r = np.array([])
    for i in range(signal1.shape[0]):
        S = np.cov(signal1[i,:], signal2[i,:])
        z = S[0,1]/np.product( np.sqrt(S.diagonal() ) )
        r = np.append(r, z)
    return r 