import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fftpack import fft 
from scipy.signal import windows
import scipy.signal as sg

def FFT (signal):
    f1 = 20*np.log10(np.abs(fft(signal)))
    N = len(signal)
    w1 = windows.hann(N)
    w2 = windows.hamming(N)
    w3 = windows.blackman(N)
    f2 = 20*np.log10(np.abs(fft(signal*w1)))
    f3 = 20*np.log10(np.abs(fft(signal*w2)))
    f4 = 20*np.log10(np.abs(fft(signal*w3)))
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True, constrained_layout=True)
    ax0.plot(f1[0:len(f1)//2]); ax0.set(title='FFT sin ventana', ylabel='dB')
    ax1.plot(f2[0:len(f2)//2]); ax1.set(title='FFT con ventana hann', ylabel='dB')
    ax2.plot(f3[0:len(f3)//2]); ax2.set(title='FFT con ventana hamming', ylabel='dB')
    ax3.plot(f4[0:len(f4)//2]); ax3.set(title='FFT con ventana blackman', ylabel='dB', xlabel='Frecuencia [Hz]')

    plt.show()
    return fig

def psd(s, l, t, sr):
    index=np.array([0, l])
    while index[-1]+l<len(s):
        index= np.append(index, np.array([index[-1]-l*(t/100), index[-1]-l*(t/100)+l]) )
    index = np.array(np.sort(np.delete(index, -1))).astype(int)
    epochs = np.array([s[i:i+l] for i in index])
    w = np.linspace(0, 2*sr, l)
    mfft = (20*np.log10(np.abs(fft(epochs))))
    mfft = (1/len(epochs[:,0]))*np.sum(mfft, axis=0)
    fig, ax = plt.subplots()
    ax.plot(w[0:int(len(mfft)/2)], mfft[0:int(len(mfft)/2)]);ax.set(title='PSD', xlabel='Frecuencia (Hz)', ylabel='dB')
    plt.show()  
    return fig

def filter(a, b, sr):
    w, h = sg.freqz(b, a)
    fig, ax =plt.subplots()
    ax.plot(w*sr/np.pi, 20*np.log10(np.abs(h))); ax.set(title='Respuesta del filtro', xlabel='Frecuencia (Hz)', ylabel='dB')
    plt.show()
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
    fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True, sharex=True)
    ax0.plot(time, s); ax0.set(title='Señal sin filtrar', ylabel='Voltaje')
    ax1.plot(time, filt); ax1.set(title='Señal filtrada', ylabel='Voltaje', xlabel='Tiempo')
    plt.show()
    return fig