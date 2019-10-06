import numpy as np 
import matplotlib.pyplot as plt 
from scipy.fftpack import fft 
from scipy.signal import windows

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
    ax0.plot(f1[0:int(len(f1)/2)]); ax0.set(title='FFT sin ventana', ylabel='dB')
    ax1.plot(f2[0:int(len(f2)/2)]); ax1.set(title='FFT con ventana hann', ylabel='dB')
    ax2.plot(f3[0:int(len(f3)/2)]); ax2.set(title='FFT con ventana hamming', ylabel='dB')
    ax3.plot(f4[0:int(len(f4)/2)]); ax3.set(title='FFT con ventana blackman', ylabel='dB', xlabel='Frecuencia [Hz]')

    plt.show()
    return fig

def psd(s, l, t, sr):
    index = np.concatenate((np.arange(0, len(s)-l, l), np.delete((np.arange(0, len(s)-l, l)-(l*(t/100))),0))).astype(int)
    epochs = np.array([s[i:i+l] for i in index])
    w = np.linspace(0, sr, l)
    mfft = (np.abs(fft(epochs)))
    mfft = (1/len(epochs[:,0]))*np.sum(mfft, axis=0)
    fig, ax = plt.subplots()
    ax.plot(w[0:int(len(mfft)/2)], mfft[0:int(len(mfft)/2)])
    plt.show()  
    return fig


