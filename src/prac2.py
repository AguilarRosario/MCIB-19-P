import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import MCI 

s = np.loadtxt('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\data\\Practica 1\\Registro1Act1.txt')
fig, ax = plt.subplots()
ax.plot(s[:,0], s[:,2])
plt.show(block=False)

fig = MCI.FFT(s[:,2])
fig.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\pruebaFFT.png')

sr = 1000
l = 128
tr = 25
fig = MCI.psd(s[:,2], l, tr, sr)
fig.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\pruebaPSD.png')

fc = 50
b, a = sg.butter(10, fc/sr, btype='lowpass')
fig = MCI.filter(a, b,sr)
fig.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\pruebafilter.png')

fig = MCI.media_movil(s[:,2], l, tr, s[:,0])
fig.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\pruebamedia_movil.png')