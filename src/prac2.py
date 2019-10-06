import numpy as np 
import matplotlib.pyplot as plt 
import MCI 

s = np.loadtxt('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\data\\Practica 1\\Registro1Act1.txt')
fig, ax = plt.subplots()
ax.plot(s[:,0], s[:,2])
plt.show(block=False)

fig = MCI.FFT(s[:,2])
fig.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\pruebaFFT.png')

sr = 60000
l = 256
tr = 25
fig = MCI.psd(s[:,2], l, tr, sr)
fig.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\pruebaPSD.png')