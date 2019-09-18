import numpy as np
import matplotlib.pyplot as plt

#Archivos
sig = np.loadtxt('C:\\Users\HP desktop\Documents\Trimestre19P\MCIB\MCIB-19-P\data\Practica 1\Registro1Act1.txt')
t1 = int(1./sig[1,0])
t2 = int(1.25/sig[1,0])

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True, constrained_layout=True)
ax0.plot(sig[t1:t2,0], sig[t1:t2,1]); ax0.set(title = 'Respirograma', ylabel='V')
ax1.plot(sig[t1:t2,0], sig[t1:t2,2]); ax1.set(title = 'ECG', ylabel='V')
ax2.plot(sig[t1:t2,0], sig[t1:t2,3]); ax2.set(title = 'EMG', ylabel='V')
ax3.plot(sig[t1:t2,0], sig[t1:t2,4]); ax3.set(title = 'Oximetro', ylabel='V', xlabel='tiempo [min]')

plt.show()
fig.savefig('C:\\Users\HP desktop\Documents\Trimestre19P\MCIB\MCIB-19-P\images\Practica 1\p1.png')


#Onda cuadrada
k = 500
sr = 300
f = 5 

