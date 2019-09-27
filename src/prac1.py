import numpy as np
import matplotlib.pyplot as plt

#Archivos
sig = np.loadtxt('C:\\Users\\HP desktop\\Documents\\Trimestre19P\\MCIB\MCIB-19-P\\data\\Practica 1\\Registro1Act1.txt')
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
sr = 300
f = 5 
t = np.linspace(0,1,sr)[np.newaxis]   #vector de tiempo 

n = int(input("""Dame el número de armónicos: """))
F = np.linspace(1,2*n-1,n)
F = F[:,np.newaxis]

# Aproximación de onda cuadrada
S = F*(2*np.pi*t*f)
S = 1/F*(np.sin(S))
S = 4*(np.sum(S, axis=0))/np.pi

fig,ax = plt.subplots()
ax.plot(t[0,:],S); ax.set(xlabel='Tiempo [s]', ylabel='Amplitud', title='Aproximación de onda cuadrada para %d armónicos'%n)
plt.show(block=False)

# Aproximación de onda triangular 
s2 = F*(2*np.pi*t*f)
s2 = np.cos(s2)
k = 1/(F**2)
s2 = 8*np.sum(k*s2,axis=0)/np.pi**2

fig,ax = plt.subplots()
ax.plot(t[0,:],s2); ax.set(xlabel='Tiempo [s]', ylabel='Amplitud', title='Aproximación de onda triangular para %d armónicos'%n)
plt.show()