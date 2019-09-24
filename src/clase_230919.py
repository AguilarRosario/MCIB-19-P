import numpy             as np 
import matplotlib.pyplot as plt

sr = 320 
L = 5

F = np.array([])
F = np.append(F,5)
F = np.append(F,30)

t = np.linspace(0,L,L*sr)[:,np.newaxis] #Agregar un nuevo eje
F = F[np.newaxis] #Agregar un nuevo eje

P = (2*np.pi*t*F)
P = np.sin(P)
P = np.sum(P,axis=1)+2

fig, ax = plt.subplots()
ax.plot(t[:300],P[:300]); ax.set(xlabel='Tiempo [seg]', ylabel='Amplitud')
#plt.show(block=False)

#Filtrado
from scipy import signal

#Filtro pasabajas 
# b, a = signal.butter(4,2/sr*15)
# y_f  = signal.lfilter(b,a,P)
# y_ff = signal.lfilter(b,a,y_f[::-1])[::-1]

# fig,ax = plt.subplots()
# ax.plot(t[:300],P[:300],linewidth=2)
# ax.plot(t[:300],y_f[:300],linewidth=2)
# ax.plot(t[:300],y_ff[:300],linewidth=2)
# plt.show()

# #Filtro pasa altas
# b, a = signal.butter(4,2/sr*15, btype='high')
# y_pa  = signal.lfilter(b,a,P)
# y_paff = signal.lfilter(b,a,y_pa[::-1])[::-1]

# fig,ax = plt.subplots()
# ax.plot(t[:300],P[:300],linewidth=2)
# ax.plot(t[:300],y_pa[:300],linewidth=2)
# ax.plot(t[:300],y_paff[:300],'purple',linewidth=2)
# plt.show()

# Función para filtrar con FIR (firwin)
# N = 40
# b = signal.firwin(N, 2/sr*15)
# y_fir = signal.lfilter(b,[1],P)

# fig, ax = plt.subplots()
# ax.plot(t[:300],P[:300])
# ax.plot(t[:300],y_fir[N//2:300+N//2])
# plt.show()

#Implementar filtro con promedio móvil
def prom_mov(N,x):
    l = len(x)
    y_f = np.array([])
    for n in range(N+1,l):
        y = x[n] - np.sum(x[n-N:n])/(N+1)
        y_f = np.append(y_f,y)

    return y_f


#Filtrado con promedio móvil 
P_f = prom_mov(21,P)
fig, ax = plt.subplots()
ax.plot(P[:300])
ax.plot(-P_f[:300])
plt.show()

