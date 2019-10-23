import numpy             as np 
import matplotlib.pyplot as plt 
import MCI
import os 
import scipy.signal      as signal

datapath = os.path.abspath('')
reg1 = np.loadtxt(datapath + '\\..\\data\\Practica 1\\Registro1Act1.txt')
sr = 1000

L = 16384
t = reg1[:,0]*60
Resp  = MCI.detrend(reg1[:,1], t, L)
ECG   = MCI.detrend(reg1[:,2], t, 1024)
EMG   = MCI.detrend(reg1[:,3], t, 1024)
pulso = MCI.detrend(reg1[:,4], t, 1024)

# Señales con eliminación de tendencia lineal
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)
ax0.plot(t, Resp)
ax1.plot(t, ECG)
ax2.plot(t, EMG)
ax3.plot(t, pulso)
#plt.show()

# Filtrado de la señal 
 
b, a = signal.butter(4,2/sr*30)
ECG_f  = signal.filtfilt(b,a,ECG)
Resp_f  = signal.filtfilt(b,a,Resp)
pulso_f  = signal.filtfilt(b,a,pulso)

fig, (ax0, ax1, ax3) = plt.subplots(nrows=3)
ax0.plot(t, Resp_f)
ax1.plot(t, ECG_f)
ax3.plot(t, pulso_f)
#plt.show()

ini = np.array([])
ini = np.append(ini, [Resp_f[0:15*sr], ECG_f[0:15*sr], EMG[0:15*sr], pulso_f[0:15*sr]]).reshape(4, 15000) 
ada = np.array([])
ada = np.append(ada, [Resp_f[38*sr:53*sr], ECG_f[38*sr:53*sr], EMG[38*sr:53*sr], pulso_f[38*sr:53*sr]]).reshape(4, 15000)  
eje = np.array([])
eje = np.append(eje, [Resp_f[60*sr:75*sr], ECG_f[60*sr:75*sr], EMG[60*sr:75*sr], pulso_f[60*sr:75*sr]]).reshape(4, 15000)  
fin = np.array([])
fin = np.append(fin, [Resp_f[90*sr:105*sr], ECG_f[90*sr:105*sr], EMG[90*sr:105*sr], pulso_f[90*sr:105*sr]]).reshape(4, 15000)  

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)
ax0.plot(eje[0,:])
ax1.plot(eje[1,:])
ax2.plot(eje[2,:])
ax3.plot(eje[3,:])
# plt.show()

# Dejar la señal en un intervalo entre [-1,1]
iniree = (((ini-np.min(ini, axis=1)[:,np.newaxis])*2)/(np.max(ini-np.min(ini, axis=1)[:,np.newaxis], axis=1)[:,np.newaxis]))-1
adaree = (((ada-np.min(ada, axis=1)[:,np.newaxis])*2)/(np.max(ada-np.min(ada, axis=1)[:,np.newaxis], axis=1)[:,np.newaxis]))-1
ejeree = (((eje-np.min(eje, axis=1)[:,np.newaxis])*2)/(np.max(eje-np.min(eje, axis=1)[:,np.newaxis], axis=1)[:,np.newaxis]))-1
finree = (((fin-np.min(fin, axis=1)[:,np.newaxis])*2)/(np.max(fin-np.min(fin, axis=1)[:,np.newaxis], axis=1)[:,np.newaxis]))-1
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)
ax0.plot(ejeree[0,:])
ax1.plot(ejeree[1,:])
ax2.plot(ejeree[2,:])
ax3.plot(ejeree[3,:])
# plt.show()

#z-score, obtenemos la media y varianza con los estimadores
def zscore(s):
    media = (1/s.shape[1])*(np.sum(s, axis=1))
    sigma = np.sqrt((1/(s.shape[1]-1))*(np.sum((s-media[:,np.newaxis])**2, axis=1)))
    z     = (s-media[:,np.newaxis])/sigma[:,np.newaxis]
    return z

iniz = zscore(ini)
adaz = zscore(ada)
ejez = zscore(eje)
finz = zscore(fin)
fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=4)
ax0.plot(ejez[0,:])
ax1.plot(ejez[1,:])
ax2.plot(ejez[2,:])
ax3.plot(ejez[3,:])
# plt.show()

#Histogramas para señal de respirograma en la etapa inicial (esto está mal)
nbins = int(round(np.sqrt(ini[0,:].shape[0])))
bins0, hist0       = MCI.histograma(ini[0,:], nbins)
bins0ree, hist0ree = MCI.histograma(iniree[0,:], nbins)
bins0z, hist0z     = MCI.histograma(iniz[0,:], nbins)
bins1, hist1       = MCI.histograma(ada[0,:], nbins)
bins1ree, hist1ree = MCI.histograma(adaree[0,:], nbins)
bins1z, hist1z     = MCI.histograma(adaz[0,:], nbins)
bins2, hist2       = MCI.histograma(eje[0,:], nbins)
bins2ree, hist2ree = MCI.histograma(ejeree[0,:], nbins)
bins2z, hist2z     = MCI.histograma(ejez[0,:], nbins)
bins3, hist3       = MCI.histograma(fin[0,:], nbins)
bins3ree, hist3ree = MCI.histograma(finree[0,:], nbins)
bins3z, hist3z     = MCI.histograma(finz[0,:], nbins)
fig,(ax0,ax1,ax2,ax3) = plt.subplots(nrows=4, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.5); ax0.bar(bins0ree, hist0ree, width=0.05, alpha=0.5); ax0.bar(bins0z, hist0z, width=0.05, alpha=0.5)
ax1.bar(bins1, hist1, width=0.05, alpha=0.5); ax1.bar(bins1ree, hist1ree, width=0.05, alpha=0.5); ax1.bar(bins1z, hist1z, width=0.05, alpha=0.5)
ax2.bar(bins2, hist2, width=0.05, alpha=0.5); ax2.bar(bins2ree, hist2ree, width=0.05, alpha=0.5); ax2.bar(bins2z, hist2z, width=0.05, alpha=0.5)
ax3.bar(bins3, hist3, width=0.05, alpha=0.5); ax3.bar(bins3ree, hist3ree, width=0.05, alpha=0.5); ax3.bar(bins3z, hist3z, width=0.05, alpha=0.5)
plt.show()





