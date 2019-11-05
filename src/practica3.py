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
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True)
ax0.plot(t, Resp); ax0.set(ylabel='Respirograma')
ax1.plot(t, ECG); ax1.set(ylabel='ECG')
ax2.plot(t, EMG); ax2.set(ylabel='EMG')
ax3.plot(t, pulso); ax3.set(xlabel='Tiempo [seg]',ylabel='Pulso')
fig.suptitle('Eliminación de tendencia lineal')
fig.savefig(datapath + '\\..\\images\\Practica 3\\Detrended.png')

# Filtrado de la señal 
b, a = signal.butter(4,2/sr*30)
ECG_f  = signal.filtfilt(b,a,ECG)
Resp_f  = signal.filtfilt(b,a,Resp)
pulso_f  = signal.filtfilt(b,a,pulso)

fig, (ax0, ax1, ax2 ,ax3) = plt.subplots(nrows=4, constrained_layout=True)
ax0.plot(t, Resp_f); ax0.set(ylabel='Respirograma')
ax1.plot(t, ECG_f); ax1.set(ylabel='ECG')
ax2.plot(t, EMG); ax2.set(ylabel='EMG')
ax3.plot(t, pulso_f); ax3.set(ylabel='Pulso', xlabel='Tiempo [seg]')
fig.suptitle('Eliminación de tendencia lineal y filtrado')
fig.savefig(datapath + '\\..\\images\\Practica 3\\Detrendedyfilt.png')

ini = np.array([])
ini = np.append(ini, [Resp_f[0:15*sr], ECG_f[0:15*sr], EMG[0:15*sr], pulso_f[0:15*sr]]).reshape(4, 15000) 
ada = np.array([])
ada = np.append(ada, [Resp_f[38*sr:53*sr], ECG_f[38*sr:53*sr], EMG[38*sr:53*sr], pulso_f[38*sr:53*sr]]).reshape(4, 15000)  
eje = np.array([])
eje = np.append(eje, [Resp_f[60*sr:75*sr], ECG_f[60*sr:75*sr], EMG[60*sr:75*sr], pulso_f[60*sr:75*sr]]).reshape(4, 15000)  
fin = np.array([])
fin = np.append(fin, [Resp_f[90*sr:105*sr], ECG_f[90*sr:105*sr], EMG[90*sr:105*sr], pulso_f[90*sr:105*sr]]).reshape(4, 15000)  

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True)
ax0.plot(ini[1,:]); ax0.set(ylabel='Inicial')
ax1.plot(ada[1,:]); ax1.set(ylabel='Adaptacion')
ax2.plot(eje[1,:]); ax2.set(ylabel='Ejercicio')
ax3.plot(fin[1,:]); ax3.set(ylabel='Final')
fig.suptitle('ECG para los 4 segmentos')
fig.savefig(datapath + '\\..\\images\\Practica 3\\Segmentos.png')

# Dejar la señal en un intervalo entre [-1,1]
iniree = (((ini-np.min(ini, axis=1)[:,np.newaxis])*2)/(np.max(ini-np.min(ini, axis=1)[:,np.newaxis], axis=1)[:,np.newaxis]))-1
adaree = (((ada-np.min(ada, axis=1)[:,np.newaxis])*2)/(np.max(ada-np.min(ada, axis=1)[:,np.newaxis], axis=1)[:,np.newaxis]))-1
ejeree = (((eje-np.min(eje, axis=1)[:,np.newaxis])*2)/(np.max(eje-np.min(eje, axis=1)[:,np.newaxis], axis=1)[:,np.newaxis]))-1
finree = (((fin-np.min(fin, axis=1)[:,np.newaxis])*2)/(np.max(fin-np.min(fin, axis=1)[:,np.newaxis], axis=1)[:,np.newaxis]))-1
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True)
ax0.plot(iniree[0,:]); ax0.set(ylabel='Respirograma')
ax1.plot(adaree[1,:]); ax1.set(ylabel='ECG')
ax2.plot(ejeree[2,:]); ax2.set(ylabel='EMG')
ax3.plot(finree[3,:]); ax3.set(ylabel='Pulso')
fig.suptitle('Reescalamiento de [-1,1] de segmento de actividad para 4 señales')
fig.savefig(datapath + '\\..\\images\\Practica 3\\Reescalamiento.png')

#z-score, obtenemos la media y varianza con los estimadores
iniz = MCI.zscore(ini)
adaz = MCI.zscore(ada)
ejez = MCI.zscore(eje)
finz = MCI.zscore(fin)
fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=4, constrained_layout=True)
ax0.plot(ejez[0,:]); ax0.set(ylabel='Respirograma')
ax1.plot(ejez[1,:]); ax1.set(ylabel='ECG')
ax2.plot(ejez[2,:]); ax2.set(ylabel='EMG')
ax3.plot(ejez[3,:]); ax3.set(ylabel='Pulso')
fig.suptitle('Reescalamiento zscore de segmento de actividad para 4 señales')
fig.savefig(datapath + '\\..\\images\\Practica 3\\zscore.png')

#Histogramas para señal de respirograma sin reescalar
nbins = int(round(np.sqrt(ini[0,:].shape[0])))
bins0, hist0 = MCI.histograma(ini[0,:], nbins)
bins1, hist1 = MCI.histograma(ada[0,:], nbins)
bins2, hist2 = MCI.histograma(eje[0,:], nbins)
bins3, hist3 = MCI.histograma(fin[0,:], nbins)
fig,ax0 = plt.subplots(nrows=1, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.3, label='Etapa inicial');ax0.bar(bins1, hist1, width=0.05, alpha=0.3, label='Adaptación');ax0.bar(bins2, hist2, width=0.05, alpha=0.3, label='Ejercicio');ax0.bar(bins3, hist3, width=0.05, alpha=0.3, label='Etapa final');plt.legend()
fig.suptitle('Histogramas del respirograma sin reescalar')
fig.savefig(datapath + '\\..\\images\\Practica 3\\HistResp.png')

#Histogramas para la señal de ECG sin reescalar
bins0, hist0 = MCI.histograma(ini[1,:], nbins)
bins1, hist1 = MCI.histograma(ada[1,:], nbins)
bins2, hist2 = MCI.histograma(eje[1,:], nbins)
bins3, hist3 = MCI.histograma(fin[1,:], nbins)
fig,ax0 = plt.subplots(nrows=1, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.3, label='Etapa inicial');ax0.bar(bins1, hist1, width=0.05, alpha=0.3, label='Adaptación');ax0.bar(bins2, hist2, width=0.05, alpha=0.3, label='Ejercicio');ax0.bar(bins3, hist3, width=0.05, alpha=0.3, label='Etapa final');plt.legend()
fig.suptitle('Histogramas del ECG sin reescalar')
fig.savefig(datapath + '\\..\\images\\Practica 3\\HistECG.png')

#Histogramas para la señal de EMG sin reescalar
bins0, hist0 = MCI.histograma(ini[2,:], nbins)
bins1, hist1 = MCI.histograma(ada[2,:], nbins)
bins2, hist2 = MCI.histograma(eje[2,:], nbins)
bins3, hist3 = MCI.histograma(fin[2,:], nbins)
fig,ax0 = plt.subplots(nrows=1, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.2, label='Etapa inicial');ax0.bar(bins1, hist1, width=0.05, alpha=0.2, label='Adaptación');ax0.bar(bins2, hist2, width=0.05, alpha=1, label='Ejercicio');ax0.bar(bins3, hist3, width=0.05, alpha=0.3, label='Etapa final');plt.legend()
fig.suptitle('Histogramas del EMG sin reescalar')
fig.savefig(datapath + '\\..\\images\\Practica 3\\HistEMG.png')

#Histogramas para la señal de Onda de pulso sin reescalar
bins0, hist0 = MCI.histograma(ini[3,:], nbins)
bins1, hist1 = MCI.histograma(ada[3,:], nbins)
bins2, hist2 = MCI.histograma(eje[3,:], nbins)
bins3, hist3 = MCI.histograma(fin[3,:], nbins)
fig,ax0 = plt.subplots(nrows=1, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.3, label='Etapa inicial');ax0.bar(bins1, hist1, width=0.05, alpha=0.3, label='Adaptación');ax0.bar(bins2, hist2, width=0.05, alpha=1, label='Ejercicio');ax0.bar(bins3, hist3, width=0.05, alpha=0.3, label='Etapa final');plt.legend()
fig.suptitle('Histogramas de la señal de pulso sin reescalar')
fig.savefig(datapath + '\\..\\images\\Practica 3\\HistPul.png')

#Histogramas para señal de respirograma
nbins = int(round(np.sqrt(ini[0,:].shape[0])))
bins0, hist0 = MCI.histograma(iniree[0,:], nbins)
bins1, hist1 = MCI.histograma(adaree[0,:], nbins)
bins2, hist2 = MCI.histograma(ejeree[0,:], nbins)
bins3, hist3 = MCI.histograma(finree[0,:], nbins)
fig,ax0 = plt.subplots(nrows=1, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.3, label='Etapa inicial');ax0.bar(bins1, hist1, width=0.05, alpha=0.3, label='Adaptación');ax0.bar(bins2, hist2, width=0.05, alpha=0.3, label='Ejercicio');ax0.bar(bins3, hist3, width=0.05, alpha=0.3, label='Etapa final');plt.legend()
fig.suptitle('Histogramas del respirograma reescalado')
fig.savefig(datapath + '\\..\\images\\Practica 3\\HistRespRee.png')

#Histogramas para la señal de ECG
bins0, hist0 = MCI.histograma(iniree[1,:], nbins)
bins1, hist1 = MCI.histograma(adaree[1,:], nbins)
bins2, hist2 = MCI.histograma(ejeree[1,:], nbins)
bins3, hist3 = MCI.histograma(finree[1,:], nbins)
fig,ax0 = plt.subplots(nrows=1, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.3, label='Etapa inicial');ax0.bar(bins1, hist1, width=0.05, alpha=0.3, label='Adaptación');ax0.bar(bins2, hist2, width=0.05, alpha=0.3, label='Ejercicio');ax0.bar(bins3, hist3, width=0.05, alpha=0.3, label='Etapa final');plt.legend()
fig.suptitle('Histogramas del ECG reescalado')
fig.savefig(datapath + '\\..\\images\\Practica 3\\HistECGRee.png')

#Histogramas para la señal de EMG
bins0, hist0 = MCI.histograma(iniree[2,:], nbins)
bins1, hist1 = MCI.histograma(adaree[2,:], nbins)
bins2, hist2 = MCI.histograma(ejeree[2,:], nbins)
bins3, hist3 = MCI.histograma(finree[2,:], nbins)
fig,ax0 = plt.subplots(nrows=1, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.2, label='Etapa inicial');ax0.bar(bins1, hist1, width=0.05, alpha=0.2, label='Adaptación');ax0.bar(bins2, hist2, width=0.05, alpha=1, label='Ejercicio');ax0.bar(bins3, hist3, width=0.05, alpha=0.3, label='Etapa final');plt.legend()
fig.suptitle('Histogramas del EMG reescalado')
fig.savefig(datapath + '\\..\\images\\Practica 3\\HistEMGRee.png')

#Histogramas para la señal de Onda de pulso
bins0, hist0 = MCI.histograma(iniree[3,:], nbins)
bins1, hist1 = MCI.histograma(adaree[3,:], nbins)
bins2, hist2 = MCI.histograma(ejeree[3,:], nbins)
bins3, hist3 = MCI.histograma(finree[3,:], nbins)
fig,ax0 = plt.subplots(nrows=1, sharex=True)
ax0.bar(bins0, hist0, width=0.05, alpha=0.3, label='Etapa inicial');ax0.bar(bins1, hist1, width=0.05, alpha=0.3, label='Adaptación');ax0.bar(bins2, hist2, width=0.05, alpha=1, label='Ejercicio');ax0.bar(bins3, hist3, width=0.05, alpha=0.3, label='Etapa final');plt.legend()
fig.suptitle('Histogramas de la señal de pulso reescalada')
fig.savefig(datapath + '\\..\\images\\Practica 3\\HistPulRee.png')

# Obtenemos segmentos de tiempo de las cuatro señales, la obtener su matriz de correación
segt = 3
resp1 = MCI.epochs(Resp_f, segt, sr)
ECG1  = MCI.epochs(ECG_f, segt, sr)
EMG1  = MCI.epochs(EMG, segt, sr)
pulso1 = MCI.epochs(pulso_f, segt, sr)

r1 = MCI.coef_corr(resp1, ECG1)
r2 = MCI.coef_corr(resp1, EMG1)
r3 = MCI.coef_corr(resp1, pulso1)
r4 = MCI.coef_corr(ECG1, EMG1)
r5 = MCI.coef_corr(ECG1, pulso1)
r6 = MCI.coef_corr(EMG1, pulso1)

fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, constrained_layout=True)
ax0.plot(r1); ax0.set(ylabel='resp-ECG')
ax1.plot(r2); ax1.set(ylabel='resp-EMG')
ax2.plot(r3); ax2.set(ylabel='resp-pulso')
ax3.plot(r4); ax3.set(ylabel='ECG-EMG')
ax4.plot(r5); ax4.set(ylabel='ECG-pulso')
ax5.plot(r6); ax5.set(ylabel='EMG-pulso')
fig.suptitle('Epocas vs Coeficiente de correlacion')
fig.savefig(datapath + '\\..\\images\\Practica 3\\Correlacion.png')

#Coherencia 
f, co11 = MCI.cohe(ini[0,:], ini[1, :], sr)
f, co21 = MCI.cohe(ada[0,:], ada[1,:], sr)
f, co31 = MCI.cohe(eje[0,:], eje[1,:], sr)
f, co41 = MCI.cohe(fin[0,:], fin[1, :], sr)
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True, sharex=True)
ax0.plot(f, co11); ax0.set(ylabel='Inicial')
ax1.plot(f, co21); ax1.set(ylabel='Adaptacion')
ax2.plot(f, co31); ax2.set(ylabel='Ejercicio')
ax3.plot(f, co41); ax3.set(xlabel='Hz', ylabel='Final')
fig.suptitle('Coherencia entre el respirograma y el ECG')
fig.savefig(datapath + '\\..\\images\\Practica 3\\CohResECG.png')

f, co12 = MCI.cohe(ini[0,:], ini[2, :], sr)
f, co22 = MCI.cohe(ada[0,:], ada[2,:], sr)
f, co32 = MCI.cohe(eje[0,:], eje[2,:], sr)
f, co42 = MCI.cohe(fin[0,:], fin[2, :], sr)
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True, sharex=True)
ax0.plot(f, co12); ax0.set(ylabel='Inicial')
ax1.plot(f, co22); ax1.set(ylabel='Adaptacion')
ax2.plot(f, co32); ax2.set(ylabel='Ejercicio')
ax3.plot(f, co42); ax3.set(xlabel='Hz', ylabel='Final')
fig.suptitle('Coherencia entre el respirograma y el EMG')
fig.savefig(datapath + '\\..\\images\\Practica 3\\CohResEMG.png')

f, co13 = MCI.cohe(ini[0,:], ini[3, :], sr)
f, co23 = MCI.cohe(ada[0,:], ada[3,:], sr)
f, co33 = MCI.cohe(eje[0,:], eje[3,:], sr)
f, co43 = MCI.cohe(fin[0,:], fin[3, :], sr)
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True, sharex=True)
ax0.plot(f, co13); ax0.set(ylabel='Inicial')
ax1.plot(f, co23); ax1.set(ylabel='Adaptacion')
ax2.plot(f, co33); ax2.set(ylabel='Ejercicio')
ax3.plot(f, co43); ax3.set(xlabel='Hz', ylabel='Final')
fig.suptitle('Coherencia entre el respirograma y la señal de pulso')
fig.savefig(datapath + '\\..\\images\\Practica 3\\CohResPul.png')

f, co14 = MCI.cohe(ini[1,:], ini[2, :], sr)
f, co24 = MCI.cohe(ada[1,:], ada[2,:], sr)
f, co34 = MCI.cohe(eje[1,:], eje[2,:], sr)
f, co44 = MCI.cohe(fin[1,:], fin[2, :], sr)
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True, sharex=True)
ax0.plot(f, co14); ax0.set(ylabel='Inicial')
ax1.plot(f, co24); ax1.set(ylabel='Adaptacion')
ax2.plot(f, co34); ax2.set(ylabel='Ejercicio')
ax3.plot(f, co44); ax3.set(xlabel='Hz', ylabel='Final')
fig.suptitle('Coherencia entre el ECG y el EMG')
fig.savefig(datapath + '\\..\\images\\Practica 3\\CohECGEMG.png')

f, co15 = MCI.cohe(ini[1,:], ini[3, :], sr)
f, co25 = MCI.cohe(ada[1,:], ada[3,:], sr)
f, co35 = MCI.cohe(eje[1,:], eje[3,:], sr)
f, co45 = MCI.cohe(fin[1,:], fin[3, :], sr)
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True, sharex=True)
ax0.plot(f, co15); ax0.set(ylabel='Inicial')
ax1.plot(f, co25); ax1.set(ylabel='Adaptacion')
ax2.plot(f, co35); ax2.set(ylabel='Ejercicio')
ax3.plot(f, co45); ax3.set(xlabel='Hz', ylabel='Final')
fig.suptitle('Coherencia entre el ECG y la señal de pulso')
fig.savefig(datapath + '\\..\\images\\Practica 3\\CohECGPulso.png')

f, co16 = MCI.cohe(ini[2,:], ini[3, :], sr)
f, co26 = MCI.cohe(ada[2,:], ada[3,:], sr)
f, co36 = MCI.cohe(eje[2,:], eje[3,:], sr)
f, co46 = MCI.cohe(fin[2,:], fin[3, :], sr)
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, constrained_layout=True, sharex=True)
ax0.plot(f, co16); ax0.set(ylabel='Inicial')
ax1.plot(f, co26); ax1.set(ylabel='Adaptacion')
ax2.plot(f, co36); ax2.set(ylabel='Ejercicio')
ax3.plot(f, co46); ax3.set(xlabel='Hz', ylabel='Final')
fig.suptitle('Coherencia entre el EMG y la señal de pulso')
fig.savefig(datapath + '\\..\\images\\Practica 3\\CohEMGPul.png')

#Detector de picos
ECG_peaks, p = signal.find_peaks(ECG_f, height=1)
Pulso_peaks, p = signal.find_peaks(pulso_f, height=0.1, distance=500)
fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)
ax0.plot(ECG_f); ax0.set(ylabel='ECG')
ax0.plot(ECG_peaks, ECG_f[ECG_peaks], "x")
ax1.plot(pulso_f); ax1.set(ylabel='Pulso')
ax1.plot(Pulso_peaks, pulso_f[Pulso_peaks], "x")
fig.suptitle('Deteccion de picos')
fig.savefig(datapath + '\\..\\images\\Practica 3\\DeteccionPeaks.png')

#Frecuencia
FC = (1/(np.diff(ECG_peaks)/sr))*60
FP = (1/(np.diff(Pulso_peaks)/sr))*60
fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)
ax0.plot(FC); ax0.set(ylabel='Frecuencia ECG')
ax1.plot(FP); ax1.set(ylabel='Frecuencia Pulso')
fig.suptitle('Frecuencia cardiaca')
fig.savefig(datapath + '\\..\\images\\Practica 3\\Frecuencia.png')

FC = np.delete(FC, -1)
cor = MCI.coef_corr(FC[np.newaxis, :], FP[np.newaxis, :])
print('Coefieciente de correlacion entre las señales: %.3f' %cor)

# Obtenemos envolvente y área de la señal de EMG para sacar la correlación con otras señales
EMGenv, EMGA= MCI.envolvente(EMG, sr)
inienv, iniA= MCI.envolvente(iniree[2,:], sr)
adaenv, adaA= MCI.envolvente(adaree[2,:], sr)
ejeenv, ejeA= MCI.envolvente(ejeree[2,:], sr)
finenv, finA= MCI.envolvente(finree[2,:], sr)
print("""Área de EMG
Etapa inicial: %.4f
Etapa adaptación: %.4f
Etapa ejercicio: %.4f
Etapa final: %.4f
EMG: %.4f""" %(iniA, adaA, ejeA, finA, EMGA))
fig, ax = plt.subplots()
ax.plot(EMGenv)
plt.show()