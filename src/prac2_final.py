import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import MCI 

s0 = np.loadtxt('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\data\\Practica 1\\Registro1Act1.txt')
s1 = np.loadtxt('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\data\\Practica 1\\Registro6Act1.txt')
s2 = np.loadtxt('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\data\\Practica 1\\Registro1Act2.txt')
s3 = np.loadtxt('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\data\\Practica 1\\Registro5Act2.txt')
sr = 1000

fig0 = MCI.FFT(s0[:,1], sr)
fig0.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTResp1.png')
fig1 = MCI.FFT(s1[:,1], sr)
fig1.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTResp2.png')
fig2 = MCI.FFT(s2[:,1], sr)
fig2.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTResp3.png')
fig3 = MCI.FFT(s3[:,1], sr)
fig3.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTResp4.png')
fig4 = MCI.FFT(s0[:,2], sr)
fig4.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTECG1.png')
fig5 = MCI.FFT(s1[:,2], sr)
fig5.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTECG2.png')
fig6 = MCI.FFT(s2[:,2], sr)
fig6.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTECG3.png')
fig7 = MCI.FFT(s3[:,2], sr)
fig7.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTECG4.png')
fig8 = MCI.FFT(s0[:,3], sr)
fig8.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTEMG1.png')
fig9 = MCI.FFT(s1[:,3], sr)
fig9.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTEMG2.png')
fig10 = MCI.FFT(s2[:,3], sr)
fig10.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTEMG3.png')
fig11 = MCI.FFT(s3[:,3], sr)
fig11.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTEMG4.png')
fig12 = MCI.FFT(s0[:,4], sr)
fig12.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTPulso1.png')
fig13 = MCI.FFT(s1[:,4], sr)
fig13.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTPulso2.png')
fig14 = MCI.FFT(s2[:,4], sr)
fig14.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTPulso3.png')
fig15 = MCI.FFT(s3[:,4], sr)
fig15.savefig('C:\\Users\\Mouzhroq\\Desktop\\Python\\MCIB-19-P\\images\\Practica 2\\FFTPulso4.png')
