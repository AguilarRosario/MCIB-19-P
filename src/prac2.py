import numpy as np 
import matplotlib.pyplot as plt 
import MCI 

sr = 320
t = np.linspace(0, 1, sr)
s = np.sin(2*np.pi*12*t)+np.sin(2*np.pi*17.5*t)

fig, ax = plt.subplots()
ax.plot(t, s)

plt.show(block=False)

MCI.FFT(s)