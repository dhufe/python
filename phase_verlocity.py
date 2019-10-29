import matplotlib.pyplot as plt
import numpy as np

dT = 1.5e-6#np.arange(0,10,0.1)*1e-6
d = 5e-3
vair = 2100
phi = np.arange(0.05,0.5,0.01)*np.pi
vpla = 343

vphase = (vpla*np.sqrt((dT/d)**2.0 - (2*dT/d)*np.cos(phi)/vair +(1/vair)**2.0))**1

fig,ax = plt.subplots(1)
ax.plot(phi/(0.5*np.pi), vphase/(2*d))

plt.show()