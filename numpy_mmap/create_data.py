import numpy as np
import matplotlib.pyplot as plt 

dx = 68.6e-6

NX = 500
NY = 500

x = np.arange ( (-NX//2)*dx, (NX//2)*dx, dx )
y = np.arange ( (-NX//2)*dx, (NY//2)*dx, dx )

xx, yy = np.meshgrid( y, x )
P = np.sin ( xx*yy ) * np.cos ( xx*yy )

fig, ax = plt.subplots(1)
ax.pcolor ( xx, yy, P )

plt.savefig ( 'test_data.png', dpi = 300)
np.savez ('example.npz', P = P, xx=xx, yy=yy )

plt.show()