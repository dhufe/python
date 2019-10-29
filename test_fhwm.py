import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("Helper")

from HelpersAcoustic import *
from HelpersSignal import *

pulsesig, TPulse, t = us_impulse( 0.125e6, 100e6 , NOrder = 4 )
min_left_idx, min_right_idx, tp = fhwm ( t, np.abs(pulsesig), trsh = 0.1 )

fig, ax = plt.subplots(1)
ax.plot ( t, pulsesig, 'b-' )
ax.plot ( t, np.abs(pulsesig) , 'r-.' )
ax.plot ( t[min_left_idx], np.abs(pulsesig)[min_left_idx] , 'kx' )
ax.plot ( t[min_right_idx], np.abs(pulsesig)[min_right_idx] , 'kx' )

plt.show()
