import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal

import sys
import csv
sys.path.append("Helper")

from HelpersSignal import *

import seaborn as sns

def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif',style="ticks")

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

def set_size(fig):
    fig.set_size_inches(6, 4)
    fig.tight_layout()

def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int( value / (2 * 14.4e3 ) )
    if N == 0:
        return "0"
    else:
        return r"${0}f_d$".format(2*N)

fd = 14.4e3
alpha = 0.9
KK = .1
df = 0.1
fend = fd * 2 * 5 + 1
f = np.arange ( 0, fend, 1)
K = 2 * np.pi * f / (2*fd)
# Combfilter in feedback form
dComp = KK * np.abs ( 1 / ( np.sqrt ( ( 1 + alpha**2) - 2 * alpha * np.cos( K ) ) ) )

xticks = np.arange ( 0, fend, 2*fd ) 

set_style()
fig, ax = plt.subplots(1)
ax.plot ( f, 20*np.log10(dComp), 'k-', label=r'$|H\left(\omega\right)|$' )
ax.legend( loc='best')
ax.set_ylabel ( r'$|H\left(\omega\right)|$ / dB' )
#ax.set_xlim ( f[0]*1e-3, f[-1]*1e-3 )
ax.set_xlabel ( r'Normalised Frequency' )
ax.set_xticks( xticks )
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

set_size(fig)
plt.savefig ( 'Combfilter_schematic.png', dpi = 300 )
plt.savefig ( 'Combfilter_schematic.pdf', dpi = 300 )
plt.show()