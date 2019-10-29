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
    fig.set_size_inches(6, 10)
    fig.tight_layout()

def NextPowerOfTwo( NSamples ):
    return np.ceil ( np.log2( NSamples ) )

def loadData( fileName, Gain ):
        raw_time = []
        raw_samples = []

        try:
            with open( fileName, 'r', encoding='utf-8') as csvfile:
                c = csv.reader(csvfile)
                in_header = True
                lineCnt   = 0
                for row in c:
                    if len(row) <= 0:
                        continue;

                    if in_header == False:
                        raw_time.append(float(row[0]))
                        raw_samples.append(float(row[1]))

                    if lineCnt < 3:
                        in_header = True
                    else:
                        in_header = False

                    lineCnt += 1
        except UnicodeDecodeError:
            print ( self.fileName )

        #raw_samples = np.array ( raw_samples ) * Gain / 4.9e-3 
        raw_samples = (np.array ( raw_samples ) * Gain / 1000) / 4.9e-3
        raw_time = np.array ( raw_time )
        SampleRate = 1 / ( ( raw_time[2] - raw_time[1]) * 1e-3)
        Samples = raw_samples
        Times   = raw_time
        return Samples, Times, SampleRate

def current_reconstruction ( data, f, fd, alpha=0.9, KK=1):
    """ data : """
    K = 2 * np.pi * f / (2*fd)
    # Combfilter in feedback form
    dComp = KK * np.abs ( 1 / ( np.sqrt ( ( 1 + alpha**2) - 2 * alpha * np.cos( K ) ) ) )
    # Filtering
    dataFiltered = dComp * data

    return dataFiltered

def Combfilter ( f, fd, alpha=0.9, KK=.1):
    """ data : """
    K = 2 * np.pi * f / (2*fd)
    # Combfilter in feedback form
    dComp = KK * np.abs ( 1 / ( np.sqrt ( ( 1 + alpha**2) - 2 * alpha * np.cos( K ) ) ) )
    return dComp


fs = .25e6
N  = 1*8192
N2 = 2**int(NextPowerOfTwo(N))
fd = 14.4e3
Tend = 4e6/(2*fd)
Fend = 0.5*fs*1e-3
dt = 1/fs
df = fs/N2

t = np.arange ( 0, N*dt  , dt)
t2 = np.arange ( 0, N2*dt, dt)
f = np.fft.rfftfreq(N2, d=1./fs)

noise = np.random.normal(0,.01,N)

phase = 0.5*np.pi
yion = np.sin ( 2*np.pi*fd*t - phase)
y = phaseshifting ( phase, fd, t)**2 + yion# + noise
yn = phaseshifting ( phase, fd, t2)**2

# 0 is the mean of the normal distribution you are choosing from
# 1 is the standard deviation of the normal distribution
# 100 is the number of elements you get in array noise

dF = np.fft.rfft( y , N2 )
fFilt = current_reconstruction ( dF, f, fd )
fFilt *= 10**(20/-20)
nIndices = np.array ( ( 1.0, 2.0, 4.0, 8.0)) * fd / df  
wdths = 2 * np.abs ( fFilt [  np.int_(nIndices) ] ) / N2

#print ( wdths )

dr = np.fft.irfft ( fFilt, N2 ) / N2

yp = 0
idx = 1
for weigth in wdths:
	yp += weigth * np.cos ( idx * 2 * np.pi * fd * t + phase )
	idx *= 2

#ddr = signal.savgol_filter(dr, 7, 5) # window size 51, polynomial order 3
#dr /= 2.5*8
dr -= np.median(dr)

set_style()

fig, ax = plt.subplots(4)
ax[0].plot ( t*1e6, y, 'k-', label=r'Acoustic signal' )
ax[0].set_xlim ( 0, Tend )
ax[0].legend( loc=1)
ax[0].set_ylabel ( r'Pressure / $Pa$' )
ax[0].set_xlabel ( r'Time / $\mu$s' )
ax[0].text( -15, 1.2, r'a)' )

ax[1].plot ( f*1e-3, 2 * np.abs(dF) / N2, 'k-', label=r'Input spectrum' )
ax[1].plot ( f*1e-3, Combfilter(f, fd ) , '--', color = 'grey', label=r'Filter function' )
ax[1].set_xlim ( 0, Fend )
ax[1].legend( loc=1)
ax[1].set_ylabel ( r'Pressure / ARB' )
ax[1].set_xlabel ( r'Frequency / kHz' )
ax[1].text( -15, 1.1, r'b)' )

ax[2].plot ( f*1e-3, 2 * np.abs(fFilt) / N2 , 'k-', label=r'Comp filtered spectrum' )
ax[2].set_xlim ( 0, Fend )
ax[2].legend( loc=1)
ax[2].set_ylabel ( r'Current / ARB' )
ax[2].set_xlabel ( r'Frequency / kHz' )
ax[2].text( -15, 0.6, r'c)' )

ax[3].plot ( t2*1e6, dr*1e4, 'k-', label=r'Comp filtered signal' )
ax[3].plot ( t2*1e6, yn*np.amax(dr)*1e4, 'r-.', label=r'Scaled model' )
ax[3].set_xlim ( 0, Tend )
ax[3].legend( loc=1)
ax[3].set_ylabel ( r'Current / ARB' )
ax[3].set_xlabel ( r'Time / $\mu$s' )
ax[3].text( -15, 1.2, r'd)' )

#ax[0][1].plot ( t2*1e6, yn*np.amax(ddr), 'r-.', label=r'Scaled model' )
#ax[0][1].set_xlim ( 0, Tend )
fig.align_ylabels ( ax )
set_size(fig)
plt.savefig ( 'Reconstructed_current_scheme.png', dpi = 300 )
plt.savefig ( 'Reconstructed_current_scheme.pdf', dpi = 300 )

plt.show()
