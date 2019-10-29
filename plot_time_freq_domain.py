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


Gain = 10**(-24/-20)
yin, t, fs = loadData ( 'Argon/0Grad_22.csv', Gain )
t *= 1e-3
N  = t.size
N2 = 2**int(NextPowerOfTwo(N))

sos = sigpy.butter(10, 100, 'hp', fs=fs, output='sos')
yin = sigpy.sosfilt(sos, yin)
fd = PitchDetechtion(yin, fs )

Tend = 10e6/(fd)
Fend = 0.5*fs*1e-3
dt = 1/fs
df = fs/N2

f = np.fft.rfftfreq(N2, d=1./fs)

dF = 2 * np.abs ( np.fft.rfft( yin , N2 ) ) / N2

Freqs = np.array ( ( 1.0, 2.0, 4.0, 6.0)) * fd / df 
Labels = np.array ( (r'$f_d$', r'$2f_d$', r'$4f_d$', r'$6f_d$') )
Shifts = np.array ( (0.0,-1.0,-1.0,-3.0) )
#Amps = dF [  np.int_(Freqs) ] 

set_style()

fig, ax = plt.subplots(2)
ax[0].plot ( t*1e6, yin*1e3, 'k-', label=r'Acoustic signal' )
ax[0].set_xlim ( 0, Tend )

ax[0].set_ylabel ( r'Pressure / mPa' )
ax[0].set_xlabel ( r'Time / $\mu$s' )
ax[0].text( -35, 540, r'a)' )

ax[1].plot ( f*1e-3, dF*1e3, 'k-', label=r'Input spectrum' )
ax[1].set_xlim ( 0, Fend )

for (iFreq, iLabel, iShift) in zip(Freqs, Labels, Shifts):
	ax[1].text( iFreq*df*1e-3 -1.5 + iShift, 20, iLabel )

ax[1].set_ylabel ( r'Pressure / mPa' )
ax[1].set_xlabel ( r'Frequency / kHz' )
ax[1].set_ylim ( 0, 25 )
ax[1].text( -5, 27, r'b)' )

fig.align_ylabels ( ax )
set_size(fig)
plt.savefig ( 'Acoustic_Emission_Argon.png', dpi = 300 )
plt.savefig ( 'Acoustic_Emission_Argon.pdf', dpi = 300 )

plt.show()
