import numpy as np
import scipy.signal as sigpy
import matplotlib.pyplot as plt

def rect(T):
    """create a centered rectangular pulse of width $T"""
    return lambda t: (-T/2 <= t) & (t < T/2)

def pulse_train(t, at, shape):
    """create a train of pulses over $t at times $at and shape $shape"""
    return np.sum(shape(t - at[:,np.newaxis]), axis=0)

def phaseshifting ( alpha, f, t , phase = 0):
    """ Creates a phase controlled waveform. alpha describes the switching point """
    omega = np.pi * 2 * f
    w = alpha / omega
    d = np.arange ( .5/f, t[-1], .5/f )
    d = d - .5*w - phase/omega;
    y = np.sin( omega * t + phase);

    y2 = pulse_train(
        t=t,                           # time domain
        at=d,                          # times of pulses
        shape=rect(w)                  # shape of pulse
    )
    return y*y2

def PitchDetechtion(Samples,
          fs,
          window=np.hanning,
          partials=5,
          plot=False):

    N = Samples.size
    N2 = int(2**np.log2(N))
    wdw = window(N)
    frequencies = np.fft.rfftfreq(N2, 1 / fs)

    spectrum = np.fft.rfft( wdw*Samples, N2 )
    # Downsample spectrum.
    spectra = []
    for n in range(1, partials + 1):
        s = sigpy.resample(spectrum, len(spectrum) // n)
        spectra.append(s)

    # Truncate to most downsampled spectrum.
    l = min(len(s) for s in spectra)
    a = np.zeros((len(spectra), l), dtype=spectrum.dtype)
    for i, s in enumerate(spectra):
        a[i] += s[:l]

    # Multiply spectra per frequency bin.
    hps = np.product(np.abs(a), axis=0)

    #kernel = signal.gaussian(9, 1)
    #hps = signal.fftconvolve(hps, kernel, mode='same')
    #peaks = sp.signal.find_peaks_cwt(np.abs(hps), np.arange(1, 3))
    # Pick largest peak, it's likely f0.
    peak = np.argmax(hps)
    f0 = frequencies[peak]

    if plot:
        # Plot partial magnitudes individually.
        for s, ax in zip(spectra, plt.subplots(len(spectra), sharex=True)[1]):
            ax.plot(frequencies[0:len(s)] , np.abs(s))
            ax.set_xlim( 0, fs*.5 )
            plt.suptitle('Partials')

        plt.plot(np.arange(len(hps)), np.abs(hps))
        plt.scatter(peak, np.abs(hps[peak]), color='r')
        plt.title('HPS peak')
        plt.xlim( 0, 100 )

    FundamentalIdx = int ( f0*N/fs)
    Fundamental    = f0
    return Fundamental
