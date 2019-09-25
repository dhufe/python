import numpy as np

def rect(T):
    """create a centered rectangular pulse of width $T"""
    return lambda t: (-T/2 <= t) & (t < T/2)

def pulse_train(t, at, shape):
    """create a train of pulses over $t at times $at and shape $shape"""
    return np.sum(shape(t - at[:,np.newaxis]), axis=0)

def phaseshifting ( alpha, f, t ):
    """ Creates a phase controlled waveform. alpha describes the switching point """
    omega = np.pi * 2 * f
    w = alpha / omega
    d = np.arange ( .5/f, t[-1], .5/f )
    d = d - .5*w;
    y = np.sin( omega * t);

    y2 = pulse_train(
        t=t,                           # time domain
        at=d,                          # times of pulses
        shape=rect(w)                  # shape of pulse
    )
    return y*y2
