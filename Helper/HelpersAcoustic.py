import numpy as np

def ShockwaveDistance ( f, beta=1.2, c0=343.21, p0=101325, rho0=1.2041 ):
    import numpy as np
    vmax = p0 / ( rho0 * c0)
    xs = np.power ( c0, 2.0 ) / ( beta * 2 * np.pi * f * vmax)
    return xs

def GetMicTransferfunction ( f ):
    import numpy as np
    # load data
    data = np.loadtxt ( '../data/20180827-B+K_microphone_transferfunction/BK_microphone_pressure.csv', skiprows=1 )
    z = np.poly1d(np.polyfit(data[:,0], data[:,1] , 7 ) )
    mic_poly = z(f)
    return mic_poly

def us_impulse( nfreq, fa , NOrder = 4 ):
    TPulse = (NOrder * 2 * np.pi / ( 2 * np.pi * nfreq))
    t = np.arange ( 0, TPulse, 1/fa )
    pulsesig = .5 * (-1)**(NOrder) * ( 1 - np.cos(2 * np.pi * nfreq * t / NOrder ) ) * np.cos ( 2 * np.pi * nfreq * t )
    return pulsesig, TPulse, t
