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