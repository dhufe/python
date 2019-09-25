def MHCDEigenFrequencies ( dg, r,  Uhv=4e3, rho=1.2041, yDirNum=4, xDirNum=4 ):
    """Eigenfrequency calculatioon of a micro hollow cathode discharge.

    Args:
        dg      :   Distance of electrodes (discharge gap).
        r       :   Radius of the discharge hole.
        Uhv     :   Voltage between electrodes.
        rho     :   Gas density.
        yDirNum :   Col Number of eigenfrequencies.
        xDirNum :   Row number of eigenfrequencies.

    Returns:
        Returns the eigenfrequencies of the discharge.

    """
    from scipy.special import jn, jn_zeros
    import numpy as np

    k = 3
    e0 = 8.854187817e-12 # As x (Vm)^-1
    vm = np.sqrt (  e0 / rho ) * Uhv / dg 
    bessel_roots = np.zeros ( ( xDirNum, yDirNum) )

    for x in range ( 0, xDirNum):
        b = jn_zeros (x, yDirNum)
        for y in range ( 0, yDirNum):
            bessel_roots[y][x] = b[y]

    f = bessel_roots * vm * k / ( 2*np.pi*r)
    return f

def ReducedElectricalField ( E, p=101325, T=293.15 ):
    """Reduced electric field strength calculation using ideal gas law.

    Args:
        E       :   Electric field.
        p       :   Ambient pressure during discharge.
        T       :   Ambient temperature during discharge.        

    Returns:
        Returns the reduced electric field strength.
    """

    kB = 1.3806485279e-23   # Boltzmann constant (J x K^-1)
    N = p / ( kB * T)       # Particle density per cubic meter (m^-3)
    Td = E / N              # Calculate the reduced electrical field strength
    return (Td/10e-21)      # Normalise