import numpy as np
import matplotlib.pyplot as plt
import zipfile

NX = 0
NY = 0
# colormap
colormap = 'coolwarm'
fileName = 'C:\\Users\\dhufschl\\OneDrive - Bundesanstalt für Materialforschung und -prüfung (BAM)\\FTDT-PALLUP\\2D_FTDT_Result_8.npz'

def npz_headers(npz):
    """Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, _ , dtype = np.lib.format._read_array_header(npy, version)
            yield name[:-4], shape, dtype


def loadData ( fileName, iFrame, dtype=np.float64, NX = 0, NY = 0):
    nFrameOffset = iFrame*(NX*NY)
    nOffset = nFrameOffset*np.float64(1).itemsize

    data = np.memmap( fileName , mode="r",
                  dtype=np.float64, offset = nOffset, shape=(NX, 1) )
    return data

vars = npz_headers ( fileName )

for n in vars:
    if n[0] == 'P':
        print ( n )
        NFrames = n[1][0]
        NX = n[1][2]
        NY = n[1][1]
    
nFile = np.load( fileName,  mmap_mode='r')
dx = nFile['dx']
dt = nFile['dt']
c = nFile['c']
f = nFile['f']
NSources = nFile['NSources']
SourceWidth = nFile['SourceWidth']
SourceGap = nFile['SourceGap']

NAperture = dx * ( NSources * ( SourceGap + SourceWidth ) - SourceGap )
RNearField = np.power(NAperture , 2.0 ) * f / ( 4 * c )

print ( 'Recorded propagation times: %f us' % ( NFrames * 50 * dt * 1e6 ) )
print ( 'Recorded propagation distance : %f mm' % ( NFrames * 50 * dt * c * 1e3 ) )
print ( 'Aperture : %f mm ' %( NAperture * 1e3) )
print ( 'Points needed : %d' %( RNearField / dx ) )
print ( 'Nearfield distance : %f mm' % (RNearField*1e3 ) )
print ( 'Nearfield tof %f us' % (RNearField*1e6/c ))

NXStart = 0#NX//2 - int( .5 * RNearField / dx )
NXEnd   = NX#//2 + int( .5 * RNearField / dx )

NYStart = 0#100 + 10
NYEnd   = NY#100 + int( .5 * RNearField / dx ) + 10

x = np.arange ( -(NX//2)*dx, (NX//2)*dx, dx )
y = np.arange ( 0, (NY)*dx, dx )
xx, yy = np.meshgrid( x, y )

fig, ax = plt.subplots(1)
Psum = np.sum ( nFile['P'][0:300,NYStart:NYEnd,NXStart:NXEnd] , axis = 0)
Pmax = np.amax( Psum )
Pmin = np.amin( Psum )
print ( Psum / Pmax )
#print ( Psum / Pmin )
ax.pcolor ( xx[NYStart:NYEnd,NXStart:NXEnd], yy[NYStart:NYEnd,NXStart:NXEnd], Psum , vmin=-1, vmax=1, cmap=colormap, shading='auto')
plt.savefig('FTDT_Test_8.png', dpi=300)
plt.show()