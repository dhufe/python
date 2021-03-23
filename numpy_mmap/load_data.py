import numpy as np
import matplotlib.pyplot as plt
import zipfile

NX = 0
NY = 0

fileName = 'numpy_mmap/example.npz'

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
        NX = n[1][0]
        NY = n[1][0]

nFile = np.load( fileName,  mmap_mode='r')

fig, ax = plt.subplots(1)
ax.pcolor ( nFile['P'][:,:] )

plt.show()