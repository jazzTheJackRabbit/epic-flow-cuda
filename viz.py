import numpy as np
import os
import sys

# WARNING: this will work on little-endian architectures (eg Intel x86) only!
if '__main__' == __name__:
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print 'Magic number incorrect. Invalid .flo file'
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                print 'Reading %d x %d flo file' % (w, h)
                data = np.fromfile(f, np.float32, count=2*w*h)
                # Reshape data into 3D array (columns, rows, bands)
                data2D = np.resize(data, (w, h, 2))
    else:
        print 'Specify a .flo file on the command line.'