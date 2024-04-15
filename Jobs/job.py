import sys
import numpy as np
sys.path.append('../')
from taunet import mpi
from taunet.simulation import CMBmap

taus = np.round(np.arange(0.01,0.13,5e-4),4)
cmb = CMBmap(taus,nsim=240000,verbose=False)

jobs = np.arange(0,240000)

for i in jobs[mpi.rank::mpi.size]:
    cmb.QU(i)
