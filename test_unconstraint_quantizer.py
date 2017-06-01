import numpy as np

from Unconstraint_quantizer import Unconstraint_quantizer
from init_DOGS import init_DOGS

ii = 0
n = 2
scale = 1
while(ii<20):
    S = np.random.random_sample((n,1))
    lattice = "An "
    [matrix, B, plane] = init_DOGS(n, lattice)
    [Sq, Errq, Sz, S] = Unconstraint_quantizer(S, scale, lattice)

    X = [S, Sq]
    e = Errq
    ii = ii + 1

    print X