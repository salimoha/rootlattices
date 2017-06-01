import numpy as np
import numpy.linalg as nplg
import sys

'''MIT License
Copyright (c) Shahrouz Ryan Alimo 2017
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Modified: Feb. 2017 '''


# %% Restriction Algorithm Library
# %
# %
# %  Title: Quantization process generation for unconstraint lattices
# %
# %  Author: Shahrouz Ryan Alimo Feb. 2017
# %
# %  Description:
# %    Domain setup.
# %    This function will restrict a point x onto the lattice defined by the variable 'lattice'
# %defined by the coarseness variable 'scale = 1/MeshSize'.


from init_DOGS import init_DOGS


def Unconstraint_quantizer(S, scale, lattice):
    # % Input:
    # % S: Intial set of points, note that: S = [x1;x2;...;xn]. Each column of
    # % the point matrix S is a one point. (like mathematical vectors).
    # %
    # % scale: the coarseness variable 'scale = 1/MeshSize'.
    # %
    # % lattice: type of desired lattice:
    # %   lattice=1: Zn or Cartesian lattice
    # %              Z2: square
    # %              Z3: cube
    # %   lattice=2: An or Zero-sum lattice. see Conway&Sloane
    # %              A2: hexagonal
    # %              A3: face-centered cubic (FCC)
    # %              A8: zero-sum
    # %   lattice=3: An* or Dual of zero-sum lattice.
    # %
    # %   lattice=4: Dn or Checkerboard lattice. %page 445 conway and sloane
    # %              D2: rotated squares
    # %   lattice=5: Dn* or Dual of Checkerboard lattice.
    # %
    # %   lattice=6: En or Gosset lattice.
    # %
    # %   lattice=7: En* or Dual of Gosset lattice.
    # %
    sr, sc = np.shape(S)
    Sz = np.empty((sr+1,sc))
    Sz[:] = np.NAN

    Sq = np.zeros((sr, sc))
    Errq = np.zeros(sc)

    # quantization process
    for k in range(0, sc): #iterate through matrix of pts
        x = S[:, k]
        Sz[0:sr, k] = x
        N = sr
        if lattice == "Zn ": # cartesian lattice
            x = x * scale         # scale x
            x = np.around(x)       # round to integer
            x = x / scale         # scale back to grid
        elif lattice == "An ": #An lattice, see Conway&Sloane
            x = x * scale * np.sqrt(2.0) # scale to integers
            neigh, matrix, plane = init_DOGS(N, lattice)
            x = np.dot(plane.transpose(), x)

            # algorithm from Conway & Sloane
            s = np.sum(x)
            x_p = x - s/(N+1.0) * np.ones((1,N+1))  # project onto plane

            # calculate deficiency
            f_x = np.round(x_p)
            DELTA = int(np.sum(f_x))
            delta = x_p - f_x
            i = np.argsort(delta)
            delta = np.sort(delta)

            if DELTA > 0:
                for j in range(0, DELTA):
                    f_x[0,i[0,j]] = f_x[0,i[0,j]] - 1

            if DELTA < 0:
                for j in range(N-DELTA, N+1, -1):
                    f_x[0,i[j]] = f_x[0,i[j]] + 1

            # back into N
            x = (nplg.lstsq(plane.transpose(), f_x.transpose())[0]) / scale / np.sqrt(2.0)
            x = x[:,0]

            # this is all straight out of C&S except for dividing by sqrt(2)
            # in the last step. This is necessary because otherwise the
            # algorithm will restrict to the lattice An which has a base
            # vector length of sqrt(2).
            Sz[:, k] = f_x / scale / np.sqrt(2.0)
        elif lattice == "Dn ":  # Dn
            x = x * scale
            f_x = np.around(x)
            g_x = np.copy(f_x)
            d = x - g_x
            #a = np.max(np.abs(d))
            j = np.argmax(np.abs(d))
            if x[j] - g_x[j] > 0:
                g_x[j] = g_x[j]+1
            elif x[j] - g_x[j] <= 0:
                g_x[j] = g_x[j]-1

            a = np.sum(f_x)
            #b = np.sum(g_x)
            #x = np.copy(f_x)
            if np.fmod(a, 2.0) == 0:
                x = np.copy(f_x)
            else:
                x = np.copy(g_x)
            x = x / scale
        elif lattice == "An*":  # An*
            x = x * scale
            S = np.zeros((N, N + 1))
            for s in range(0, N+1):
                for kk in range(0, N):
                    if kk+1 <= N + 1 - s:
                        S[kk, s] = s / (N + 1.0)
                    else:
                        S[kk, s] = (s - N - 1.0) / (N + 1.0)

            lattice = "An "
            Xq = np.zeros((len(x), N+1))
            xx = np.reshape(x, (len(x), 1))
            for ii in range(0, N + 1):
                tmp = x - S[:, ii]
                tmp = np.reshape(tmp, (len(tmp), 1))
                a = Unconstraint_quantizer(tmp, 1, lattice)
                b = np.reshape(S[:, ii], (len(x), 1))
                Xq[:, ii] = a[:, 0]+b[:, 0]

            #a, b, xq = mindis(x, Xq);
            #x = xq / scale
            lattice = "An*"
        elif lattice == "Dn*":
            x = x.transpose()
            x = x * scale
            x1 = np.round(x)
            x2 = np.round(x - 0.5 * np.ones((1, N))) + 0.5 * np.ones((1, N))
            if nplg.norm(x - x2) < nplg.norm(x - x1):
                x=x2
            else:
                x=x1
            x = x / scale
            x = x.transpose()
        elif lattice == "E6 ":
            x = x.transpose()
            x = x / scale * np.sqrt(3.0)
            x = restrict_e6_fast(x)
            x = x * scale / np.sqrt(3.0)
            x = x.transpose()
        elif lattice == "E7 ":
            x = x.transpose()
            x = x / scale * 2.0
            N = 7
            W = np.array([[0, 1, 0, 1, 0, 1, 0, 1],
                          [0, 0, 1, 1, 0, 0, 1, 1],
                          [0, 0, 0, 0, 1, 1, 1, 1],
                          [0, 0, 1, 1, 1, 1, 0, 0],
                          [0, 1, 0, 1, 1, 0, 1, 0],
                          [0, 1, 1, 0, 0, 1, 1, 0],
                          [0, 1, 1, 0, 1, 0, 0, 1]])
            rw, cw = np.shape(W)

            # quantize to the 2*R grid shifted by each W vector
            d = np.zeros(cw)
            pts1 = np.zeros((cw, len(x)))
            for i in range(0, cw):
                shift = W[:, i]
                x1 = 2.0 * (np.round((x - shift) / 2.0)) + shift
                d[i] = nplg.norm(x - x1)
                pts1[i,:] = x1
            b = np.argmin(d)
            x = pts1[b,:]
            x = x * scale / 2.0
            x = x.transpose()
        elif lattice == "E8 ":
            x = x.transpose()
            # first quantize to Dn *
            x = x * scale
            xorg = np.copy(x)
            xb = np.copy(x)
            x1 = np.around(x)
            x2 = (np.around(x - 0.5 * np.ones((1, N))) + 0.5 * np.ones((1, N)))
            if nplg.norm(x - x2) < nplg.norm(x - x1):
                x = np.copy(x2)
            else:
                x = np.copy(x1)
            xa = np.copy(x) # save x as xa

            x = xb + 0.5 * np.ones((1, N))
            x1 = np.around(x)
            x2 = (np.around(x - 0.5 * np.ones((1, N))) + 0.5 * np.ones((1, N)))
            if nplg.norm(x - x2) < nplg.norm(x - x1):
                x = np.copy(x2)
            else:
                x = np.copy(x1)
            xb = np.copy(x)  # save x as xb

            # figure out which the original x is closer to
            d1 = nplg.norm(xorg - xa)
            d2 = nplg.norm(xorg - xb)
            if d1 < d2:
                x = np.copy(xa)
            else:
                x = np.copy(xb)
            x = x / scale
            x = x.transpose()
        Sq[:,k] = np.copy(x)
        Errq[k] = nplg.norm(x-S[:,k])
    return(Sq, Errq, Sz, S)

def restrict_e6_fast(x):
    xorg = np.copy(x)
    W1 = np.array([[0, 0, 0, 0, 0, 0],
                   [0, -1, 0, -1, 0, -1],
                   [0, -2, 0, -2, 0, -2]])
    lW = 3
    x = [x[1], -x[0], x[3], -x[2], x[5], -x[4]]
    x = [x, x, x] - W1
    x_hat = x / np.sqrt(3.0)

    #quantize each pair of numbers to A2
    for ii in range(0, lW):
        for j in range(1, 6, 2):
            x = [x_hat[ii,j-1], x_hat[ii,j]]
            x = restrictA2(x)
            x_hat[ii,j-1] = x[0]
            x_hat[ii,j] = x[1]
    x_hat = [-x_hat[:, 1], x_hat[:, 0], -x_hat[:, 3], x_hat[:, 2], -x_hat[:, 5], x_hat[:, 4]] * np.sqrt(3.0)
    x_hat = [x_hat[:, 0]-W1[:, 1], x_hat[:, 1], x_hat[:, 2]-W1[:, 3], x_hat[:, 3], x_hat[:, 4]-W1[:, 5], x_hat[:, 5]]

    d = np.zeros((1, 3))
    for ii in range(0,lW):
        d[ii] = nplg.norm(xorg - x_hat[ii,:])
    b = np.argmin(d)
    x = x_hat[b,:].transpose()
    return (x)

def restrictA2(x):
    return (x)

# S = np.array([[0.656],[0.123]])
# scale = 1.0
# Unconstraint_quantizer(S,scale, "E6 ")
# Unconstraint_quantizer(S,scale, "E8 ")

# S = np.array([[-0.01066667],[0.45633333]])
# c = Unconstraint_quantizer(S, 1, "An ")
# print c
