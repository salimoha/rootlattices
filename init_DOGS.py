import numpy as np
import sys
import numpy.linalg as nplg

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
def init_DOGS(N, lattice):
    # calculate the cloud of nearest neighbors, the basis matrix, and the plane
    # that the lattice lies on, if applicable. Optioins for the variable lattice
    # are 'An', 'An*', 'Dn', 'Dn*', 'E8 '
    print ("initializing  lattice .... " + lattice + " ....")

    # build the lattice basis matrix
    matrix, v, lenn = make_matrix(lattice,N)
    iter = 2

    # find neighbors in matrix space
    [nr, nc] = np.shape(matrix)
    indx = -iter * np.ones(nc)
    indxorg = np.copy(indx)
    neigh = np.empty((0,nr))
    plane = np.empty((0, nr))

    nindx = len(indx)
    indx[nindx-1] = indx[nindx-1] - 1.0

    nloop = (2*iter+1)**N
    for j in range(0, nloop):
        indx[nindx - 1] = indx[nindx - 1] + 1.0
        for i in range(nindx-1, 0, -1):
            if (indx[i] > -indxorg[i]):
                indx[i] = indxorg[i]
                indx[i - 1] = indx[i - 1] + 1.0

        pt = np.dot(matrix, indx.transpose())
        v = nplg.norm(pt)
        if(v>0 and v <1.001*lenn):
            neigh = np.append(neigh, np.array([pt]), axis=0)

    # done with neighbors in matrix space
    [nr, nc] = np.shape(neigh)
    if(nr>nc):
        nn = nr
    else:
        nn = nc

    neigh2 = np.empty((0, N))
    if (lattice == "An "):
        plane, A = QRHouseholder(np.ones((N+1,1)))
        plane = plane[1:,:]

        # QRHouseholder returns 2:end orthogonal vectors - plane basis
        for i in range(0, nn):
            b = neigh[i, :]
            c = nplg.lstsq(plane.transpose(), b)[0]
            neigh2 = np.append(neigh2, np.array([c]), axis=0)

        neigh = np.copy(neigh2)
    elif (lattice == "An*"):
        v = np.ones((N+1,1))
        v[len(v)-1] = -N
        plane, A = QRHouseholder(v)
        plane = plane[1:,:]
        for i in range(0, nn):
            b = neigh[i, :]
            c = nplg.lstsq(plane.transpose(), b)[0]
            neigh2 = np.append(neigh2, np.array([c]), axis=0)
        neigh = np.copy(neigh2)
    print ("initialization complete, DELTA DOGS Lambda starting...")
    return(neigh, matrix, plane)

def QRHouseholder(A):
    # Compute a QR decomposition A=QR by applying a sequence of Householder reflections
    # to any MxN matrix A to reduce it to upper triangular form.
    [M, N] = np.shape(A)
    Q = np.eye(M, M)

    for i in range(0, min(N, M - 1)):
        A[i:M, i:N], sigma, w = Reflect(A[i:M, i:N])
        wdot = w.transpose()
        a = np.dot(Q[:, i:M], w)
        a = np.reshape(a, (len(a), 1))
        b = sigma * wdot
        Q[:, i:M] = Q[:,i:M] - a * b
    return(Q,A)

def Reflect(X):
    # Apply a Householder reflector matrix H?H to a MxN matrix X (i.e., calculate H?H*X),
    # with [sigma,w] arranged to give zeros in the (2:end,1) locations of the result.
    x = X[:, 0]
    if (np.real(x[1]) < 0):
        s = -1
    else:
        s = 1

    nu = s * nplg.norm(x) # Eqn (1.7b)

    if (nu == 0):
        sig = 0
        w = 0
    else:
        sig = (x[0] + nu) / nu
        w = np.append(x[0]+nu, x[1:]) / (x[0] + nu)

    X[0, 0] = -nu
    X[1:, 0] = 0 # Eqn (1.8)

    wdot = w.transpose()
    tmp = X[:, 1:len(X)]
    if(tmp.size>0): # prevent getting null matrix
        X[:, 1:] = X[:, 1:] - (np.conj(sig) * w) * (wdot * X[:, 1:]) #Eqn (1.9a)

    return (X,sig, w)

# %%%%%% Make_Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_matrix(lattice, N):
    if(lattice=="Zn "):
        matrix = np.eye(N)
        lenn = 1
        v = np.empty(shape=(0, 0))
    elif (lattice =="An "):
        v = np.ones(N)
        M = np.zeros((N,N+1))
        M[:, 0:N] = np.diag(v)
        P = np.diag(v,1)
        P = P[0:N, :]
        matrix = (-M+P).transpose()
        lenn = 1
        matrix = matrix / np.sqrt(2.0)
    elif (lattice == "Dn "):
        v = -1 * np.ones(N)
        matrix = np.diag(v)
        P = np.diag(-v,1)
        [nr,nc] = np.shape(P)
        P = P[0:nr-1, 0:nc-1]
        matrix = matrix+P
        matrix[1,0] = -1
        v = np.empty(shape=(0, 0))
        lenn = np.sqrt(2.0)
    elif (lattice == "Dn*"):
        v = np.ones(N)
        matrix = np.diag(v)
        matrix[:,N-1] = 0.5*np.ones(N)
        v = np.empty(shape=(0, 0))
        len1 = nplg.norm(matrix[:,0])
        len2 = nplg.norm(matrix[:,N-1])
        lenn = min(len1,len2)
    elif (lattice == "An*"):
        P = np.ones((N+1,N))
        P0 = np.diag(-np.ones(N),-1)
        P[1:N+1, :] = P0[1:N+1,0:N]
        P[:,N-1] = (1.0/(N+1.0)) * np.ones(N+1)
        P[0,N-1] = -N/(N+1.0)
        matrix = P
        v = np.ones((N+1,1))
        lenn = nplg.norm(matrix[:,N-1])
    elif (lattice == "E8 "):
        N = 8
        matrix = np.diag(np.ones(8))
        matrix = matrix + np.diag(-1.0 * np.ones(7), 1)
        matrix[0:4, N-1] = 0.5
        matrix[3:N, N-1] = -0.5
        matrix[0,0] = 2.0
        v = np.empty(shape=(0, 0))
        lenn = np.sqrt(2.0)
    else:
        print ("no viable lattice entered")
        sys.exit()

    return(matrix, v, lenn)

# neigh,matrix,plane = init_DOGS(2, "An ")
# print neigh
# print matrix
# print plane