from __future__ import division
import numpy as np
from numpy.linalg import inv,norm

def cgls_eq_mag(x,y,z,zj,shape,data,F,h,itmax):
    '''
    Calculates the estimate physical property distribution of
    an equivalent layer that repreduces a total-field anomaly
    data through the conjugate gradient least square iterative
    method [2].

    [2]Aster, Richard C., Brian Borchers, and Clifford H. Thurber. 
    Parameter estimation and inverse problems. Elsevier, 2018.
    p.166

    input
    x, y: numpy array - the x, y coordinates
    of the grid and equivalent layer points.
    z: numpy array - the height of observation points.
    zj: numpy array - the depth of the equivalent layer.
    shape: tuple - grid size.
    data: numpy array - the total-field anomaly 
    potential field data at the x,y and z grid points.
    F: numpy array - cosines directions of the main magnetic field.
    h: numpy array - cosines directions of the body's magnetization.
    itmax: scalar - number of iterations of the CGLS method.

    output
    p_cg: numpy array - final equivalent layer property estimative.
    datap: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == zj.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    #assert zj.all() > z.all(), 'The equivalent layer must be beneath\
	#the observation points'
    
    # Dataset lenght 
    N = shape[0]*shape[1]
    
    A = np.empty((N, N), dtype=np.float)
    for i in range (N):
        a = (x-x[i])
        b = (y-y[i])
        c = (zj-z[i])
        r = (a*a+b*b+c*c)
        r3 = r**(-1.5)
        r5 = r**(2.5)
        Hxx = -r3+3*(a*a)/r5
        Hxy = 3*(a*b)/r5
        Hxz = 3*(a*c)/r5
        Hyy = -r3+3*(b*b)/r5
        Hyz = 3*(b*c)/r5
        Hzz = -r3+3*(c*c)/r5
        A[i] = 100*((F[0]*Hxx+F[1]*Hxy+F[2]*Hxz)*h[0] + (F[0]*Hxy+F[1]*Hyy+
        F[2]*Hyz)*h[1] + (F[0]*Hxz+F[1]*Hyz+F[2]*Hzz)*h[2])

    #CGLS inversion method for Equivalent layer
    p_cgls = cgls_loop(A,shape,N,data,itmax)

    #Final predicted data
    datap = A.dot(p_cgls)

    return p_cgls, datap
    
def cgls_eq_bccb_mag(x,y,z,zj,shape,data,F,h,itmax):
    '''
    Calculates the estimate physical property distribution of
    an equivalent layer that repreduces a total-field anomaly
    data through the conjugate gradient least square iterative
    method [2].
    This implementation uses a fast way to calculate the forward
    problem at each iteration taking advantage of the BTTB
    (Block-Toeplitz Toreplitz-Block) structures.

    [2]Aster, Richard C., Brian Borchers, and Clifford H. Thurber. 
    Parameter estimation and inverse problems. Elsevier, 2018.
    p.166

    input
    x, y: numpy array - the x, y coordinates
    of the grid and equivalent layer points.
    z: numpy array - the height of observation points.
    zj: numpy array - the depth of the equivalent layer.
    shape: tuple - grid size.
    data: numpy array - the total-field anomaly 
    potential field data at the x,y and z grid points.
    F: numpy array - cosines directions of the main magnetic field.
    h: numpy array - cosines directions of the body's magnetization.
    itmax: scalar - number of iterations of the CGLS method.

    output
    p_cg: numpy array - final equivalent layer property estimative.
    datap: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == zj.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    #assert zj.all() > z.all(), 'The equivalent layer must be beneath\
	#the observation points'
    
    # Dataset lenght 
    N = shape[0]*shape[1]

    #if N < 150000:
        # Calculates the first and last rows of the first and last rows of
        # blocks of the sensitivity matrix for N < 150 000
    #    bttb_0, bttb_1, bttb_2, bttb_3 = bttb_mag(x,y,z,zj,F,h,shape)
        # Calculates the eigenvalues of BCCB and the transposed matrix
    #    cev = cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
    #    cev_row = cev_mag_row(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
    #else:
        # Calculates the first row of each component of the second derivatives
        # of the function 1/r
    hxx,hxy,hxz,hyy,hyz,hzz = h_bttb_mag(x,y,z,zj,F,h,shape)
        # Calculates the eigenvalues of BCCB and the transposed matrix
    cev = ones_cev_mag_no_for(hxx,hxy,hxz,hyy,hyz,hzz,shape,N,F,h)
    cev_row = ones_cev_mag_row_no_for(hxx,hxy,hxz,hyy,hyz,hzz,shape,N,F,h)

    #CGLS inversion method for Equivalent layer
    p_cgls = cgls_bccb_loop(cev,cev_row,shape,N,data,itmax)

    #Final predicted data
    datap = fast_forward_bccb_mag(cev,p_cgls,shape,N)

    return p_cgls, datap

def sensitivity_matrix(x,y,z,zj,F,h,N):
    '''
    Calculates a full NxN matrix given by
    the first derivative of the function
    1/r.

    input
    x, y: numpy array - the x, y coordinates of
    the grid and equivalent layer points.
	z: numpy array - the height of observation points.
    h: numpy array - the depth of the equivalent layer.
    N: scalar - number of observation points.

    output
    A: matrix - full NxN matrix given by
    the first derivative of the function
    1/r.
    '''
    A = np.empty((N, N), dtype=np.float)
    for i in range (N):
        a = (x-x[i])
        b = (y-y[i])
        c = (z-zj[i])
        r = (a*a+b*b+c*c)
        r3 = r**(-1.5)
        r5 = r**(2.5)
        Hxx = -r3+3*(a*a)/r5
        Hxy = 3*(a*b)/r5
        Hxz = 3*(a*c)/r5
        Hyy = -r3+3*(b*b)/r5
        Hyz = 3*(b*c)/r5
        Hzz = -r3+3*(c*c)/r5
        A[:,i] = 100*((F[0]*Hxx+F[1]*Hxy+F[2]*Hxz)*h[0] + (F[0]*Hxy+F[1]*Hyy+
        F[2]*Hyz)*h[1] + (F[0]*Hxz+F[1]*Hyz+F[2]*Hzz)*h[2])
    return A

def classic_mag(x,y,z,zj,F,h,N,data):
    '''
    Calculates the estimate physical property distribution of
    an equivalent layer that repreduces a total-field anomaly
    data by solving a linear inversion problem.

    input
    x, y: numpy array - the x, y coordinates
    of the grid and equivalent layer points.
    z: numpy array - the height of observation points.
    zj: numpy array - the depth of the equivalent layer.
    shape: tuple - grid size.
    potential field at the grid points.
    F: numpy array - cosines directions of the main magnetic field.
    h: numpy array - cosines directions of the body's magnetization.
    data: numpy array - numpy array - the total-field anomaly 
    potential field data at the x,y and z grid points.

    output
    p: numpy array - final equivalent layer property estimative.
    data: numpy array - the predicted data.
    '''
    A = np.empty((N, N), dtype=np.float)
    for i in range (N):
        a = (x-x[i])
        b = (y-y[i])
        c = (z-zj[i])
        r = (a*a+b*b+c*c)
        r3 = r**(-1.5)
        r5 = r**(2.5)
        Hxx = -r3+3*(a*a)/r5
        Hxy = 3*(a*b)/r5
        Hxz = 3*(a*c)/r5
        Hyy = -r3+3*(b*b)/r5
        Hyz = 3*(b*c)/r5
        Hzz = -r3+3*(c*c)/r5
        A[:,i] = 100*((F[0]*Hxx+F[1]*Hxy+F[2]*Hxz)*h[0] + (F[0]*Hxy+F[1]*Hyy+
        F[2]*Hyz)*h[1] + (F[0]*Hxz+F[1]*Hyz+F[2]*Hzz)*h[2])
    I = np.identity(N)
    ATA = A.T.dot(A)
    mu = (np.trace(ATA)/N)*10**(-4)
    AI = inv(ATA+mu*I)
    p = (AI.dot(A.T)).dot(data)
    data = A.dot(p)
    return p, data

def bttb_mag(x,y,z,zj,F,h,shape):
    '''
    Calculates the first and last rows of the first and last rows of
    blocks of the sensitivity matrix

    input
    x, y: numpy array - the x, y coordinates
    of the grid and equivalent layer points.
    z: numpy array - the height of observation points.
    h: numpy array - the depth of the equivalent layer.
    shape: tuple - grid size.
    potential field at the grid points.
    F: numpy array - cosines directions of the main magnetic field.
    h: numpy array - cosines directions of the body's magnetization.

    output
    bttb_0: numpy array - first line of the first row of blocks of the
    sensitivity matrix.
    bttb_1: numpy array - last line of the first row of blocks of the
    sensitivity matrix.
    bttb_2: numpy array - first line of the last row of blocks of the
    sensitivity matrix.
    bttb_3: numpy array - last line of the last row of blocks of the
    sensitivity matrix.
    '''
    # First row of BTTB first row
    a = (x-x[0])
    b = (y-y[0])
    c = (zj-z[0])
    r = (a*a+b*b+c*c)
    r3 = r**(-1.5)
    r5 = r**(2.5)
    Hxx = -r3+3*(a*a)/r5
    Hxy = 3*(a*b)/r5
    Hxz = 3*(a*c)/r5
    Hyy = -r3+3*(b*b)/r5
    Hyz = 3*(b*c)/r5
    Hzz = -r3+3*(c*c)/r5
    bttb_0 = 100*((F[0]*Hxx+F[1]*Hxy+F[2]*Hxz)*h[0] + (F[0]*Hxy+F[1]*Hyy+
    F[2]*Hyz)*h[1] + (F[0]*Hxz+F[1]*Hyz+F[2]*Hzz)*h[2])

    # Last row of BTTB first row
    a = (x-x[shape[1]-1])
    b = (y-y[shape[1]-1])
    c = (zj-z[shape[1]-1])
    r = (a*a+b*b+c*c)
    r3 = r**(-1.5)
    r5 = r**(2.5)
    Hxx = -r3+3*(a*a)/r5
    Hxy = 3*(a*b)/r5
    Hxz = 3*(a*c)/r5
    Hyy = -r3+3*(b*b)/r5
    Hyz = 3*(b*c)/r5
    Hzz = -r3+3*(c*c)/r5
    bttb_1 = 100*((F[0]*Hxx+F[1]*Hxy+F[2]*Hxz)*h[0] + (F[0]*Hxy+F[1]*Hyy+
    F[2]*Hyz)*h[1] + (F[0]*Hxz+F[1]*Hyz+F[2]*Hzz)*h[2])

    # First row of BTTB last row
    a = (x-x[-shape[1]])
    b = (y-y[-shape[1]])
    c = (zj-z[-shape[1]])
    r = (a*a+b*b+c*c)
    r3 = r**(-1.5)
    r5 = r**(2.5)
    Hxx = -r3+3*(a*a)/r5
    Hxy = 3*(a*b)/r5
    Hxz = 3*(a*c)/r5
    Hyy = -r3+3*(b*b)/r5
    Hyz = 3*(b*c)/r5
    Hzz = -r3+3*(c*c)/r5
    bttb_2 = 100*((F[0]*Hxx+F[1]*Hxy+F[2]*Hxz)*h[0] + (F[0]*Hxy+F[1]*Hyy+
    F[2]*Hyz)*h[1] + (F[0]*Hxz+F[1]*Hyz+F[2]*Hzz)*h[2])

    # Last row of BTTB last row
    a = (x-x[-1])
    b = (y-y[-1])
    c = (zj-z[-1])
    r = (a*a+b*b+c*c)
    r3 = r**(-1.5)
    r5 = r**(2.5)
    Hxx = -r3+3*(a*a)/r5
    Hxy = 3*(a*b)/r5
    Hxz = 3*(a*c)/r5
    Hyy = -r3+3*(b*b)/r5
    Hyz = 3*(b*c)/r5
    Hzz = -r3+3*(c*c)/r5
    bttb_3 = 100*((F[0]*Hxx+F[1]*Hxy+F[2]*Hxz)*h[0] + (F[0]*Hxy+F[1]*Hyy+
    F[2]*Hyz)*h[1] + (F[0]*Hxz+F[1]*Hyz+F[2]*Hzz)*h[2])

    return bttb_0, bttb_1, bttb_2, bttb_3
    
def h_bttb_mag(x,y,z,zj,F,h,shape):
    '''
    Calculates the first and last rows of the first and last rows of
    blocks of the sensitivity matrix

    input
    x, y: numpy array - the x, y coordinates
    of the grid and equivalent layer points.
    z: numpy array - the height of observation points.
    h: numpy array - the depth of the equivalent layer.
    shape: tuple - grid size.
    potential field at the grid points.
    F: numpy array - cosines directions of the main magnetic field.
    h: numpy array - cosines directions of the body's magnetization.

    output
    bttb_0: numpy array - first line of the first row of blocks of the
    sensitivity matrix.
    bttb_1: numpy array - last line of the first row of blocks of the
    sensitivity matrix.
    bttb_2: numpy array - first line of the last row of blocks of the
    sensitivity matrix.
    bttb_3: numpy array - last line of the last row of blocks of the
    sensitivity matrix.
    '''
    # First row of each component of the second derivatives
    a = (x-x[0])
    b = (y-y[0])
    c = (z-zj[0])
    r = (a*a+b*b+c*c)
    r3 = r**(-1.5)
    r5 = r**(2.5)
    Hxx = -r3+3*(a*a)/r5
    Hxy = 3*(a*b)/r5
    Hxz = 3*(a*c)/r5
    Hyy = -r3+3*(b*b)/r5
    Hyz = 3*(b*c)/r5
    Hzz = -r3+3*(c*c)/r5

    return Hxx,Hxy,Hxz,Hyy,Hyz,Hzz
    
def ones_cev_mag(Hxx,Hxy,Hxz,Hyy,Hyz,Hzz,shape,N,F,h):
    '''
    Calculates the eigenvalues of the BCCB matrix using 
    only the effect of one equivalent source.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    Hxx,Hxy,Hxz,Hyy,Hyz,Hzz: numpy array - first row of each component of 
    the second derivatives of 1/r necessary to calculate the BCCB eigenvalues.

    output
    cev: numpy array - eigenvalues of the BCCB matrix.
    '''
    bccb_xx = np.zeros(4*N, dtype='complex128')
    bccb_xy = np.zeros(4*N, dtype='complex128')
    bccb_xz = np.zeros(4*N, dtype='complex128')
    bccb_yy = np.zeros(4*N, dtype='complex128')
    bccb_yz = np.zeros(4*N, dtype='complex128')
    bccb_zz = np.zeros(4*N, dtype='complex128')
    k = 2*shape[0]-1

    for i in range (shape[0]):
        block_xx = Hxx[shape[1]*(i):shape[1]*(i+1)]
        block_xy = Hxy[shape[1]*(i):shape[1]*(i+1)]
        block_xz = Hxz[shape[1]*(i):shape[1]*(i+1)]
        block_yy = Hyy[shape[1]*(i):shape[1]*(i+1)]
        block_yz = Hyz[shape[1]*(i):shape[1]*(i+1)]
        block_zz = Hzz[shape[1]*(i):shape[1]*(i+1)]
        rev_xx = block_xx[::-1]
        rev_xy = -block_xy[::-1]
        rev_xz = block_xz[::-1]
        rev_yy = block_yy[::-1]
        rev_yz = -block_yz[::-1]
        rev_zz = block_zz[::-1]
        bccb_xx[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_xx,
        0,rev_xx[:-1]), axis=None)
        bccb_xy[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_xy,
        0,rev_xy[:-1]), axis=None)
        bccb_xz[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_xz,
        0,rev_xz[:-1]), axis=None)
        bccb_yy[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_yy,
        0,rev_yy[:-1]), axis=None)
        bccb_yz[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_yz,
        0,rev_yz[:-1]), axis=None)
        bccb_zz[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_zz,
        0,rev_zz[:-1]), axis=None)
        if i > 0:
            bccb_xx[shape[1]*(2*k):shape[1]*(2*k+2)] = bccb_xx[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_xy[shape[1]*(2*k):shape[1]*(2*k+2)] = -bccb_xy[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_xz[shape[1]*(2*k):shape[1]*(2*k+2)] = -bccb_xz[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_yy[shape[1]*(2*k):shape[1]*(2*k+2)] = bccb_yy[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_yz[shape[1]*(2*k):shape[1]*(2*k+2)] = bccb_yz[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_zz[shape[1]*(2*k):shape[1]*(2*k+2)] = bccb_zz[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            k -= 1

    bccb = 100*((F[0]*bccb_xx+F[1]*bccb_xy+F[2]*bccb_xz)*h[0] + (F[0]*
    bccb_xy+F[1]*bccb_yy+F[2]*bccb_yz)*h[1] + (F[0]*bccb_xz+F[1]*
    bccb_yz+F[2]*bccb_zz)*h[2])

    BCCB = bccb.reshape(2*shape[0],2*shape[1]).T
    cev_mag = np.fft.fft2(BCCB)
    return cev_mag
    
def ones_cev_mag_row(Hxx,Hxy,Hxz,Hyy,Hyz,Hzz,shape,N,F,h):
    '''
    Calculates the eigenvalues of the transposed BCCB matrix using 
    only the effect of one equivalent source.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    Hxx,Hxy,Hxz,Hyy,Hyz,Hzz: numpy array - first row of each component of
    the second derivatives of 1/r necessary to calculate the transposed
    BCCB eigenvalues.

    output
    cev: numpy array - eigenvalues of the transposed BCCB matrix.
    '''
    bccb_xx = np.zeros(4*N, dtype='complex128')
    bccb_xy = np.zeros(4*N, dtype='complex128')
    bccb_xz = np.zeros(4*N, dtype='complex128')
    bccb_yy = np.zeros(4*N, dtype='complex128')
    bccb_yz = np.zeros(4*N, dtype='complex128')
    bccb_zz = np.zeros(4*N, dtype='complex128')
    k = 2*shape[0]-1

    for i in range (shape[0]):
        block_xx = Hxx[shape[1]*(i):shape[1]*(i+1)]
        block_xy = Hxy[shape[1]*(i):shape[1]*(i+1)]
        block_xz = -Hxz[shape[1]*(i):shape[1]*(i+1)]
        block_yy = Hyy[shape[1]*(i):shape[1]*(i+1)]
        block_yz = -Hyz[shape[1]*(i):shape[1]*(i+1)]
        block_zz = Hzz[shape[1]*(i):shape[1]*(i+1)]
        rev_xx = block_xx[::-1]
        rev_xy = -block_xy[::-1]
        rev_xz = block_xz[::-1]
        rev_yy = block_yy[::-1]
        rev_yz = -block_yz[::-1]
        rev_zz = block_zz[::-1]
        bccb_xx[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_xx,
        0,rev_xx[:-1]), axis=None)
        bccb_xy[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_xy,
        0,rev_xy[:-1]), axis=None)
        bccb_xz[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_xz,
        0,rev_xz[:-1]), axis=None)
        bccb_yy[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_yy,
        0,rev_yy[:-1]), axis=None)
        bccb_yz[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_yz,
        0,rev_yz[:-1]), axis=None)
        bccb_zz[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_zz,
        0,rev_zz[:-1]), axis=None)
        if i > 0:
            bccb_xx[shape[1]*(2*k):shape[1]*(2*k+2)] = bccb_xx[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_xy[shape[1]*(2*k):shape[1]*(2*k+2)] = -bccb_xy[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_xz[shape[1]*(2*k):shape[1]*(2*k+2)] = -bccb_xz[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_yy[shape[1]*(2*k):shape[1]*(2*k+2)] = bccb_yy[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_yz[shape[1]*(2*k):shape[1]*(2*k+2)] = bccb_yz[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            bccb_zz[shape[1]*(2*k):shape[1]*(2*k+2)] = bccb_zz[shape[1]*
            (2*i):shape[1]*(2*i+2)]
            k -= 1

    bccb = 100*((F[0]*bccb_xx+F[1]*bccb_xy+F[2]*bccb_xz)*h[0] + (F[0]*
    bccb_xy+F[1]*bccb_yy+F[2]*bccb_yz)*h[1] + (F[0]*bccb_xz+F[1]*
    bccb_yz+F[2]*bccb_zz)*h[2])

    BCCB = bccb.reshape(2*shape[0],2*shape[1]).T
    cev_mag_row = np.fft.fft2(BCCB)
    return cev_mag_row

def ones_cev_mag_no_for(Hxx,Hxy,Hxz,Hyy,Hyz,Hzz,shape,N,F,h):
    '''
    Calculates the eigenvalues of the BCCB matrix using 
    only the effect of one equivalent source.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    Hxx,Hxy,Hxz,Hyy,Hyz,Hzz: numpy array - first row of each component of 
    the second derivatives of 1/r necessary to calculate the BCCB eigenvalues.

    output
    cev: numpy array - eigenvalues of the BCCB matrix.
    '''
    Hxx_matrix = Hxx.reshape(shape[0],shape[1])
    Hxy_matrix = Hxy.reshape(shape[0],shape[1])
    Hxz_matrix = Hxz.reshape(shape[0],shape[1])
    Hyy_matrix = Hyy.reshape(shape[0],shape[1])
    Hyz_matrix = Hyz.reshape(shape[0],shape[1])

    bccb_xx = np.zeros((shape[0]*2,shape[1]*2))
    bccb_xx[:shape[0],:shape[1]] = Hxx_matrix
    bccb_xx[:shape[0],shape[1]+1:] = Hxx_matrix[:,1:shape[1]][:,::-1]
    bccb_xx[shape[0]+1:,:] = bccb_xx[1:shape[0],:][::-1]

    bccb_xy = np.zeros((shape[0]*2,shape[1]*2))
    bccb_xy[:shape[0],:shape[1]] = Hxy_matrix
    bccb_xy[:shape[0],shape[1]+1:] = -Hxy_matrix[:,1:shape[1]][:,::-1]
    bccb_xy[shape[0]+1:,:] = -bccb_xy[1:shape[0],:][::-1]

    bccb_xz = np.zeros((shape[0]*2,shape[1]*2))
    bccb_xz[:shape[0],:shape[1]] = Hxz_matrix
    bccb_xz[:shape[0],shape[1]+1:] = Hxz_matrix[:,1:shape[1]][:,::-1]
    bccb_xz[shape[0]+1:,:] = -bccb_xz[1:shape[0],:][::-1]

    bccb_yy = np.zeros((shape[0]*2,shape[1]*2))
    bccb_yy[:shape[0],:shape[1]] = Hyy_matrix
    bccb_yy[:shape[0],shape[1]+1:] = Hyy_matrix[:,1:shape[1]][:,::-1]
    bccb_yy[shape[0]+1:,:] = bccb_yy[1:shape[0],:][::-1]

    bccb_yz = np.zeros((shape[0]*2,shape[1]*2))
    bccb_yz[:shape[0],:shape[1]] = Hyz_matrix
    bccb_yz[:shape[0],shape[1]+1:] = -Hyz_matrix[:,1:shape[1]][:,::-1]
    bccb_yz[shape[0]+1:,:] = bccb_yz[1:shape[0],:][::-1]

    bccb_zz = -bccb_xx-bccb_yy

    bccb = 100*((F[0]*bccb_xx+F[1]*bccb_xy+F[2]*bccb_xz)*h[0] + 
                           (F[0]*bccb_xy+F[1]*bccb_yy+F[2]*bccb_yz)*h[1] +
                           (F[0]*bccb_xz+F[1]*bccb_yz+F[2]*bccb_zz)*h[2])

    BCCB = bccb.T
    cev_mag = np.fft.fft2(BCCB)
    return cev_mag
    
def ones_cev_mag_row_no_for(Hxx,Hxy,Hxz,Hyy,Hyz,Hzz,shape,N,F,h):
    '''
    Calculates the eigenvalues of the transposed BCCB matrix using 
    only the effect of one equivalent source.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    Hxx,Hxy,Hxz,Hyy,Hyz,Hzz: numpy array - first row of each component of
    the second derivatives of 1/r necessary to calculate the transposed
    BCCB eigenvalues.

    output
    cev: numpy array - eigenvalues of the transposed BCCB matrix.
    '''
    Hxx_matrix = Hxx.reshape(shape[0],shape[1])
    Hxy_matrix = Hxy.reshape(shape[0],shape[1])
    Hxz_matrix = -Hxz.reshape(shape[0],shape[1])
    Hyy_matrix = Hyy.reshape(shape[0],shape[1])
    Hyz_matrix = -Hyz.reshape(shape[0],shape[1])

    bccb_xx = np.zeros((shape[0]*2,shape[1]*2))
    bccb_xx[:shape[0],:shape[1]] = Hxx_matrix
    bccb_xx[:shape[0],shape[1]+1:] = Hxx_matrix[:,1:shape[1]][:,::-1]
    bccb_xx[shape[0]+1:,:] = bccb_xx[1:shape[0],:][::-1]

    bccb_xy = np.zeros((shape[0]*2,shape[1]*2))
    bccb_xy[:shape[0],:shape[1]] = Hxy_matrix
    bccb_xy[:shape[0],shape[1]+1:] = -Hxy_matrix[:,1:shape[1]][:,::-1]
    bccb_xy[shape[0]+1:,:] = -bccb_xy[1:shape[0],:][::-1]

    bccb_xz = np.zeros((shape[0]*2,shape[1]*2))
    bccb_xz[:shape[0],:shape[1]] = Hxz_matrix
    bccb_xz[:shape[0],shape[1]+1:] = Hxz_matrix[:,1:shape[1]][:,::-1]
    bccb_xz[shape[0]+1:,:] = -bccb_xz[1:shape[0],:][::-1]

    bccb_yy = np.zeros((shape[0]*2,shape[1]*2))
    bccb_yy[:shape[0],:shape[1]] = Hyy_matrix
    bccb_yy[:shape[0],shape[1]+1:] = Hyy_matrix[:,1:shape[1]][:,::-1]
    bccb_yy[shape[0]+1:,:] = bccb_yy[1:shape[0],:][::-1]

    bccb_yz = np.zeros((shape[0]*2,shape[1]*2))
    bccb_yz[:shape[0],:shape[1]] = Hyz_matrix
    bccb_yz[:shape[0],shape[1]+1:] = -Hyz_matrix[:,1:shape[1]][:,::-1]
    bccb_yz[shape[0]+1:,:] = bccb_yz[1:shape[0],:][::-1]

    bccb_zz = -bccb_xx-bccb_yy

    bccb = 100*((F[0]*bccb_xx+F[1]*bccb_xy+F[2]*bccb_xz)*h[0] + 
                           (F[0]*bccb_xy+F[1]*bccb_yy+F[2]*bccb_yz)*h[1] +
                           (F[0]*bccb_xz+F[1]*bccb_yz+F[2]*bccb_zz)*h[2])

    BCCB = bccb.T
    cev_mag_row = np.fft.fft2(BCCB)
    return cev_mag_row    

def cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N):
    '''
    Calculates the eigenvalues of the BCCB matrix.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    bttb_0,bttb_1,bttb_2,bttb_3: numpy array - rows of the
    sensitivity matrix needed to calculate the BCCB eigenvalues.

    output
    cev: numpy array - eigenvalues of the BCCB matrix.
    '''
    # First column of BCCB
    BCCB = np.zeros(4*N, dtype='complex128')
    q = shape[0]-1
    k = 2*shape[0]-1
    for i in range (shape[0]):
        block_2 = bttb_2[shape[1]*(q):shape[1]*(q+1)]
        block_3 = bttb_3[shape[1]*(q):shape[1]*(q+1)]
        c_2 = block_2[::-1]
        BCCB[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_3[::-1],
        0,c_2[:-1]), axis=None)
        q -= 1
    
        if i > 0:
            block_0 = bttb_0[shape[1]*(i):shape[1]*(i+1)]
            block_1 = bttb_1[shape[1]*(i):shape[1]*(i+1)]
            c_0 = block_0[::-1]
            BCCB[shape[1]*(2*k):shape[1]*(2*k+2)] = np.concatenate((block_1[::-1],
            0,c_0[:-1]), axis=None)
            k -= 1
    
    BCCB = BCCB.reshape(2*shape[0],2*shape[1]).T
    cev_mag = np.fft.fft2(BCCB)
    return cev_mag

def cev_mag_row(bttb_0,bttb_1,bttb_2,bttb_3,shape,N):
    '''
    Calculates the eigenvalues of the BCCB matrix transposed.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    bttb_0,bttb_1,bttb_2,bttb_3: numpy array - rows of the
    sensitivity matrix needed to calculate the BCCB eigenvalues.

    output
    cev: numpy array - eigenvalues of the BCCB matrix.
    '''
    # First row of BCCB
    BCCB = np.zeros(4*N, dtype='complex128')
    q = shape[0]-2
    k = 2*shape[0]-1
    for i in range (shape[0]): # Upper part of BCCB first column
        block_0 = bttb_0[shape[1]*(i):shape[1]*(i+1)]
        block_1 = bttb_1[shape[1]*(i):shape[1]*(i+1)]
        BCCB[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((block_0,
        0,block_1[:-1]), axis=None)

        if i > 0: # Lower part of BCCB first column
            block_2 = bttb_2[shape[1]*(q):shape[1]*(q+1)]
            block_3 = bttb_3[shape[1]*(q):shape[1]*(q+1)]
            BCCB[shape[1]*(2*k):shape[1]*(2*k+2)] = np.concatenate((block_2,
            0,block_3[:-1]), axis=None)
            q -= 1
            k -= 1
    
    BCCB = BCCB.reshape(2*shape[0],2*shape[1]).T
    cev_mag_row = np.fft.fft2(BCCB)
    return cev_mag_row

def fast_forward_bccb_mag(cev_mag,p,shape,N):
    '''
    Matrix-vector product where the matrix has a structure
    of a BTTB.

    input
    cev_mag: numpy array - eigenvalues of the BCCB matrix.
    p: the physical property distribution
    (magnetic moment Am^2).
    shape: tuple - grid size.
    N: scalar - number of observation points.

    output
    p_cg: numpy array - the estimate physical property
    '''
    # First column of BCCB
    v = np.zeros(4*N, dtype='complex128')

    for i in range (shape[0]):
        v[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((p[shape[1]*
        (i):shape[1]*(i+1)], np.zeros(shape[1])), axis=None)

    v = v.reshape(2*shape[0],2*shape[1]).T
    # Matrix-vector product
    dobs_bccb = np.fft.ifft2(np.fft.fft2(v)*cev_mag)
    dobs_bccb = np.ravel(np.real(dobs_bccb[:shape[1],:shape[0]]).T)
    return dobs_bccb

def cgls_loop(A,shape,N,data,itmax):
    '''
    Linear conjugate gradient least squares iterative method.

    input
    cev_mag: numpy array - eigenvalues of the BCCB matrix.
    cev_mag_row: numpy array - eigenvalues of the transposed BCCB matrix.
    shape: tuple - grid size.
    N: scalar - number of observation points.
    data: numpy array - the total-field anomaly 
    potential field data at the x,y and z grid points.
    itmax: scalar - number of iterations of the CGLS method.

    output
    m: numpy array - the estimate physical property
    distribution (magnetic moment Am^2).

    '''
    # Vector of initial guess of parameters
    m = np.ones(N)
    # Matrix-vector product with initial guess
    A_m = A.dot(m)
    # Residual
    s = data - A_m
    res_new = A.T.dot(s)
    #res_list = []
    # Basis vector mutually conjugate with respect to sensitivity matrix
    p = np.zeros(N)
    norm_res_new = 1.
    # Conjugate gradient loop
    for i in range (itmax):
        norm_res_old = norm_res_new
        norm_res_new = res_new.dot(res_new)
        #norm_res = np.linalg.norm(data - fast_forward_bccb_mag(cev_mag,m,shape,N))/N
        #res_list.append(norm_res)
        #if i > 0 and res_list[i-1]-res_list[i] < 10**(-4):
        #    print (i)
        #    break
        Beta = norm_res_new/norm_res_old
        p = res_new + Beta*p
        A_p = A.dot(p)
        t_k = norm_res_new/(A_p.dot(A_p)) # Step size parameter
        m = m + t_k*p
        s = s - t_k*A_p
        res_new = A.T.dot(s)
    return m
    
def cgls_bccb_loop(cev_mag,cev_mag_row,shape,N,data,itmax):
    '''
    Linear conjugate gradient least squares iterative method.

    input
    cev_mag: numpy array - eigenvalues of the BCCB matrix.
    cev_mag_row: numpy array - eigenvalues of the transposed BCCB matrix.
    shape: tuple - grid size.
    N: scalar - number of observation points.
    data: numpy array - the total-field anomaly 
    potential field data at the x,y and z grid points.
    itmax: scalar - number of iterations of the CGLS method.

    output
    m: numpy array - the estimate physical property
    distribution (magnetic moment Am^2).

    '''
    # Vector of initial guess of parameters
    m = np.zeros(N)
    # Matrix-vector product with initial guess
    A_m = fast_forward_bccb_mag(cev_mag,m,shape,N)
    # Residual
    s = data - A_m
    res_new = fast_forward_bccb_mag(cev_mag_row,s,shape,N)
    #res_list = []
    # Basis vector mutually conjugate with respect to sensitivity matrix
    p = np.zeros(N)
    norm_res_new = 1.
    # Conjugate gradient loop
    for i in range (itmax):
        norm_res_old = norm_res_new
        norm_res_new = res_new.dot(res_new)
        #norm_res = np.linalg.norm(data - fast_forward_bccb_mag(cev_mag,m,shape,N))/N
        #res_list.append(norm_res)
        #if i > 0 and res_list[i-1]-res_list[i] < 10**(-4):
        #    print (i)
        #    break
        Beta = norm_res_new/norm_res_old
        p = res_new + Beta*p
        A_p = fast_forward_bccb_mag(cev_mag,p,shape,N)
        #p_A = fast_forward_bccb_mag(cev_mag_row,p,shape,N)
        t_k = norm_res_new/(A_p.dot(A_p)) # Step size parameter
        m = m + t_k*p
        s = s - t_k*A_p
        res_new = fast_forward_bccb_mag(cev_mag_row,s,shape,N)
    return m