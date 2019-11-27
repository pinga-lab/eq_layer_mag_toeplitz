from __future__ import division
import numpy as np
from fatiando.constants import G, SI2MGAL
from scipy.sparse import diags
from scipy.linalg import toeplitz
from numpy.linalg import inv,norm

def cg_eq_bccb_mag(x,y,z,zj,shape,data,F,h,itmax):
    '''
    Calculates the estimate physical property distribution of
    an equivalent layer that repreduces a total-field anomaly
    data through the conjugate gradient iterative 
    method [1].
    This implementation uses a fast way to calculate the forward
    problem at each iteration taking advantage of the BTTB
    (Block-Toeplitz Toreplitz-Block) structures.

    [1]Aster, Richard C., Brian Borchers, and Clifford H. Thurber.
    Parameter estimation and inverse problems. Elsevier, 2018.
    p.164

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
    tfp: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == zj.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    assert zj.all() > z.all(), 'The equivalent layer must be beneath\
	the observation points'

    #Calculates the first and last lines of the first and last line of
    #blocks of the sensitivity matrix
    bttb_0, bttb_1, bttb_2, bttb_3 = bttb_mag(x,y,z,zj,F,h,shape)

    N = shape[0]*shape[1]

    #Calculates the eigenvalues of BCCB matrix
    cev = cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)

    #CG inversion method Equivalent layer
    p_cg = cg_eq_mag(cev,shape,N,data,itmax)

    #Final predicted data
    tfp = fast_forward_bccb_mag(cev,p_cg,shape,N)

    return p_cg, tfp

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
    tfp: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == zj.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    assert zj.all() > z.all(), 'The equivalent layer must be beneath\
	the observation points'

    #Calculates the first and last rows of the first and last rows of
    #blocks of the sensitivity matrix
    bttb_0, bttb_1, bttb_2, bttb_3 = bttb_mag(x,y,z,zj,F,h,shape)

    N = shape[0]*shape[1]

    #Calculates the eigenvalues of BCCB matrix
    cev = cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
    cev_row = cev_mag_row(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)

    #CGLS inversion method Equivalent layer
    p_cgls = cgls_eq_mag(cev,cev_row,shape,N,data,itmax)

    #Final predicted data
    tfp = fast_forward_bccb_mag(cev,p_cgls,shape,N)

    return p_cgls, tfp
    
def cgls_eq_bccb_mag_tikho_0(x,y,z,zj,shape,data,F,h,itmax):
    '''
    Calculates the estimate physical property distribution of
    an equivalent layer that repreduces a total-field anomaly
    data through the conjugate gradient least square iterative
    method [2] and the Tikhonov order 0 regularization [3].
    This implementation uses a fast way to calculate the forward
    problem at each iteration taking advantage of the BTTB
    (Block-Toeplitz Toreplitz-Block) structures.

    [2]Aster, Richard C., Brian Borchers, and Clifford H. Thurber. 
    Parameter estimation and inverse problems. Elsevier, 2018.
    p.166

    [3]ZHANG, Jianjun; WANG, Qin. An iterative conjugate gradient
    regularization method for image restoration. Journal of
    Information and Computing Science, v. 5, n. 1, 2010.
    p. 055-062.

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
    tfp: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == zj.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    assert zj.all() > z.all(), 'The equivalent layer must be beneath\
	the observation points'

    #Calculates the first and last rows of the first and last rows of
    #blocks of the sensitivity matrix
    bttb_0, bttb_1, bttb_2, bttb_3 = bttb_mag(x,y,z,zj,F,h,shape)

    N = shape[0]*shape[1]

    #Calculates the eigenvalues of BCCB matrix
    cev = cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
    cev_row = cev_mag_row(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)

    #CGLS inversion method Equivalent layer
    p_cgls = cgls_eq_mag_tikho_0(cev,cev_row,shape,N,data,itmax)

    #Final predicted data
    tfp = fast_forward_bccb_mag(cev,p_cgls,shape,N)

    return p_cgls, tfp

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
    tf: numpy array - the predicted data.
    '''
    A = np.empty((N, N), dtype=np.float)
    for i in xrange (N):
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
    I = np.identity(N)
    ATA = A.T.dot(A)
    mu = (np.trace(ATA)/N)*10**(-4)
    AI = inv(ATA+mu*I)
    p = (AI.dot(A.T)).dot(data)
    tf = A.dot(p)
    return p, tf

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
    for i in xrange (shape[0]):
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
    q = shape[0]-1
    k = 2*shape[0]-1
    for i in xrange (shape[0]): # Upper part of BCCB first column
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

    for i in xrange (shape[0]):
        v[shape[1]*(2*i):shape[1]*(2*i+2)] = np.concatenate((p[shape[1]*
        (i):shape[1]*(i+1)], np.zeros(shape[1])), axis=None)

    v = v.reshape(2*shape[0],2*shape[1]).T
    # Matrix-vector product
    dobs_bccb = np.fft.ifft2(np.fft.fft2(v)*cev_mag)
    dobs_bccb = np.ravel(np.real(dobs_bccb[:shape[1],:shape[0]]).T)
    return dobs_bccb

def cg_eq_mag(cev_mag,shape,N,data,itmax):
    '''
    Linear conjugate gradient iterative method.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    data: numpy array - the total-field anomaly
    potential field data at the x,y and z grid points.
    itmax: scalar - number of iterations of the CG method.

    output
    p_cg: numpy array - the estimate physical property
    distribution (magnetic moment Am^2).

    '''
    # Vector of initial guess of parameters
    p_cg = np.ones(N)
    # Matrix-vector product with initial guess
    A_p_cg = fast_forward_bccb_mag(cev_mag,p_cg,shape,N)
    # Residual
    res_new = data - A_p_cg
    # Basis vector mutually conjugate with respect to A_p_cg
    p = np.zeros(N)
    norm_res_new = 1.
    # Conjugate gradient loop
    for i in range (itmax):
        norm_res_old = norm_res_new
        norm_res_new = res_new.dot(res_new)
        Beta = norm_res_new/norm_res_old
        p = res_new + Beta*p
        s = fast_forward_bccb_mag(cev_mag,p,shape,N)
        t_k = norm_res_new/(p.dot(s)) # Step size parameter
        p_cg = p_cg + t_k*p
        res_new = res_new - t_k*s
    return p_cg

def cgls_eq_mag(cev_mag,cev_mag_row,shape,N,data,itmax):
    '''
    Linear conjugate gradient least squares iterative method.

    input
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
    A_m = fast_forward_bccb_mag(cev_mag,m,shape,N)
    # Residual
    s = data - A_m
    res_new = fast_forward_bccb_mag(cev_mag_row,s,shape,N)
    # Basis vector mutually conjugate with respect to sensitivity matrix
    p = np.zeros(N)
    norm_res_new = 1.
    # Conjugate gradient loop
    for i in range (itmax):
        norm_res_old = norm_res_new
        norm_res_new = res_new.dot(res_new)
        Beta = norm_res_new/norm_res_old
        p = res_new + Beta*p
        A_p = fast_forward_bccb_mag(cev_mag,p,shape,N)
        #p_A = fast_forward_bccb_mag(cev_mag_row,p,shape,N)
        t_k = norm_res_new/(A_p.dot(A_p)) # Step size parameter
        m = m + t_k*p
        s = s - t_k*A_p
        res_new = fast_forward_bccb_mag(cev_mag_row,s,shape,N)
    return m
    
def cgls_eq_mag_tikho_00(cev_mag,cev_mag_row,shape,N,data,itmax):
    '''
    Linear conjugate gradient least squares iterative method
    using the Tikhonov order 0 regularization.

    input
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
    # Basis vector mutually conjugate with respect to sensitivity matrix
    p = np.zeros(N)
    alpha = 10**-(4.965)
    # Matrix-vector product with initial guess
    A_m = fast_forward_bccb_mag(cev_mag,m,shape,N)
    A_m_tik = fast_forward_bccb_mag(cev_mag_row,A_m,shape,N) + alpha*p
    # Residual
    s = data - A_m_tik
    res_new = fast_forward_bccb_mag(cev_mag_row,s,shape,N)
    norm_res_new = 1.
    # Conjugate gradient loop
    for i in range (itmax):
        norm_res_old = norm_res_new
        norm_res_new = res_new.dot(res_new)
        Beta = norm_res_new/norm_res_old
        p = res_new + Beta*p
        A_p = fast_forward_bccb_mag(cev_mag,p,shape,N)
        A_p_tik = fast_forward_bccb_mag(cev_mag_row,A_p,shape,N) + alpha*p
        t_k = norm_res_new/(A_p_tik.dot(A_p_tik)) # Step size parameter
        m = m + t_k*p
        s = s - t_k*A_p_tik
        res_new = fast_forward_bccb_mag(cev_mag_row,s,shape,N)
    return m
    
def cgls_eq_mag_tikho_0(cev_mag,cev_mag_row,shape,N,data,itmax):
    '''
    Linear conjugate gradient least squares iterative method
    using the Tikhonov order 0 regularization. The algorithm
    ICGR can be found in [3].
    
    [3]ZHANG, Jianjun; WANG, Qin. An iterative conjugate gradient
    regularization method for image restoration. Journal of
    Information and Computing Science, v. 5, n. 1, 2010.
    p. 055-062.

    input
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
    alpha = 0.
    norm_data = data.dot(data)
    A_m = fast_forward_bccb_mag(cev_mag,m,shape,N)
    res = data - A_m
    # ICGR outter loop
    for j in range (itmax):
        s = fast_forward_bccb_mag(cev_mag_row,res,shape,N)
        rho_0 = s.dot(s)
        l = 1.
        # ICGR inner loop
        while l <= 3: #or np.sqrt(rho_old) < 0.0001*np.sqrt(rho_0):
            if l == 1:
                beta = 0.
                p = s
                rho_old = rho_0
            else:
                rho_old_2 = rho_old
                rho_old = rho_new
                beta = rho_old/rho_old_2
                p = s + beta*p
            q = fast_forward_bccb_mag(cev_mag,p,shape,N)
            w = fast_forward_bccb_mag(cev_mag_row,q,shape,N) + alpha*p
            alpha = rho_old/(p.dot(w))
            z = m + alpha*p
            s = s - alpha*w
            rho_new = res.dot(res)
            l = l + 1
        rres = np.linalg.norm(m-z)/np.linalg.norm(z)
        m = z
        A_m = fast_forward_bccb_mag(cev_mag,m,shape,N)
        res = data - A_m
        alpha = np.linalg.norm(res)/(2*norm_data)
        if rres < 10**(-12):
            break
    return m

def pre_cg_eq_mag(cev_mag,pre_cev,shape,N,data,itmax):
    '''
    Linear conjugate gradient iterative method with preconditioned BCCB.

    input
    shape: tuple - grid size.
    N: scalar - number of observation points.
    data: numpy array - the total-field anomaly
    potential field data at the x,y and z grid points.
    itmax: scalar - number of iterations of the CG method.

    output
    p_cg: numpy array - the estimate physical property
    distribution (magnetic moment Am^2).

    '''
    # Vector of initial guess of parameters
    p_cg = np.ones(N)
    # Matrix-vector product with initial guess
    A_p_cg = fast_forward_bccb_mag(cev_mag,p_cg,shape,N)
    # Residual
    res_new = data - A_p_cg
    # Basis vector mutually conjugate with respect to A_p_cg
    d = np.zeros_like(res_new)
    norm_res_new = 1.
    # Conjugate gradient loop
    for i in range (itmax):
        pre = fast_forward_bccb_mag(pre_cev,res_new,shape,N)
        norm_res_old = norm_res_new
        norm_res_new = pre.dot(res_new)
        Beta = norm_res_new/norm_res_old
        d = pre + Beta*d
        s = fast_forward_bccb_mag(cev_mag,d,shape,N)
        t_k = norm_res_new/(d.dot(s)) # Step size parameter
        p_cg = p_cg + t_k*d
        res_new = res_new - t_k*s
    return p_cg