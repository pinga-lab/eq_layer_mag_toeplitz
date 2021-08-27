from __future__ import division
import numpy as np
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
    datap: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == zj.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    #assert zj.all() > z.all(), 'The equivalent layer must be beneath\
	#the observation points'

    #Calculates the first and last lines of the first and last line of
    #blocks of the sensitivity matrix
    bttb_0, bttb_1, bttb_2, bttb_3 = bttb_mag(x,y,z,zj,F,h,shape)

    N = shape[0]*shape[1]

    #Calculates the eigenvalues of BCCB matrix
    cev = cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)

    #CG inversion method Equivalent layer
    p_cg = cg_eq_mag(cev,shape,N,data,itmax)

    #Final predicted data
    datap = fast_forward_bccb_mag(cev,p_cg,shape,N)

    return p_cg, datap

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

    if N < 150000:
        # Calculates the first and last rows of the first and last rows of
        # blocks of the sensitivity matrix for N < 150 000
        bttb_0, bttb_1, bttb_2, bttb_3 = bttb_mag(x,y,z,zj,F,h,shape)
        # Calculates the eigenvalues of BCCB and the transposed matrix
        cev = cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
        cev_row = cev_mag_row(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
    else:
        # Calculates the first row of each component of the second derivatives
        # of the function 1/r
        hxx,hxy,hxz,hyy,hyz,hzz = h_bttb_mag(x,y,z,zj,F,h,shape)
        # Calculates the eigenvalues of BCCB and the transposed matrix
        cev = ones_cev_mag(hxx,hxy,hxz,hyy,hyz,hzz,shape,N,F,h)
        cev_row = ones_cev_mag_row(hxx,hxy,hxz,hyy,hyz,hzz,shape,N,F,h)

    #CGLS inversion method for Equivalent layer
    p_cgls = cgls_bccb_loop(cev,cev_row,shape,N,data,itmax)

    #Final predicted data
    datap = fast_forward_bccb_mag(cev,p_cgls,shape,N)

    return p_cgls, datap

def cgls_eq_bccb_mag_tikho_00(x,y,z,zj,shape,data,F,h,itmax):
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

    #Calculates the first and last rows of the first and last rows of
    #blocks of the sensitivity matrix
    #bttb_0, bttb_1, bttb_2, bttb_3 = bttb_mag(x,y,z,zj,F,h,shape)
    hxx,hxy,hxz,hyy,hyz,hzz = h_bttb_mag(x,y,z,zj,F,h,shape)

    N = shape[0]*shape[1]

    #Calculates the eigenvalues of BCCB matrix
    #cev = cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
    #cev_row = cev_mag_row(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
    
    cev = ones_cev_mag(hxx,hxy,hxz,hyy,hyz,hzz,shape,N)
    cev_row = ones_cev_mag_row(hxx,hxy,hxz,hyy,hyz,hzz,shape,N)

    #CGLS inversion method Equivalent layer
    p_cgls = cgls_eq_mag_tikho_00(cev,cev_row,shape,N,data,itmax)

    #Final predicted data
    datap = fast_forward_bccb_mag(cev,p_cgls,shape,N)

    return p_cgls, datap

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
    datap: numpy array - the predicted data.
    '''
    assert x.size == y.size == z.size == zj.size == data.size, 'x, y,\
    z, h and data must have the same number of elements'
    #assert zj.all() > z.all(), 'The equivalent layer must be beneath\
	#the observation points'

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
    datap = fast_forward_bccb_mag(cev,p_cgls,shape,N)

    return p_cgls, datap
    
def lsqr_eq_bccb_mag_tikho_0(x,y,z,zj,shape,data,F,h,itmax):
    '''
    This code for LSQR method is a modification of the one found
    in the Scipy library of Python. The following description is the
    same as found in the Scipy code, wich is the same as described by
    Paige in his work [1,2].
    
    All modifications are related to the new calculation of the forward
    problem using the eigenvalues of the diagonalized circulant matrix
    at each iteration.
    
    Find the least-squares solution to a large, sparse, linear system
    of equations.

    The function solves ``Ax = b``  or  ``min ||b - Ax||^2`` or
    ``min ||Ax - b||^2 + d^2 ||x||^2``.

    The matrix A may be square or rectangular (over-determined or
    under-determined), and may have any rank.

    ::

      1. Unsymmetric equations --    solve  A*x = b

      2. Linear least squares  --    solve  A*x = b
                                     in the least-squares sense

      3. Damped least squares  --    solve  (   A    )*x = ( b )
                                            ( damp*I )     ( 0 )
                                     in the least-squares sense

    Notes
    -----
    LSQR uses an iterative method to approximate the solution.  The
    number of iterations required to reach a certain accuracy depends
    strongly on the scaling of the problem.  Poor scaling of the rows
    or columns of A should therefore be avoided where possible.

    For example, in problem 1 the solution is unaltered by
    row-scaling.  If a row of A is very small or large compared to
    the other rows of A, the corresponding row of ( A  b ) should be
    scaled up or down.

    In problems 1 and 2, the solution x is easily recovered
    following column-scaling.  Unless better information is known,
    the nonzero columns of A should be scaled so that they all have
    the same Euclidean norm (e.g., 1.0).

    In problem 3, there is no freedom to re-scale if damp is
    nonzero.  However, the value of damp should be assigned only
    after attention has been paid to the scaling of A.

    The parameter damp is intended to help regularize
    ill-conditioned systems, by preventing the true solution from
    being very large.  Another aid to regularization is provided by
    the parameter acond, which may be used to terminate iterations
    before the computed solution becomes very large.

    If some initial estimate ``x0`` is known and if ``damp == 0``,
    one could proceed as follows:

      1. Compute a residual vector ``r0 = b - A*x0``.
      2. Use LSQR to solve the system  ``A*dx = r0``.
      3. Add the correction dx to obtain a final solution ``x = x0 + dx``.

    This requires that ``x0`` be available before and after the call
    to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
    to solve A*x = b and k2 iterations to solve A*dx = r0.
    If x0 is "good", norm(r0) will be smaller than norm(b).
    If the same stopping tolerances atol and btol are used for each
    system, k1 and k2 will be similar, but the final solution x0 + dx
    should be more accurate.  The only way to reduce the total work
    is to use a larger stopping tolerance for the second system.
    If some value btol is suitable for A*x = b, the larger value
    btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.

    Preconditioning is another way to reduce the number of iterations.
    If it is possible to solve a related system ``M*x = b``
    efficiently, where M approximates A in some helpful way (e.g. M -
    A has low rank or its elements are small relative to those of A),
    LSQR may converge more rapidly on the system ``A*M(inverse)*z =
    b``, after which x can be recovered by solving M*x = z.

    If A is symmetric, LSQR should not be used!

    Alternatives are the symmetric conjugate-gradient method (cg)
    and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
    applies to any symmetric A and will converge more rapidly than
    LSQR.  If A is positive definite, there are other implementations
    of symmetric cg that require slightly less work per iteration than
    SYMMLQ (but will take the same number of iterations).

    References
    ----------
    .. [1] C. C. Paige and M. A. Saunders (1982a).
           "LSQR: An algorithm for sparse linear equations and
           sparse least squares", ACM TOMS 8(1), 43-71.
    .. [2] C. C. Paige and M. A. Saunders (1982b).
           "Algorithm 583.  LSQR: Sparse linear equations and least
           squares problems", ACM TOMS 8(2), 195-209.
    .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
           systems using LSQR and CRAIG", BIT 35, 588-604.

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

    #Calculates the first and last rows of the first and last rows of
    #blocks of the sensitivity matrix
    bttb_0, bttb_1, bttb_2, bttb_3 = bttb_mag(x,y,z,zj,F,h,shape)

    N = shape[0]*shape[1]

    #Calculates the eigenvalues of BCCB matrix
    cev = cev_mag(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)
    cev_row = cev_mag_row(bttb_0,bttb_1,bttb_2,bttb_3,shape,N)

    #LSQR inversion method Equivalent layer
    p_lsqr = lsqr_eq_mag_tikho_0(cev,cev_row,shape,N,data,itmax)

    #Final predicted data
    datap = fast_forward_bccb_mag(cev,p_lsqr,shape,N)

    return p_lsqr, datap

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

def cg_eq_mag(cev_mag,shape,N,data,itmax):
    '''
    Linear conjugate gradient iterative method.

    input
    cev_mag: numpy array - eigenvalues of the BCCB matrix.
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
    # Basis vector mutually conjugate with respect to sensitivity matrix
    p = np.zeros(N)
    norm_res_new = 1.
    # Conjugate gradient loop
    for i in range (itmax):
        norm_res_old = norm_res_new
        norm_res_new = res_new.dot(res_new)
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
    # Basis vector mutually conjugate with respect to sensitivity matrix
    p = np.zeros(N)
    alpha = 10**-(20)
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
        #t_k = norm_res_new/(p.dot(A_p_tik))
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
    delta = 0.0001
    rho_new = 1.
    alpha = 0.
    norm_data = data.dot(data)
    A_m = fast_forward_bccb_mag(cev_mag,m,shape,N)
    res = data - A_m
    # ICGR outter loop
    for j in range (itmax):
        s = fast_forward_bccb_mag(cev_mag_row,res,shape,N)
        rho_0 = s.dot(s)
        z = m
        l = 1.
        # ICGR inner loop
        while l <= 20 and np.sqrt(rho_new) > delta*np.sqrt(rho_0):
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
            #alpha = 0.0031
            #print alpha
            z = z + alpha*p
            s = s - alpha*w
            rho_new = res.dot(res)
            #rho_new = s.dot(s)
            #print rho_new
            l = l + 1
        rres = np.linalg.norm(m-z)/np.linalg.norm(z)
        m = z
        A_m = fast_forward_bccb_mag(cev_mag,m,shape,N)
        res = data - A_m
        alpha = (res.dot(res))/(2*norm_data)
        #alpha = 0.013
        #print alpha
        if rres < 10**(-12):
            break
    return m

def pre_cg_eq_mag(cev_mag,pre_cev,shape,N,data,itmax):
    '''
    Linear conjugate gradient iterative method with preconditioned BCCB.

    input
    cev_mag: numpy array - eigenvalues of the BCCB matrix.
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
    
def lsqr_eq_mag_tikho_0(cev_mag,cev_mag_row,shape,N,tf,itmax):
    '''
    LSQR method for solving linear sparsed nonsymmetric systems.

    input
    cev_mag: numpy array - eigenvalues of the BCCB matrix.
    cev_mag_row: numpy array - eigenvalues of the transposed BCCB matrix.
    shape: tuple - grid size.
    N: scalar - number of observation points.
    data: numpy array - the total-field anomaly
    potential field data at the x,y and z grid points.
    itmax: scalar - number of iterations of the CG method.

    output
    p_cg: numpy array - the estimate physical property
    distribution (magnetic moment Am^2).

    '''
    eps = np.finfo(np.float64).eps
    it = 0.
    damp=0.0
    atol=1e-8
    btol=1e-8
    conlim=1e8

    itn = 0.
    istop = 0.
    ctol = 0.
    if conlim > 0.:
        ctol = 1./conlim
    anorm = 0.
    acond = 0.
    dampsq = damp**2.
    ddnorm = 0.
    res2 = 0.
    xnorm = 0.
    xxnorm = 0.
    z = 0.
    cs2 = -1.
    sn2 = 0.

    u = tf
    norm = np.linalg.norm(tf)
    p_cg = np.zeros(N)
    beta = norm.copy()
    u = (1./beta) * u
    v = fast_forward_bccb_mag(cev_mag_row,u,shape,N)
    alfa = np.linalg.norm(v)
    v = (1./alfa) * v
    w = v.copy()

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    while it < itmax:
        u = fast_forward_bccb_mag(cev_mag,v,shape,N) - alfa * u
        beta = np.linalg.norm(u)

        if beta > 0.:
            u = (1./beta) * u
            anorm = np.sqrt(anorm**2 + alfa**2 + beta**2 + damp**2)
            v = fast_forward_bccb_mag(cev_mag_row,u,shape,N) - beta * v
            alfa = np.linalg.norm(v)
            if alfa > 0.:
                v = (1. / alfa) * v

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matr
        rhobar1 = np.sqrt(rhobar**2 + damp**2)
        cs1 = rhobar / rhobar1
        sn1 = damp / rhobar1
        psi = sn1 * phibar
        phibar = cs1 * phibar

        # Use a plane rotation to eliminate the subdiagonal element (bet
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal mat
        if beta == 0.:
            cs, sn, rho = np.sign(rhobar1), 0., np.abs(rhobar1)
        elif rhobar1 == 0.:
            cs, sn, rho =  0., np.sign(beta), np.abs(beta)
        elif np.abs(beta) > np.abs(rhobar1):
            tau = rhobar1 / beta
            sn = np.sign(beta) / np.sqrt(1. + tau * tau)
            cs = sn * tau
            rho = beta / sn
        else:
            tau = beta / rhobar1
            cs = np.sign(rhobar1) / np.sqrt(1+tau*tau)
            sn = cs * tau
            rho = rhobar1 / cs

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update p_cg and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1. / rho) * w

        p_cg = p_cg + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + np.linalg.norm(dk)**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(p_cg).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = np.sqrt(xxnorm + zbar**2)
        gamma = np.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * np.sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = np.sqrt(res1 + res2)
        arnorm = alfa * np.abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||p_cg||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||p_cg||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq = rnorm**2 - dampsq * xxnorm
        r1norm = np.sqrt(np.abs(r1sq))
        if r1sq < 0.:
            r1norm = -r1norm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / norm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1. / (acond + eps)
        t1 = test1 / (1. + anorm * xnorm / norm)
        rtol = btol + atol * anorm * xnorm / norm

        it += 1.
    return p_cg