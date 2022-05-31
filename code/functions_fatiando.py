import copy as cp
import numpy as np

#: The default thermal diffusivity in :math:`m^2/s`
THERMAL_DIFFUSIVITY = 0.000001

#: The default thermal diffusivity but in :math:`m^2/year`
THERMAL_DIFFUSIVITY_YEAR = 31.5576

#: Conversion factor from SI units to Eotvos: :math:`1/s^2 = 10^9\ Eotvos`
SI2EOTVOS = 1000000000.0

#: Conversion factor from SI units to mGal: :math:`1\ m/s^2 = 10^5\ mGal`
SI2MGAL = 100000.0

#: The gravitational constant in :math:`m^3 kg^{-1} s^{-1}`
G = 0.00000000006673

#: Proportionality constant used in the magnetic method in henry/m (SI)
CM = 10. ** (-7)

#: Conversion factor from tesla to nanotesla
T2NT = 10. ** (9)

#: The mean earth radius in meters
MEAN_EARTH_RADIUS = 6378137.0

#: Permeability of free space in :math:`N A^{-2}`
PERM_FREE_SPACE = 4 * \
    3.141592653589793115997963468544185161590576171875 * (10 ** -7)

def utils_ang2vec(intensity, inc, dec):
    return np.transpose([intensity * i for i in utils_dircos(inc, dec)])

def utils_contaminate(data, stddev, percent=False, return_stddev=False, seed=None):
    r"""
    Add pseudorandom gaussian noise to an array.

    Noise added is normally distributed with zero mean.

    Parameters:

    * data : array or list of arrays
        Data to contaminate
    * stddev : float or list of floats
        Standard deviation of the Gaussian noise that will be added to *data*
    * percent : True or False
        If ``True``, will consider *stddev* as a decimal percentage and the
        standard deviation of the Gaussian noise will be this percentage of
        the maximum absolute value of *data*
    * return_stddev : True or False
        If ``True``, will return also the standard deviation used to
        contaminate *data*
    * seed : None or int
        Seed used to generate the pseudo-random numbers. If `None`, will use a
        different seed every time. Use the same seed to generate the same
        random sequence to contaminate the data.

    Returns:

    if *return_stddev* is ``False``:

    * contam : array or list of arrays
        The contaminated data array

    else:

    * results : list = [contam, stddev]
        The contaminated data array and the standard deviation used to
        contaminate it.

    Examples:

    >>> import numpy as np
    >>> data = np.ones(5)
    >>> noisy = contaminate(data, 0.1, seed=0)
    >>> print noisy
    [ 1.03137726  0.89498775  0.95284582  1.07906135  1.04172782]
    >>> noisy, std = contaminate(data, 0.05, seed=0, percent=True,
    ...                          return_stddev=True)
    >>> print std
    0.05
    >>> print noisy
    [ 1.01568863  0.94749387  0.97642291  1.03953067  1.02086391]
    >>> data = [np.zeros(5), np.ones(3)]
    >>> noisy = contaminate(data, [0.1, 0.2], seed=0)
    >>> print noisy[0]
    [ 0.03137726 -0.10501225 -0.04715418  0.07906135  0.04172782]
    >>> print noisy[1]
    [ 0.81644754  1.20192079  0.98163167]

    """
    np.random.seed(seed)
    # Check if dealing with an array or list of arrays
    if not isinstance(stddev, list):
        stddev = [stddev]
        data = [data]
    contam = []
    for i in range(len(stddev)):
        if stddev[i] == 0.:
            contam.append(data[i])
            continue
        if percent:
            stddev[i] = stddev[i] * max(abs(data[i]))
        noise = np.random.normal(scale=stddev[i], size=len(data[i]))
        # Subtract the mean so that the noise doesn't introduce a systematic
        # shift in the data
        noise -= noise.mean()
        contam.append(np.array(data[i]) + noise)
    np.random.seed()
    if len(contam) == 1:
        contam = contam[0]
        stddev = stddev[0]
    if return_stddev:
        return [contam, stddev]
    else:
        return contam

def utils_vec2ang(vector):
    """
    Convert a 3-component vector to intensity, inclination and declination.

    .. note:: Coordinate system is assumed to be x->North, y->East, z->Down.
        Inclination is positive down and declination is measured with respect
        to x (North).

    Parameter:

    * vector : array = [x, y, z]
        The vector

    Returns:

    * [intensity, inclination, declination] : floats
        The intensity, inclination and declination (in degrees)

    Examples::

        >>> s = vec2ang([1.5, 1.5, 2.121320343559643])
        >>> print "%.3f %.3f %.3f" % tuple(s)
        3.000 45.000 45.000

    """
    intensity = np.linalg.norm(vector)
    r2d = 180. / np.pi
    x, y, z = vector
    declination = r2d * np.arctan2(y, x)
    inclination = r2d * np.arcsin(z / intensity)
    return [intensity, inclination, declination]

def utils_dircos(inc, dec):
    """
    Returns the 3 coordinates of a unit vector given its inclination and
    declination.

    .. note:: Coordinate system is assumed to be x->North, y->East, z->Down.
        Inclination is positive down and declination is measured with respect
        to x (North).

    Parameter:

    * inc : float
        The inclination of the vector (in degrees)
    * dec : float
        The declination of the vector (in degrees)

    Returns:

    * vect : list = [x, y, z]
        The unit vector

    """
    d2r = np.pi / 180.
    vect = [np.cos(d2r * inc) * np.cos(d2r * dec),
            np.cos(d2r * inc) * np.sin(d2r * dec),
            np.sin(d2r * inc)]
    return vect

def gridder_regular(area, shape, z=None):
    """
    Create a regular grid.

    The x directions is North-South and y East-West. Imagine the grid as a
    matrix with x varying in the lines and y in columns.

    Returned arrays will be flattened to 1D with ``numpy.ravel``.

    .. warning::

        As of version 0.4, the ``shape`` argument was corrected to be
        ``shape = (nx, ny)`` instead of ``shape = (ny, nx)``.


    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[x, y, z]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    Examples:

    >>> x, y = regular((0, 10, 0, 5), (5, 3))
    >>> print(x)
    [  0.    0.    0.    2.5   2.5   2.5   5.    5.    5.    7.5   7.5   7.5
      10.   10.   10. ]
    >>> print(x.reshape((5, 3)))
    [[  0.    0.    0. ]
     [  2.5   2.5   2.5]
     [  5.    5.    5. ]
     [  7.5   7.5   7.5]
     [ 10.   10.   10. ]]
    >>> print(y.reshape((5, 3)))
    [[ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]]
    >>> x, y = regular((0, 0, 0, 5), (1, 3))
    >>> print(x.reshape((1, 3)))
    [[ 0.  0.  0.]]
    >>> print(y.reshape((1, 3)))
    [[ 0.   2.5  5. ]]
    >>> x, y, z = regular((0, 10, 0, 5), (5, 3), z=-10)
    >>> print(z.reshape((5, 3)))
    [[-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]]


    """
    nx, ny = shape
    x1, x2, y1, y2 = area
    #_check_area(area)
    xs = np.linspace(x1, x2, nx)
    ys = np.linspace(y1, y2, ny)
    # Must pass ys, xs in this order because meshgrid uses the first argument
    # for the columns
    arrays = np.meshgrid(ys, xs)[::-1]
    if z is not None:
        arrays.append(z*np.ones(nx*ny, dtype=np.float))
    return [i.ravel() for i in arrays]

class GeometricElement(object):
    """
    Base class for all geometric elements.
    """

    def __init__(self, props):
        self.props = {}
        if props is not None:
            for p in props:
                self.props[p] = props[p]

    def addprop(self, prop, value):
        """
        Add a physical property to this geometric element.

        If it already has the property, the given value will overwrite the
        existing one.

        Parameters:

        * prop : str
            Name of the physical property.
        * value : float
            The value of this physical property.

        """
        self.props[prop] = value

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)

class mesher_Sphere(GeometricElement):
    """
    A sphere.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    Parameters:

    * x, y, z : float
        The coordinates of the center of the sphere
    * radius : float
        The radius of the sphere
    * props : dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> s = Sphere(1, 2, 3, 10, {'magnetization':200})
        >>> s.props['magnetization']
        200
        >>> s.addprop('density', 20)
        >>> print s.props['density']
        20
        >>> print s
        x:1 | y:2 | z:3 | radius:10 | density:20 | magnetization:200
        >>> s = Sphere(1, 2, 3, 4)
        >>> print s
        x:1 | y:2 | z:3 | radius:4
        >>> s.addprop('density', 2670)
        >>> print s
        x:1 | y:2 | z:3 | radius:4 | density:2670

    """

    def __init__(self, x, y, z, radius, props=None):
        super().__init__(props)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.radius = float(radius)
        self.center = np.array([x, y, z])

    def __str__(self):
        """Return a string representation of the sphere."""
        names = [('x', self.x), ('y', self.y), ('z', self.z),
                 ('radius', self.radius)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)


class mesher_PolygonalPrism(GeometricElement):
    """
    A 3D prism with polygonal cross-section.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    .. note:: *vertices* must be **CLOCKWISE** or will give inverse result.

    Parameters:

    * vertices : list of lists
        Coordinates of the vertices. A list of ``[x, y]`` pairs.
    * z1, z2 : float
        Top and bottom of the prism
    * props :  dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> verts = [[1, 1], [1, 2], [2, 2], [2, 1]]
        >>> p = PolygonalPrism(verts, 0, 3, props={'temperature':25})
        >>> p.props['temperature']
        25
        >>> print p.x
        [ 1.  1.  2.  2.]
        >>> print p.y
        [ 1.  2.  2.  1.]
        >>> print p.z1, p.z2
        0.0 3.0
        >>> p.addprop('density', 2670)
        >>> print p.props['density']
        2670

    """

    def __init__(self, vertices, z1, z2, props=None):
        super().__init__(props)
        self.x = np.fromiter((v[0] for v in vertices), dtype=np.float)
        self.y = np.fromiter((v[1] for v in vertices), dtype=np.float)
        self.z1 = float(z1)
        self.z2 = float(z2)
        self.nverts = len(vertices)

    def topolygon(self):
        """
        Get the polygon describing the prism viewed from above.

        Returns:

        * polygon : :func:`fatiando.mesher.Polygon`
            The polygon

        Example:

            >>> verts = [[1, 1], [1, 2], [2, 2], [2, 1]]
            >>> p = PolygonalPrism(verts, 0, 100)
            >>> poly = p.topolygon()
            >>> print poly.x
            [ 1.  1.  2.  2.]
            >>> print poly.y
            [ 1.  2.  2.  1.]

        """
        verts = np.transpose([self.x, self.y])
        return Polygon(verts, self.props)

def sphere_tf(xp, yp, zp, spheres, inc, dec, pmag=None):
    r"""
    The total-field magnetic anomaly.

    The anomaly is defined as (Blakely, 1995):

    .. math::

        \Delta T = |\mathbf{T}| - |\mathbf{F}|,

    where :math:`\mathbf{T}` is the measured field and :math:`\mathbf{F}` is a
    reference (regional) field.

    The anomaly of a homogeneous sphere can be calculated as:

    .. math::

        \Delta T \approx \hat{\mathbf{F}}\cdot\mathbf{B}.

    where :math:`\mathbf{B}` is the magnetic induction produced by the sphere.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres that are ``None`` or without
        ``'magnetization'`` will be ignored. The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization given
        as a 3-component vector.
    * inc, dec : floats
        The inclination and declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * tf : array
        The total-field anomaly

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    fx, fy, fz = utils_dircos(inc, dec)
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'magnetization' not in sphere.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = sphere.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        # Calculating v_xx, etc to calculate B is ~2x slower than this
        dotprod = mx*x + my*y + mz*z
        bx = (3*dotprod*x - r_sqr*mx)/r_5
        by = (3*dotprod*y - r_sqr*my)/r_5
        bz = (3*dotprod*z - r_sqr*mz)/r_5
        res += volume*(fx*bx + fy*by + fz*bz)
    res *= CM*T2NT
    return res


def sphere_bx(xp, yp, zp, spheres, pmag=None):
    """
    The x component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres that are ``None`` or without
        ``'magnetization'`` will be ignored. The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization given
        as a 3-component vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bx: array
        The x component of the magnetic induction

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'magnetization' not in sphere.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = sphere.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        # Calculating v_xx, etc to calculate B is ~1.3x slower than this
        dotprod = mx*x + my*y + mz*z
        res += volume*(3*dotprod*x - r_sqr*mx)/r_5
    res *= CM * T2NT
    return res


def sphere_by(xp, yp, zp, spheres, pmag=None):
    """
    The y component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres that are ``None`` or without
        ``'magnetization'`` will be ignored. The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization given
        as a 3-component vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * by: array
        The y component of the magnetic induction

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'magnetization' not in sphere.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = sphere.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        # Calculating v_xx, etc to calculate B is ~1.3x slower than this
        dotprod = mx*x + my*y + mz*z
        res += volume*(3*dotprod*y - r_sqr*my)/r_5
    res *= CM * T2NT
    return res


def sphere_bz(xp, yp, zp, spheres, pmag=None):
    """
    The z component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the physical property
        ``'magnetization'``. Spheres that are ``None`` or without
        ``'magnetization'`` will be ignored. The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization given
        as a 3-component vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bz : array
        The z component of the magnetic induction

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'magnetization' not in sphere.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = sphere.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r_sqr = x**2 + y**2 + z**2
        # This is faster than r5 = r_sqrt**2.5
        r = np.sqrt(r_sqr)
        r_5 = r*r*r*r*r
        volume = 4*np.pi*(sphere.radius**3)/3
        # Calculating v_xx, etc to calculate B is ~1.3x slower than this
        dotprod = mx*x + my*y + mz*z
        res += volume*(3*dotprod*z - r_sqr*mz)/r_5
    res *= CM * T2NT
    return res

def polyprism_tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    r"""
    The total-field magnetic anomaly of polygonal prisms.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored.
    * inc : float
        The inclination of the regional field (in degrees)
    * dec : float
        The declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils_dircos(inc, dec)
    res = 0
    for prism in prisms:
        if prism is None:
            continue
        if 'magnetization' not in prism.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        else:
            mx, my, mz = pmag
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        bx = v1*mx + v2*my + v3*mz
        by = v2*mx + v4*my + v5*mz
        bz = v3*mx + v5*my + v6*mz
        res += fx*bx + fy*by + fz*bz
    res *= CM * T2NT
    return res

def polyprism_bx(xp, yp, zp, prisms):
    """
    x component of magnetic induction of a polygonal prism.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * bx: array
        The x component of the magnetic induction

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        res += v1*mx + v2*my + v3*mz
    res *= CM * T2NT
    return res

def polyprism_by(xp, yp, zp, prisms):
    """
    y component of magnetic induction of a polygonal prism.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * by: array
        The y component of the magnetic induction

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v2 = kernelxy(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        res += v2*mx + v4*my + v5*mz
    res *= CM * T2NT
    return res

def polyprism_bz(xp, yp, zp, prisms):
    """
    z component of magnetic induction of a polygonal prism.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.

    Returns:

    * bz: array
        The z component of the magnetic induction

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    res = 0
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props):
            continue
        # Get the magnetization vector components
        mx, my, mz = prism.props['magnetization']
        v3 = kernelxz(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        res += v3*mx + v5*my + v6*mz
    res *= CM * T2NT
    return res

def kernelxx(xp, yp, zp, prism):
    r"""
    The xx second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        n = deltax/deltay
        g = X1 - Y1*n
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        cross = X1*Y2 - X2*Y1
        p = cross/dist + dummy
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        atan_diff_d2 = np.arctan2(Z2*d2, p*R22) - np.arctan2(Z1*d2, p*R21)
        atan_diff_d1 = np.arctan2(Z2*d1, p*R12) - np.arctan2(Z1*d1, p*R11)
        tmp = g*Y2*atan_diff_d2/(p*d2) + n*p*atan_diff_d2/(d2)
        tmp -= g*Y1*atan_diff_d1/(p*d1) + n*p*atan_diff_d1/(d1)
        tmp += n*np.log(
            (Z2 + R12)*(Z1 + R21)/((Z1 + R11)*(Z2 + R22) + dummy) + dummy)
        tmp *= -1/(1 + n*n)
        kernel += tmp
    return kernel


def kernelxy(xp, yp, zp, prism):
    r"""
    The xy second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        n = deltax/deltay
        g = X1 - Y1*n
        g_sqr = g*g
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        cross = X1*Y2 - X2*Y1
        p = cross/dist + dummy
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        atan_diff_d2 = np.arctan2(Z2*d2, p*R22) - np.arctan2(Z1*d2, p*R21)
        atan_diff_d1 = np.arctan2(Z2*d1, p*R12) - np.arctan2(Z1*d1, p*R11)
        tmp = (g_sqr + g*n*Y2)*atan_diff_d2/(p*d2) - p*atan_diff_d2/d2
        tmp -= (g_sqr + g*n*Y1)*atan_diff_d1/(p*d1) - p*atan_diff_d1/d1
        tmp += np.log(
            (Z2 + R22)*(Z1 + R11)/((Z1 + R21)*(Z2 + R12) + dummy) + dummy)
        tmp *= 1/(1 + n*n)
        kernel += tmp
    return kernel


def kernelxz(xp, yp, zp, prism):
    r"""
    The xz second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        n = deltax/deltay
        n_sqr_p1 = n*n + 1
        g = X1 - Y1*n
        ng = n*g
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        # Collapsing these logs decreases the precision too much leading to a
        # larger difference with the prism code.
        log_r22 = np.log((R22 - d2)/(R22 + d2) + dummy)
        log_r21 = np.log((R21 - d2)/(R21 + d2) + dummy)
        log_r12 = np.log((R12 - d1)/(R12 + d1) + dummy)
        log_r11 = np.log((R11 - d1)/(R11 + d1) + dummy)
        log_diff_d1 = (0.5/d1)*(log_r12 - log_r11)
        log_diff_d2 = (0.5/d2)*(log_r22 - log_r21)
        tmp = (Y2*n_sqr_p1 + ng)*log_diff_d2
        tmp -= (Y1*n_sqr_p1 + ng)*log_diff_d1
        tmp *= -1/n_sqr_p1
        kernel += tmp
    return kernel


def kernelyy(xp, yp, zp, prism):
    r"""
    The yy second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        m = deltay/deltax
        c = Y1 - X1*m
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        cross = X1*Y2 - X2*Y1
        p = cross/dist + dummy
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        atan_diff_d2 = np.arctan2(Z2*d2, p*R22) - np.arctan2(Z1*d2, p*R21)
        atan_diff_d1 = np.arctan2(Z2*d1, p*R12) - np.arctan2(Z1*d1, p*R11)
        tmp = c*X2*atan_diff_d2/(p*d2) + m*p*atan_diff_d2/d2
        tmp -= c*X1*atan_diff_d1/(p*d1) + m*p*atan_diff_d1/d1
        tmp += m*np.log(
            (Z2 + R12)*(Z1 + R21)/((Z2 + R22)*(Z1 + R11)) + dummy)
        tmp *= 1/(1 + m*m)
        kernel += tmp
    return kernel


def kernelyz(xp, yp, zp, prism):
    r"""
    The yz second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1 + dummy
        deltay = Y2 - Y1 + dummy
        m = deltay/deltax
        m_sqr_p1 = m*m + 1
        c = Y1 - X1*m
        cm = c*m
        dist = np.sqrt(deltax*deltax + deltay*deltay)
        d1 = (deltax*X1 + deltay*Y1)/dist + dummy
        d2 = (deltax*X2 + deltay*Y2)/dist + dummy
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        # Same remark about collapsing logs as kernelxz
        log_r11 = np.log((R11 - d1)/(R11 + d1) + dummy)
        log_r12 = np.log((R12 - d1)/(R12 + d1) + dummy)
        log_r21 = np.log((R21 - d2)/(R21 + d2) + dummy)
        log_r22 = np.log((R22 - d2)/(R22 + d2) + dummy)
        tmp = (X2*m_sqr_p1 + cm)*(0.5/d2)*(log_r22 - log_r21)
        tmp -= (X1*m_sqr_p1 + cm)*(0.5/d1)*(log_r12 - log_r11)
        tmp *= 1/m_sqr_p1
        kernel += tmp
    return kernel


def kernelzz(xp, yp, zp, prism):
    r"""
    The zz second-derivative of the kernel function :math:`\phi`.

    .. math::

        \phi(x,y,z) = \iiint_\Omega \frac{1}{r}
                      \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

    in which

    .. math::

        r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}.

    This function is used to calculate the gravity gradient tensor, magnetic
    induction, and total field magnetic anomaly.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input and output values in SI!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : object of :class:`fatiando.mesher.PolygonalPrism`
        The model used as the integration domain :math:`\Omega` of the kernel
        function.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    x, y = prism.x, prism.y
    z1, z2 = prism.z1, prism.z2
    nverts = prism.nverts
    # Calculate the effect of the prism
    Z1 = z1 - zp
    Z2 = z2 - zp
    Z1_sqr = Z1*Z1
    Z2_sqr = Z2*Z2
    kernel = 0
    for k in range(nverts):
        X1 = x[k] - xp
        Y1 = y[k] - yp
        X2 = x[(k + 1) % nverts] - xp
        Y2 = y[(k + 1) % nverts] - yp
        deltax = X2 - X1
        deltay = Y2 - Y1
        # dist is only used in divisions. Add dummy to avoid zero division
        # errors if the two vertices coincide.
        dist = np.sqrt(deltax*deltax + deltay*deltay) + dummy
        cross = X1*Y2 - X2*Y1
        p = cross/dist
        d1 = (deltax*X1 + deltay*Y1)/dist
        d2 = (deltax*X2 + deltay*Y2)/dist
        vert1_sqr = X1*X1 + Y1*Y1
        vert2_sqr = X2*X2 + Y2*Y2
        R11 = np.sqrt(vert1_sqr + Z1_sqr)
        R12 = np.sqrt(vert1_sqr + Z2_sqr)
        R21 = np.sqrt(vert2_sqr + Z1_sqr)
        R22 = np.sqrt(vert2_sqr + Z2_sqr)
        kernel += (np.arctan2(Z2*d2, p*R22) - np.arctan2(Z1*d2, p*R21) -
                   np.arctan2(Z2*d1, p*R12) + np.arctan2(Z1*d1, p*R11))
    return kernel

def sphere_gz(xp, yp, zp, spheres, dens=None):
    r"""
    The :math:`g_z` gravitational acceleration component.

    .. math::

        g_z(x, y, z) = \rho 4 \pi \dfrac{radius^3}{3} \dfrac{z - z'}{r^3}

    in which :math:`\rho` is the density and
    :math:`r = \sqrt{(x - x')^2 + (y - y')^2 + (z - z')^2}`.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    All input values should be in SI and output is in mGal.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the field will be calculated
    * spheres : list of :class:`fatiando.mesher.Sphere`
        The spheres. Spheres must have the property ``'density'``. The ones
        that are ``None`` or without a density will be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the spheres. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    res = 0
    for sphere in spheres:
        if sphere is None:
            continue
        if 'density' not in sphere.props and dens is None:
            continue
        if dens is None:
            density = sphere.props['density']
        else:
            density = dens
        x = sphere.x - xp
        y = sphere.y - yp
        z = sphere.z - zp
        r = np.sqrt(x**2 + y**2 + z**2)
        # This is faster than r3 = r_sqrt**1.5
        r_cb = r*r*r
        mass = density*4*np.pi*(sphere.radius**3)/3
        res += mass*z/r_cb
    res *= G*SI2MGAL
    return res

def polyprism_gz(xp, yp, zp, prisms):
    r"""
    z component of gravitational acceleration of a polygonal prism.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    .. note:: All input values in SI units and output in mGal!

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates of the computation points.
    * prisms : list of :class:`fatiando.mesher.PolygonalPrism`
        The model used to calculate the field.
        Prisms must have the physical property ``'density'`` will be
        ignored.

    Returns:

    * res : array
        The effect calculated on the computation points.

    References:

    Plouff, D. , 1976, Gravity and magnetic fields of polygonal prisms and
    applications to magnetic terrain corrections, Geophysics, 41(4), 727-741,
    doi:10.1190/1.1440645.

    """
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    dummy = 1e-10
    res = 0
    for prism in prisms:
        if prism is None or 'density' not in prism.props:
            continue
        x, y = prism.x, prism.y
        z1, z2 = prism.z1, prism.z2
        density = prism.props['density']
        nverts = prism.nverts
        # Calculate the effect of the prism
        Z1 = z1 - zp
        Z2 = z2 - zp
        Z1_sqr = Z1**2
        Z2_sqr = Z2**2
        kernel = 0
        for k in range(nverts):
            Xk1 = x[k] - xp
            Yk1 = y[k] - yp
            Xk2 = x[(k + 1) % nverts] - xp
            Yk2 = y[(k + 1) % nverts] - yp
            p = Xk1*Yk2 - Xk2*Yk1
            p_sqr = p**2
            Qk1 = (Yk2 - Yk1)*Yk1 + (Xk2 - Xk1)*Xk1
            Qk2 = (Yk2 - Yk1)*Yk2 + (Xk2 - Xk1)*Xk2
            Ak1 = Xk1**2 + Yk1**2
            Ak2 = Xk2**2 + Yk2**2
            R1k1 = np.sqrt(Ak1 + Z1_sqr)
            R1k2 = np.sqrt(Ak2 + Z1_sqr)
            R2k1 = np.sqrt(Ak1 + Z2_sqr)
            R2k2 = np.sqrt(Ak2 + Z2_sqr)
            Ak1 = np.sqrt(Ak1)
            Ak2 = np.sqrt(Ak2)
            Bk1 = np.sqrt(Qk1**2 + p_sqr)
            Bk2 = np.sqrt(Qk2**2 + p_sqr)
            E1k1 = R1k1*Bk1
            E1k2 = R1k2*Bk2
            E2k1 = R2k1*Bk1
            E2k2 = R2k2*Bk2
            # Simplifying these arctans with, e.g., (Z2 - Z1)*arctan2(Qk2*p -
            # Qk1*p, p*p + Qk2*Qk1) doesn't work because of the restrictions
            # regarding the angles for that identity. The regression tests
            # fail for some points by a large amount.
            kernel += (Z2 - Z1)*(np.arctan2(Qk2, p) - np.arctan2(Qk1, p))
            kernel += Z2*(np.arctan2(Z2*Qk1, R2k1*p) -
                          np.arctan2(Z2*Qk2, R2k2*p))
            kernel += Z1*(np.arctan2(Z1*Qk2, R1k2*p) -
                          np.arctan2(Z1*Qk1, R1k1*p))
            Ck1 = Qk1*Ak1
            Ck2 = Qk2*Ak2
            # dummy helps prevent zero division and log(0) errors (that's why I
            # need to add it twice)
            # Simplifying these two logs with a single one is not worth it
            # because it would introduce two pow operations.
            kernel += 0.5*p*Ak1/(Bk1 + dummy)*np.log(
                (E1k1 - Ck1)*(E2k1 + Ck1)/((E1k1 + Ck1)*(E2k1 - Ck1) + dummy) +
                dummy)
            kernel += 0.5*p*(Ak2/(Bk2 + dummy))*np.log(
                (E2k2 - Ck2)*(E1k2 + Ck2)/((E2k2 + Ck2)*(E1k2 - Ck2) + dummy) +
                dummy)
        res += kernel*density
    res *= G*SI2MGAL
    return res

def reduce_to_pole(x, y, data, shape, inc, dec, sinc, sdec):
    fx, fy, fz = utils_ang2vec(1, inc, dec)
    if sinc is None or sdec is None:
        mx, my, mz = fx, fy, fz
    else:
        mx, my, mz = utils_ang2vec(1, sinc, sdec)
    kx, ky = [k for k in _fftfreqs(x, y, shape)]
    kz_sqr = kx**2 + ky**2
    a1 = mz*fz - mx*fx
    a2 = mz*fz - my*fy
    a3 = -my*fx - mx*fy
    b1 = mx*fz + mz*fx
    b2 = my*fz + mz*fy
    # The division gives a RuntimeWarning because of the zero frequency term.
    # This suppresses the warning.
    with np.errstate(divide='ignore', invalid='ignore'):
        rtp = (kz_sqr)/(a1*kx**2 + a2*ky**2 + a3*kx*ky +
                        1j*np.sqrt(kz_sqr)*(b1*kx + b2*ky))
    rtp[0, 0] = 0
    ft_pole = rtp*np.fft.fft2(np.reshape(data, shape))
    return np.real(np.fft.ifft2(ft_pole)).ravel()

def _fftfreqs(x, y, shape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    fx = 2*np.pi*np.fft.fftfreq(nx, dx)
    dy = (y.max() - y.min())/(ny - 1)
    fy = 2*np.pi*np.fft.fftfreq(ny, dy)
    return np.meshgrid(fy, fx)[::-1]