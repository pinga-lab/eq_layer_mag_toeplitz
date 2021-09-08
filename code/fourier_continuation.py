from __future__ import division
import warnings
import numpy

def upcontinue(x, y, data, shape, height):
    r"""
    Upward continuation of potential field data.

    Calculates the continuation through the Fast Fourier Transform in the
    wavenumber domain (Blakely, 1996):

    .. math::

        F\{h_{up}\} = F\{h\} e^{-\Delta z |k|}

    and then transformed back to the space domain. :math:`h_{up}` is the upward
    continue data, :math:`\Delta z` is the height increase, :math:`F` denotes
    the Fourier Transform,  and :math:`|k|` is the wavenumber modulus.

    .. note:: Requires gridded data.

    .. note:: x, y, z and height should be in meters.

    .. note::

        It is not possible to get the FFT of a masked grid. The default
        :func:`fatiando.gridder.interp` call using minimum curvature will not
        be suitable.  Use ``extrapolate=True`` or ``algorithm='nearest'`` to
        get an unmasked grid.

    Parameters:

    * x, y : 1D-arrays
        The x and y coordinates of the grid points
    * data : 1D-array
        The potential field at the grid points
    * shape : tuple = (nx, ny)
        The shape of the grid
    * height : float
        The height increase (delta z) in meters.

    Returns:

    * cont : array
        The upward continued data

    References:

    Blakely, R. J. (1996), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """
    assert x.shape == y.shape, \
        "x and y arrays must have same shape"
    if height <= 0:
        warnings.warn("Using 'height' <= 0 means downward continuation, " +
                      "which is known to be unstable.")
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    #padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape)
    kz = numpy.sqrt(kx**2 + ky**2)
	
    upcont_ft = numpy.fft.fft2(data.reshape(shape))*numpy.exp(-height*kz)
    cont = numpy.real(numpy.fft.ifft2(upcont_ft))
    # Remove padding
    #cont = cont[padx: padx + nx, pady: pady + ny].ravel()
    return cont

def _fftfreqs(x, y, shape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    fx = 2*numpy.pi*numpy.fft.fftfreq(nx, dx)
    dy = (y.max() - y.min())/(ny - 1)
    fy = 2*numpy.pi*numpy.fft.fftfreq(ny, dy)
    return numpy.meshgrid(fy, fx)[::-1]