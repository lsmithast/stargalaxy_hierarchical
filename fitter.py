import numpy as np
import scipy.signal
import scipy.special
import pandas
import time
import _fitter
import multiprocessing as mp
from cffi import FFI
ffi = FFI()


def gengrid(npix=129, minr=0.1, maxr=25, nr=21,
            minn=0.5, maxn=5, nn=20, pixel_scale=1):
    """
    Generates a grid of models
    Parameters:
    -----------
    npix: int
        Integer size of the grid
    minr: float
        Minimium scale-length of the Sersic model (in ")
    maxr: float
        Maximum scale-length (in ")
    nr: int
        Number of models along the radii axis
    minn: float
        Minimum Sersic index
    maxn: float
        Max Sersic index
    nn: int
        Number of models along Sersic index axis
    pixel_scale:
        Scale of the image in "/px

    Returns:
    --------
    models: numpy array
        The array of models with shape of (nr*nn, npix, npix)
        The 0-th axis is axis of models.
        The fastest changing parameter is the Sersic index

    """
    rads = np.exp(np.linspace(np.log(minr), np.log(maxr), nr, True))
    ns = np.linspace(minn, maxn, nn, True)
    xgrid, ygrid = np.mgrid[0:npix, 0:npix]
    x0 = (npix-1) / 2.
    r0 = (((xgrid - x0)**2 + (ygrid - x0)**2)**.5) * pixel_scale
    bn = (2 * ns[None, :, None, None] - 1./3)
    models = np.exp(- bn * (r0[None, None, :, :] / rads[:, None,
                                                        None, None])**(1 / ns[None, :, None, None]))
    models = models.reshape(-1, npix, npix)
    return models


def getpsf(npix, psfwidth, ell=0.2, pa=0):
    """
    Get a Gaussian psf

    Parameters:
    ----------

    npix: int
        Size of the PSF cutout
    psfwidth: float
        Gaussian sigma of the PSF
    ell: float
        Ellipticity of the PSF
    pa: float
        Positional angle of the PSF

    Returns:
    --------
    psf: numpy array
        Normalized to the sum=1 PSF
    """
    xgrid, ygrid = np.mgrid[0:npix, 0:npix]
    x0 = (npix-1) / 2.
    cpa = np.cos(np.deg2rad(pa))
    spa = np.sin(np.deg2rad(pa))
    X1 = (xgrid-x0)*cpa + (ygrid-x0)*spa
    Y1 = -(xgrid-x0)*spa + (ygrid-x0)*cpa
    r0 = ((X1*np.sqrt(1-ell))**2 + (Y1/np.sqrt(1-ell))**2)**.5
    psf = np.exp(-0.5 * (r0 / psfwidth)**2)
    return psf / psf.sum()


def convolver(grid, psf):
    """
    Convolve a grid of models with the psf

    Parameters:
    -----------
    grid: numpy array
        3-D Array with dimensions (., npix, npix) of input models
    psf: numpy
        2-D array with the PSF

    Returns:
    --------
    grid: numpy
        3-D array of convolved models
    """
    rgrid = grid * 0
    for i in range(len(grid)):
        rgrid[i] = scipy.signal.fftconvolve(grid[i], psf, mode='same')
    return rgrid


def evaluator_c(ima, var, grid):
    """
    Evaluate likelihoods of a grid of models

    Parameters:
    -----------
    ima: numpy
        2-D input image
    var: numpy
        2-D variance image
    grid: numpy
        The 3-D grid of models with the size (.,npix,npix) where
        npix is the size of the image

    Returns:
    --------
    logl: numpy
        Array of log-likelihoods of a grid of models
    """
    t1 = time.time()
    ngrid = len(grid)
    ret = np.zeros(ngrid)
    args = {'ima': ima, 'var': var, 'grid': grid, 'ret': ret}

    Cargs = {}
    Xargs = {}
    for k in args.keys():
        Xargs[k] = np.require(args[k], dtype=np.float32,
                              requirements=["A", "O", "C"])
        Cargs[k] = ffi.cast('float*', Xargs[k].ctypes.data)
    t2 = time.time()

    npix = len(ima)
    _fitter.lib.procobj(Cargs['ima'], Cargs['var'],
                        Cargs['grid'], npix, ngrid, Cargs['ret'])
    t3 = time.time()

    #print ('y',t2-t1,t3-t2)

    return Xargs['ret']


def evaluator(ima, var, mgrid):
    """
    Evaluate a grid of models (OBSOLETE)

    Parameters:
    -----------
    ima: numpy
        2-D input image
    var: numpy
        2-D variance image
    grid: numpy
        The 3-D grid of models with the size (.,npix,npix) where
        npix is the size of the image

    Returns:
    --------
    logl: numpy
        Array of log-likelihoods of a grid of models

    """
    mult = ((ima / var)[None, :, :] * mgrid).sum(axis=1).sum(axis=1) / (
        (mgrid**2 / var[None, :, :]).sum(axis=1).sum(axis=1)
    )
    ret = np.sum(-0.5 * np.log(2 * np.pi * var[None, :, :]) +
                 -0.5 * (ima[None, :, :] - mult[:, None, None] * mgrid)**2 / var[None, :, :], axis=1).sum(axis=1)
    return ret


def sg_prob(ima, var, psf, grid, lgal_prior, get_grid=False):
    """
    Compute the log-odds of star/galaxy given an image and galaxy model grid

    Parameters:
    -----------
    ima: numpy array
        2-D input image
    var: numpy array
        2-D variance image
    psf: numpy array
        2-D PSF image
    grid: numpy array
        3-D array of PSF convolved models with dimensions (nmodels, npix, npix)
    lgal_prior: numpy attay
        1-D array of log-priors of galaxy models
    get_grid: bool
        flag that tells whether to return the log(P(D|star))-log(P(D|gal)) scalar
        values (when get_grid is False) or to return
        log(P(D|star)) and a grid of {log(P(D|gal_i))}

    Returns:
    --------
    ret: tuple or scalar
        The log(P(D|star)-log(P(D|gal)) if get_grid=False or
        Tuple with (log(P(D|star)) and numpy array of log(P(D|gal_i)) over
        the grid of galaxy mdels
    """
    llikes_g = evaluator_c(ima, var, grid)
    llikes_s = evaluator(ima, var, np.array([psf]))
    levidence_g = scipy.special.logsumexp(lgal_prior + llikes_g)
    levidence_s = llikes_s[0]
    # 1/0
    if get_grid:
        return llikes_s, llikes_g
    return levidence_s - levidence_g


def genima(npix, psf, sn=10, pbstar=0.5, sigell=0.2, nima=1):
    """
    Generate an image of a star/galaxy at a given central SN and with a given psf
    """
    xgrid, ygrid = np.mgrid[0:npix, 0:npix]
    x0 = npix // 2
    psfw0 = 3
    aper = np.sqrt((xgrid-x0)**2+(ygrid-x0)**2) < (2*psfw0)
    naper = aper.sum()

    star = np.random.uniform() < pbstar
    if star:
        model = psf / psf.sum()
        #maxv = model.max()
        rad, n = 0, 0
    else:
        rad = np.exp(np.random.normal(1, 1))
        n = np.random.uniform(1, 4)
        N = scipy.stats.norm(0, sigell)
        ell = N.ppf(np.random.uniform()*(N.cdf(1)-N.cdf(0))+N.cdf(0))
        pa = np.random.uniform(0, 2*np.pi)
        cpa, spa = np.cos(pa), np.sin(pa)
        X1 = (xgrid-x0)*cpa + (ygrid-x0)*spa
        Y1 = -(xgrid-x0)*spa + (ygrid-x0)*cpa
        r0 = ((X1/np.sqrt(1-ell))**2 + (Y1*np.sqrt(1-ell))**2)**.5
        N = scipy.stats.norm(0, sigell)
        model = np.exp(-(r0 / rad)**(1. / n))
        model = model/model[aper].sum()
        #maxv = model.max()
        model = scipy.signal.fftconvolve(model, psf, mode='same')
    if nima == 1:
        shape = (npix, npix)
    else:
        shape = (nima, npix, npix)
    noise = np.random.uniform(0.7, 1.3, size=shape) / sn / np.sqrt(naper)
    err = noise * np.random.normal(size=noise.shape)

    if nima == 1:
        model1 = model + err
    else:
        model1 = model[None, :, :] + err

    return model1, noise**2, star, rad, n


def dohierarch(logl_star, logl_gal, verbose=False):
    """
    Perform hierarchical inference, given the log-likelihoods

    Parameters:
    -----------

    logl_star: numpy array
        1-D array of likelihood of stellar models {log(P(D_i|star))}
    logl_gal: numpy array
        2-D array of likelihoods of galactic models log(P(D_i|gal_j))
        The dimensions of the array are (n_data, n_galaxy_models)
    verbose: bool
        print the intermediate results

    Returns:
    --------
    ret: tuple
        The tuple with the fraction of stars, and array of PDFs over
        galaxy models
    """

    dat = (logl_star, logl_gal)
    args = (dat, -1)
    ngal = logl_gal.shape[1]
    p0 = np.zeros(ngal+2)
    ret = scipy.optimize.minimize(
        like, p0, method='L-BFGS-B', jac=True, args=(dat, -1))
    retx = ret['x']
    if verbose:
        print(ret)
    pstar = np.exp(retx[0]-np.logaddexp(retx[0], retx[1]))
    pgals = np.exp(retx[2:]-scipy.special.logsumexp(retx[2:]))
    return pstar, pgals


def like(p, dat, mult=-1):
    """
    Likelihood to perform hierarchical inference

    Parameters:
    -----------
    p: numpy
        Arguments of the likelihood
        p[0] -- unnormalized log(p(star))
        p[1] -- unnormalized log(p(gal)
        p[2:] -- unnormalized log(P(gal_i))
    dat: tuple
        Tuple with the grid of stellar likelihoods (1D numpy array),
        and grid galaxy likelihoods (2D numpy array) with shape (n_data, n_gal_models)
    mult: float
        Multiplier of likelihood (to switch from maximization to minimization)

    Returns:
    --------
    ret: tuple
        Tuple with likelihood value and gradient vector over parameters
    """
    penmult = 1e5
    llike_star, llike_gal = dat
    ngal = llike_gal.shape[1]
    p01 = np.logaddexp(p[0], p[1])
    lpstar = p[0] - p01
    lpgal = p[1] - p01
    penalty1 = p01
    lpgals = p[2:] - scipy.special.logsumexp(p[2:])
    penalty2 = (scipy.special.logsumexp(p[2:]))
    #print ('x')
    llike_gal1 = scipy.special.logsumexp(lpgals[None, :] + llike_gal, axis=1)
    # 1/0
    #print ('y')
    logp1 = np.logaddexp(llike_gal1+lpgal, llike_star+lpstar)
    # 1/0
    ret = mult * (logp1.sum() - penmult*(penalty1**2 + penalty2**2))
    grad = np.zeros(ngal+2)
    grad[0] = (np.exp(lpstar+lpgal) *
               (np.exp(llike_star-logp1)-np.exp(llike_gal1-logp1))).sum()
    grad[1] = - grad[0]
    xids = np.arange(ngal)
    grad[2:] = np.exp(lpgal) * (np.exp(lpgals[None, :]+llike_gal-logp1[:, None]
                                       ).sum(axis=0)-np.exp(lpgals)*np.exp(llike_gal1-logp1).sum())
    penaltygrad = np.zeros(ngal+2)
    penaltygrad[0] = 2*np.exp(lpstar) * penalty1
    penaltygrad[1] = 2*np.exp(lpgal) * penalty1
    penaltygrad[2:] = 2*np.exp(lpgals) * penalty2
    grad = (grad-penmult*penaltygrad) * mult
    print(ret, p[:4], grad[:4])
    if not np.isfinite(ret+grad).all():
        1/0
    return ret, grad


class si:
    grid = None
    grid_c = None
    psf = None


def getoddgrid(npix, sn, pbstar, sigell=0.2):
    grid1 = si.grid1
    psf = si.psf
    ima, var, is_star, rr, nn = genima(
        npix, psf, sn=sn, pbstar=pbstar, sigell=sigell)
    logodds = sg_prob(ima, var, psf, grid1,
                      np.zeros(len(grid1)), get_grid=True)
    return logodds


def fit_many_grids(nitpsf=100, nitgal=1000, pbstar=0.5, lsn1=0, lsn2=2.5, sigell=.2):
    """
    Fit many objects and return their likelihood grids
    """
    npix = 128
    nthreads = 24
    grid = gengrid()
    prior = np.ones(len(grid))
    lprior = np.log(prior * 1. / prior.sum())
    df = pandas.DataFrame()
    lstar = []
    lgal = []
    for i in range(nitpsf):
        print(i)
        psfwidth = np.random.uniform(1, 6)
        psfell = np.random.uniform(0, 0.2)
        psf = getpsf(npix, psfwidth, ell=psfell)
        grid1 = convolver(grid, psf)
        si.grid1 = grid1
        si.psf = psf
        pool = mp.Pool(nthreads)
        pstar = 0.8
        res = []
        for j in range(nitgal):
            # code on. print (j)
            sn = 10**np.random.uniform(lsn1, lsn2)
            res.append(pool.apply_async(
                getoddgrid, (npix, sn, pbstar, sigell)))
        for r in res:
            curr = r.get()
            lstar.append(curr[0])
            lgal.append(curr[1])
        pool.close()
        pool.join()
    lstar, lgal = [np.array(_) for _ in [lstar, lgal]]
    return (lstar.flatten(), lgal)
    # return df


def cutter(ima, var, x, y, npix):
    """
    Cutout the npix x npix window from input array

    Parameters:
    -----------
    ima: numpy
        2-D input image
    var: numpy
        2-D input image
    x: float
        center of an object in the input image (C-convention)
    y: float
        center of an object in the input image (C-convention)
    npix: int
        size of the return images

    Returns:
    --------
    ret: tuple
        The tuple of (npix,npix) sized image and variance arrays
    """
    xc = int(np.round(y))
    yc = int(np.round(x))
    # Notice the  flip the axes !
    retima = np.zeros((npix, npix))
    retvar = np.zeros((npix, npix))+1e8
    nx0, ny0 = ima.shape
    dx, dy = min(xc, npix//2), min(yc, npix//2)
    dx1, dy1 = min(nx0-xc, npix//2), min(ny0-yc, npix//2)
    SL = slice(xc-dx, xc+dx1), slice(yc-dy, yc+dy1)
    SL0 = slice(npix//2-dx, npix//2+dx1), slice(npix//2-dy, npix//2+dy1)
    retima[SL0] = ima[SL]
    retvar[SL0] = var[SL]
    # 1/0
    return retima, retvar


def zeropad(psf, npix):
    psfpad = np.zeros((npix, npix))
    npsf = psf.shape[0]
    pad = (npix-npsf)//2
    assert((npix-npsf) % 2 == 0)
    psfpad[pad:pad+npsf,
           pad:pad+npsf] = psf
    return psfpad


def fitextractions(grid, lprior, D, psf, get_grid=False, benchmark=False):
    """
    Fit the extracted cutouts and return either logodds or likelihood grid

    Parameters:
    -----------
    grid: numpy array
        The 3-D array of non PSF convolved models with shape (nmodels, npix, npix)
    lprior: numpy array
        The 1-D array of log-priors of galaxy models
    D: list of data
        The list of dictionaries with the data.
        Each dictionary must have keys 'cutout', 'var', 'x','y'
        corresponding to the image, variance image and center of
        the object
    psf: numpy array
        The 2-D PSF
    get_grid: bool
        flag to switch between log-odds S/G vs
        grid of likelihoods of stars and galactic models
    benchmark: bool
        turn on benchmarking, measure average time taken per source

    Returns:
    --------
    ret: tuple of arrays or numpy array
        if get_grid=True return is the tuple with 2 elts, 1st being
        the 1-D array of  log-likelihoods of stellar models,
        2nd being a 2-D array of log-likelihoods of galaxy models.
        if get_grid is False, it returns the 1-D array of log-odds
        of star or galaxy

    """
    if benchmark:
        import time

    npix = grid.shape[2]
    grid1 = convolver(grid, psf)
    ret = []
    psfpad = zeropad(psf, npix)

    if benchmark:
        start = time.time()

    for j in range(len(D)):
        curd = D[j]
        ima, var = cutter(curd['cutout'], curd['var'],
                          curd['x'], curd['y'], npix)
        logodds = sg_prob(ima, var, psfpad, grid1, lprior, get_grid=get_grid)
        ret.append(logodds)

    if benchmark:
        end = time.time()
        print("%.2fms per source" % (((end - start)/len(D))*1000))

    if get_grid:
        logstar = [_[0] for _ in ret]
        loggal = [_[1] for _ in ret]
        return logstar, loggal
    else:
        ret = np.array(ret)
        return ret


def domany(nitpsf=10):
    """
    Generate images and classify them
    """
    # nitpsf =
    nitgal = 100
    npix = 128
    pbstar = 0.5
    grid = gengrid()
    prior = np.ones(len(grid))
    lprior = np.log(prior * 1. / prior.sum())
    df = pandas.DataFrame()
    for i in range(nitpsf):
        print(i)
        psfwidth = np.random.uniform(1, 6)
        psfell = np.random.uniform(0, 0.2)
        psf = getpsf(npix, psfwidth, ell=psfell)
        grid1 = convolver(grid, psf)
        for j in range(nitgal):
            # code on. print (j)
            sn = 10**np.random.uniform(0, 2.5)
            t1 = time.time()
            ima, var, is_star, rr, nn = genima(npix, psf, sn=sn, pbstar=pbstar)
            t2 = time.time()
            logodds = float(sg_prob(ima, var, psf, grid1, lprior))
            t3 = time.time()
            #print (t2-t1,t3-t2)
            df = df.append({'logodd': logodds, 'sn': sn,
                            'psfwidth': psfwidth, 'star': is_star,
                            'rad': rr, 'nser': nn}, True)
    return df
# mods=gengrid()
# def genima(
#err=np.random.normal(size=  2)
