#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.modeling.functional_models import Sersic2D
from matplotlib.colors import LogNorm
import scipy.signal


def gen_psf(psfsigma_arcsec, pixel_scale, ell=0, pa=0):
    psfsig = psfsigma_arcsec / pixel_scale
    thresh = 5  # this many sigma away
    npix = int(2*thresh*psfsig)
    if npix % 2 == 0:
        npix += 1
    xgrid, ygrid = np.mgrid[0:npix, 0:npix]
    x0 = (npix-1)/2.
    cpa = np.cos(np.deg2rad(pa))
    spa = np.sin(np.deg2rad(pa))
    x1, y1 = (xgrid-x0)*cpa+(ygrid-x0)*spa, (ygrid-x0)*cpa-(xgrid-x0)*spa
    R2 = (x1*np.sqrt(1-ell))**2 + (y1/np.sqrt(1-ell))**2
    psf = np.exp(-0.5*R2/psfsig**2)
    psf = psf/psf.sum()
    return psf


def gen_field(output_array_edge=4096,  # pixels
              pixel_scale=0.3,  # arcsec/pixel
              sample_ratio=2.0,  # simulated sky resolution to detector resolution ratio
              star_fraction=0.6,  # fraction of sources that are stars
              min_sersic = 1, # min sersic index
              max_sersic = 4, # max sersic index
              mean_r = 0.4, # mean of galaxy size distribution
              std_r = 0.4, # standard deviation of galaxy size distribution
              sources_per_axis=40,  # number of sources per axis
              seeing=1,  # arcseconds
              background_level=100,  # counts
              plot_image=False,
              save_to_fits=True,
              saturation=False,
              image_fname='image.fits',
              source_list_fname='source_list.fits',
              psf_fname='psf.fits',
              rdnoise=2,
              variance_fname='variance.fits',
              seed=None
              ):
    S0 = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    # total number of sources in frame

    source_count = sources_per_axis**2
    Rap = 1.5  # arcsec aperture is the one withing which we measure the flux
    # for extended sources

    # create sky array with a background level
    sky_array_edge = int(output_array_edge * sample_ratio)
    sky_array_shape = (sky_array_edge, sky_array_edge)
    sky_array = np.ones(sky_array_shape, dtype='f4') + background_level

    # initialise source list
    source_list = np.empty(source_count, dtype=[('id', 'i4'),
                                                ('X', 'f4'),
                                                ('Y', 'f4'),
                                                ('I', 'i4'),
                                                ('n', 'f4'),
                                                ('size', 'f4')])
    source_list[:] = (0, np.nan, np.nan, 0, np.nan, np.nan)

    # add ids
    source_list['id'] = np.arange(source_count, dtype='i4')

    # add sky positions
    pad = 100
    XYs = np.linspace(pad, output_array_edge-pad,  int(np.sqrt(source_count)))
    Xs, Ys = np.meshgrid(XYs, XYs)
    source_list['X'] = (Xs.flatten() * sample_ratio).astype(np.int)
    source_list['Y'] = (Ys.flatten() * sample_ratio).astype(np.int)

    # add source intensities
    minlflux = 3.25
    maxlflux = 5.25
    source_list['I'] = 10**np.random.uniform(
        minlflux, maxlflux, size=source_count)

    # select some to be galaxies
    galaxies = np.random.uniform(0, 1, size=source_count) > star_fraction
    stars = ~galaxies

    # add galaxy sersic indices
    galaxy_count = galaxies.sum()
    source_list['n'][galaxies] = np.random.uniform(
        min_sersic, max_sersic, size=galaxy_count)

    # add galaxy sizes
    source_list['size'][galaxies] = 10**np.random.normal(
        mean_r, std_r, size=galaxy_count)

    ### add stars to sky ###
    Xs = (source_list['X'][stars]).astype(np.int)
    Ys = (source_list['Y'][stars]).astype(np.int)
    sky_array[Ys, Xs] += source_list['I'][stars]

    ### add galaxies to sky ###
    # this is fairly ugly and slow, but astropy makes it easy to write slow codex
    X, Y = np.meshgrid(np.arange(sky_array_edge), np.arange(sky_array_edge))
    for n, source in enumerate(source_list[galaxies]):
        curi = source['I']
        curs = source['size']  # in arcsec
        Re = curs * 1/pixel_scale * sample_ratio  # in pix
        n = source['n']
        bn = 2 * n - 1./3 + 4./405/n+46./25515/n**2 + 131./1148175/n**3
        # ciotti bertin

        #Ie = curi / (2 * np.pi * n * Re**2 * np.exp(bn))*bn**(2*n) / \
        #     scipy.special.gammainc(2*n, bn*(Rap/curs)**(1./n))
        # http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
        # Not used because I normalize myself to a given aperture

        mod = Sersic2D(amplitude=1, r_eff=Re, n=n,
                       x_0=source['X'], y_0=source['Y'])  # ,
        # ellip=np.random.rand(1)**2, theta=np.random.rand(1)*np.pi)
        # turned off ellipticity

        # this approximates the radius at which the total lum is 99%
        # compared to total  intensity
        frac = 1e-2
        Rmax = Re * (scipy.special.gammaincinv(2*n, (1-frac))/bn)**n
        Rmax = max(Rmax, 10)
        # then only bother to evaluate the model inside that radius to save tim

        tomod = []
        minx = max(int(np.floor(source['X']-Rmax)), 0)
        miny = max(int(np.floor(source['Y']-Rmax)), 0)
        maxx = min(int(np.ceil(source['X']+Rmax)), X.shape[0])
        maxy = min(int(np.ceil(source['Y']+Rmax)), X.shape[1])
        tomod = (slice(miny, maxy), slice(minx, maxx))
        x = X[tomod]
        y = Y[tomod]
        subs = ((x-source['X'])**2+(y-source['Y']) **
                2) < (Rap/pixel_scale*sample_ratio)**2
        modxy = mod(x, y)
        modxy = curi * modxy / (modxy[subs]).sum()
        sky_array[tomod] += modxy

    # scale source list coordinates
    source_list['X'] /= sample_ratio
    source_list['Y'] /= sample_ratio

    # psf convolve
    #sky_unit_seeing = seeing * 1/pixel_scale * sample_ratio
    psf = gen_psf(seeing, pixel_scale/sample_ratio)

    sky_array = scipy.signal.fftconvolve(sky_array, psf, mode='same')

    #sky_array = gaussian_filter(sky_array, sigma=sky_unit_seeing)
    psf_0 = gen_psf(seeing, pixel_scale)  # not oversampled PSF

    # bin to detector resolution
    resample = [int(sky_array_edge // sample_ratio), int(sample_ratio)]
    detector_array = sky_array.reshape(*(resample*2))
    detector_array = detector_array.mean(axis=3).mean(axis=1)

    # add poisson noise
    variance = (detector_array + rdnoise**2)

    detector_array = np.random.poisson(
        lam=detector_array) + np.random.normal(0, rdnoise, size=detector_array.shape)

    if saturation:
        # saturation
        saturated = detector_array > 65535
        detector_array[saturated] = 0.0

    if save_to_fits:
        fits.writeto(source_list_fname, source_list, overwrite=True)
        fits.writeto(psf_fname, psf_0, overwrite=True)
        fits.writeto(variance_fname, variance, overwrite=True)
        fits.writeto(image_fname,
                     detector_array.astype(np.float64),
                     overwrite=True)

    if plot_image:
        # plot the field
        plt.figure()
        minmax = np.percentile(detector_array, [5, 95])
        # , norm=LogNorm())
        plt.imshow(detector_array, vmin=minmax[0], vmax=minmax[1])
        plt.colorbar()
        xstar = source_list['X'][stars]
        ystar = source_list['Y'][stars]
        xgal = source_list['X'][galaxies]
        ygal = source_list['Y'][galaxies]
        plt.scatter(xgal, ygal, marker='+', c='red', label='G')
        plt.scatter(xstar, ystar, facecolors='none', edgecolors='r', label='S')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2)
        plt.show()
    np.random.set_state(S0)

    return source_list, detector_array


if __name__ == "__main__":
    gen_field(plot_image=False, save_to_fits=True, seed=0)
