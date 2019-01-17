#!/usr/bin/env python
from __future__ import print_function
import matplotlib as mpl
if __name__ == '__main__':
    mpl.use('Agg')
import sewpy as sew
from astropy.io import fits
import astropy.table as atpy
from os import getcwd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import tempfile

# sextractor output columns
sexparams = ['NUMBER',
             'X_IMAGE',
             'Y_IMAGE',
             'XWIN_IMAGE',
             'YWIN_IMAGE',
             'MAG_AUTO',
             'FLUX_AUTO',
             'FLUX_APER(1)']


def sextractor_setup(var_fname, segm_fname, bg_fname, aper=5, workdir=None):
# sextractor config parameters
    sexconfig = {'CHECKIMAGE_TYPE': 'SEGMENTATION,BACKGROUND',
                 'CHECKIMAGE_NAME': ','.join([segm_fname,bg_fname]),
                 'CATALOG_TYPE': 'FITS_1.0',
                 'WEIGHT_TYPE': 'MAP_VAR',
                 'WEIGHT_IMAGE': var_fname,
                 'PHOT_APERTURES': '%f'%(aper)
                 #'CATALOG_NAME': cat_fname
}
    print (sexconfig)
    # set up sextractor
    sexpath = os.environ.get('SEX_PATH') or '/usr/bin/sextractor'
    sex = sew.SEW(workdir=workdir or getcwd(),
                  sexpath=sexpath,
                  params=sexparams,
                  config=sexconfig)
    sex._clean_workdir()
    return sex


def get_segment(image, var, bg, segmentation, source):
    segment = segmentation == source['NUMBER']
    x = np.sum(segment, axis=0)
    y = np.sum(segment, axis=1)
    nzx = np.nonzero(x)[0]
    nzy = np.nonzero(y)[0]
    x1, x2 = nzx[0], nzx[-1]
    y1, y2 = nzy[0], nzy[-1]
    region = (slice(y1, y2+1), slice(x1, x2+1))
    cutout = image[region]-bg[region]
    # IMPORTANT I background subtract the image
    varcut = var[region]
    y_pos = source['YWIN_IMAGE'] - y1 -1 
    x_pos = source['XWIN_IMAGE'] - x1 -1      
    # we subtract 1, because Sex's positions are 1-based

    varcut[~segment[region]] = np.median(varcut)*100
    # set variance to high value in pixels not marked by segmentation map
    return {'cutout':cutout,'var':varcut,'x': x_pos, 'y': y_pos, 'FLUX_AUTO':source['FLUX_AUTO'], 'FLUX_APER':source['FLUX_APER']}


def get_cutouts(sexcat, segmentation, image, var, bg, sourcelist, figs):
    # grab cutout for each source
    sources = []
    for source in sexcat:
        #print("source: {}".format(source['NUMBER']))

        if sourcelist is not None:
            # match to original source_list
            sep = np.hypot(source['XWIN_IMAGE']-sourcelist['X'],
                           source['YWIN_IMAGE']-sourcelist['Y'])
            if np.min(sep) > 5:
                warnings.warn(
                    "This source doesn't appear to correspond to real source")
                continue

            thissource = sourcelist[np.argmin(sep)]

        # get the cutout
        segm_info = get_segment(image, var, bg, segmentation, source)

        # send to dict
        if sourcelist is not None:
            segm_info['star0'] = np.isnan(thissource['n'])  # is star? 
            segm_info['flux'] = thissource['I']
            segm_info['id'] =thissource['id']
        sources.append(segm_info)
            
        # save the cutout as png files if requested
        if figs:
            plt.figure()
            plt.subplot(121)
            plt.imshow(segmentation == source['NUMBER'])
            plt.gca().invert_yaxis()
            plt.subplot(122) 
            minmax = np.percentile(cutout, [5, 95])
            plt.imshow(cutout, vmin=minmax[0], vmax=minmax[1])
            plt.scatter(XY[0], XY[1], marker='+')
            plt.gca().invert_yaxis()
            plt.savefig("source_{}.png".format(source['NUMBER']), dpi=200)
            plt.close()

    return sources


def run(imagepath, var_fname, sourcelistpath=None, figs=False):
    dir = tempfile.TemporaryDirectory(dir='./')

    bg_fname='%s/bg_temp_%d.fits'%(dir.name,os.getpid())
    segm_fname= '%s/segm_temp_%d.fits'%(dir.name,os.getpid())
    #cat_fname= './cat_temp_%d.fits'%(os.getpid())
    sexObj = sextractor_setup(var_fname, segm_fname, bg_fname, workdir=dir.name)
    # run sextractor
    R=sexObj(imagepath, returncat=False)
    cat_fname = R['catfilepath']

    # load the original image
    image = fits.getdata(imagepath, -1, view=np.array)
    bg = fits.getdata(bg_fname)
    var = fits.getdata(var_fname)
    if sourcelistpath is not None:
        # load the original source list
        sourcelist = fits.getdata(sourcelistpath, -1, view=np.recarray)
    else:
        sourcelist = None

    # load the generated segmentation map
    segmentation = fits.getdata(segm_fname, -1, view=np.array)

    # load the generated source catalogue

    sexcat = atpy.Table().read(cat_fname,format='fits')
    # get the cutouts
    sources = get_cutouts(sexcat, segmentation, image, var, bg, sourcelist, figs)

    # sources is a dict, the keys are the original source ids (i.e. the source
    # ids from the source list fits file produced by gen_field.py). The values
    # are tuples containing [0] a list of X and Y position inside the cutout,
    # [1] a numpy array of the cutout itself, and if a source list is given:
    # [2] is whether the object is a star.
    os.unlink(bg_fname)
    os.unlink(segm_fname)
    os.unlink(cat_fname)
    return sources


if __name__ == "__main__":
    imagepath = 'image.fits'
    var_fname = 'variance.fits'
    sourcelistpath = 'source_list.fits'

    run(imagepath, var_fname, figs=False)  # True)
