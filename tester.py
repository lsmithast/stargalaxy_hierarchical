#!/usr/bin/env python3

import numpy as np
import gen_cutouts
import fitter
import astropy.io.fits as pyfits
import gen_field
import multiprocessing as mp
import pandas
import os


def doone(xid, get_grid=True):
    """ Run one full simulation:
    Generate an image, extract sources, fit them
    Arguments:
    seed -- integer
    """
    
    # generate file names
    im = 'im_%d.fits' % xid
    var = 'var_%d.fits' % xid
    slist = 'slist_%d.fits' % xid
    psf_name = 'psf_%d.fits' % xid
    
    # generate some random background and seeing values
    S0 = np.random.get_state()
    np.random.seed(xid)
    bg = 10**np.random.uniform(1.5, 2.5)
    see = np.random.uniform(0.6, 1.7)
    np.random.set_state(S0)
    
    # set pixel scale
    scale = 0.3
    
    # generate test image
    print ("Generating test image, source list, psf...")
    gen_field.gen_field(plot_image=False, save_to_fits=True, seed=xid,
                        variance_fname=var, source_list_fname=slist,
                        image_fname=im, psf_fname=psf_name,
                        background_level=bg, seeing=see, pixel_scale=scale)
    print("    done")

    # load the psf
    psf = pyfits.getdata(psf_name)
    
    # generate image cutouts
    print("Generating cutouts...")
    R = gen_cutouts.run(im, var, slist)
    print("    done")
    
    # generate model grid
    print("Generating model grid...")
    grid = fitter.gengrid(pixel_scale=scale)
    print("    done")
    
    # fit model grid to cutouts
    print("Fitting model grid to cutouts...")
    lprior = np.zeros(len(grid))-np.log(len(grid))
    XR = fitter.fitextractions(grid, lprior, R, psf, get_grid=get_grid)
    print("    done")
    
    # generate output
    fluxes0 = np.array([_['flux'] for _ in R])
    star0 = np.array([_['star0'] for _ in R])
    fap = np.array([_['FLUX_APER'] for _ in R])
    nobj = len(R)
    xdict = {'flux0': fluxes0, 'star0':star0, 'FLUX_APER':fap,'bg':bg+np.zeros(nobj), 'seeing':see +np.zeros(nobj)}
    
    # remove temporary files
    os.unlink(var)
    os.unlink(im)
    os.unlink(slist)
    os.unlink(psf_name)

    # return output
    if get_grid:
        xdict['logl_s']=np.array(XR[0]).flatten()
        xdict['logl_g']=np.array(XR[1]).tolist()
    else:
        xdict['logodds'] = XR
    df = pandas.DataFrame(xdict)
    return df


def doall(N, get_grid=True):
    pool = mp.Pool(24)
    R = []
    for i in range(N):
        R.append(pool.apply_async(doone, (i,get_grid)))
    R1 = []
    for r in R:
        R1.append(r.get())
    pool.close()
    pool.join()
    return R1


if __name__=="__main__":
    doone(1, get_grid=True)
