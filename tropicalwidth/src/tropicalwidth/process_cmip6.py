#  process_ccmp.py executable
#  Version: 1.1
#  Date: March 18, 2025
#  Author: Stephen Leroy (sleroy@aer.com)
#
#  Required standard packages: os, re, sys, time, argparse, requests, datetime
#  Required non-standard packages: netCDF4, numpy. 
#  Required other: libtropicalwidth

#  Imports. 

import os
import re
import sys
import shutil
from time import time
import requests
from datetime import datetime, timedelta
import argparse 
import numpy as np
from netCDF4 import Dataset
from global_land_mask import globe
from tqdm import tqdm 
import intake_esgf
from .libtropicalwidth import regions


################################################################################
#  Error handling.
################################################################################

class Error(Exception):
    pass

class CMIP6TropicalWidthProcessError(Error): 
    def __init__(self,expression,message=''):
        self.expression = expression
        self.message = message



################################################################################
#  Calculate tropical width. 
################################################################################

def CMIP6tropicalWidth( mcat, fill_value=-999e7 ): 

    ret = { 'status': "success", 'messages': [], 'comments': [] }

    #  Initialize latitude save structure. 

    month_multiplier = 1/12.0
    out = [ { 'region': region['name'], 'lats': [] } for region in regions ]
    times = []

    #  Longitude-latitude grid and ocean mask. 

    m = mcat['uas'].uas.loc[ "1985-01-01":"2024-12-31" ]
    lats = np.array( m.lat )
    lons = np.array( m.lon )
    lons[ lons >= 180.0 ] -= 360.0

    #  Create ocean mask. 

    oceanmask = np.zeros( (lats.size,lons.size), dtype='b' )
    for ilat, lat in enumerate(lats): 
        for ilon, lon in enumerate(lons): 
            oceanmask[ilat,ilon] = globe.is_ocean( lat, lon )


    #  Define region masks. 

    lats2 = lats[ np.indices( ( lats.size, lons.size ) )[0,:,:] ]
    lons2 = lons[ np.indices( ( lats.size, lons.size ) )[1,:,:] ]
    regionmasks = np.zeros( (len(regions),lats.size,lons.size), dtype='b' )

    for iregion, region in enumerate( regions ): 

        #  Mask by latitude (always).

        regionmask = np.logical_and( region['latbounds'][0] < lats2, lats2 < region['latbounds'][1] )

        #  Mask by longitude. 

        if 'lonbounds' in region.keys(): 
            dlons2 = lons2 - region['lonbounds'][0] 
            dlons2[ dlons2 < 0.0 ] += 360.0
            dlons2[ dlons2 >= 360.0 ] -= 360.0

            dlon = np.rad2deg( region['lonbounds'][1] - region['lonbounds'][0] ) 
            if dlon >= 360.0: 
                dlon -= 360.0
            elif dlon < 0.0: 
                dlon += 360.0

            regionmask = np.logical_and( regionmask, dlons2 <= dlon )

        #  Ocean only? 

        if region['oceanmask']: 
            regionmask = np.logical_and( regionmask, oceanmask )

        regionmasks[iregion,:,:] = regionmask

    #  Define times in the file. 

    rectimes = ( np.array( m.time ).astype("datetime64[M]").astype("int") + 0.5 ) / 12 + 1970

    #  Loop over times in the stream. 

    # iterator = tqdm( enumerate(rectimes), desc="    Time" )
    iterator = enumerate(rectimes)

    for irectime, rectime in iterator: 

        u = np.array( m[irectime,:,:] )

        for iregion, region in enumerate(regions): 

            za = np.ma.masked_where( np.logical_not(regionmasks[iregion,:,:]), u.data ).mean( axis=1 )

            #  Find where zonal average is negative or 0. 

            za_indices = np.argwhere( za[:-1] * za[1:] <= 0 )

            if za_indices.size == 0:
                lat = fill_value
            else:
                # Linear interpolation to estimate null in longitude-average zonal wind. 
                idx = za_indices[0,0]
                slope = za[idx] / ( za[idx] - za[idx+1] )
                lat = lats[idx]*(1-slope) + lats[idx+1]*(slope)

            out[iregion]['lats'].append( lat )

    #  Convert lats to masked array. 

    if ret['status'] == "success": 
        times = rectimes
        for outrec in out: 
            x = np.array( outrec['lats'] )
            outrec['lats'] = np.ma.masked_where( x==fill_value, x )

        ret.update( { 'times': times, 'latdata': out } )

    return ret


################################################################################
#  Execution. 
################################################################################

def process_cmip6( outputfile:str, clobber=False, fill_value=-999 ): 
    """Process CMIP6 surface air winds data for tropical width."""

    if os.path.isfile( outputfile ): 
        if clobber: 
            print( f'{outputfile} exists; clobbering' )
            sys.stdout.flush()
            run = True
        else: 
            print( f'{outputfile} exists; exiting' )
            sys.stdout.flush()
            run = False
    else: 
        run = True

    if not run: return

    #  Get catalogue of AMIP uas files, list of models. 

    scenario = { 'experiment_id': "amip", 'table_id': "Amon", 'variable_id': ["uas"], 'member_id': "r1i1p1f1" }
    cat = intake_esgf.ESGFCatalog()
    q = cat.search( **scenario )
    models = sorted( list( q.df.source_id ) )

    #  Create output file. 

    tmpfile = "tmp.nc"
    print( "Generating " + outputfile )
    sys.stdout.flush()

    d = Dataset( tmpfile, 'w', format='NETCDF4' )

    #  Get a listing of all uas (surface air zonal wind) files. Parse the file names. 

    first_entry = True

    for imodel, model in enumerate( models ): 

        print( f'Computations for model "{model}"' )
        sys.stdout.flush()

        mm = cat.search( **scenario, source_id=model ).to_dataset_dict( prefer_streaming=True, add_measures=False )
        ret = CMIP6tropicalWidth( mm, fill_value=fill_value )

        if ret['status'] != "success": 
            print( "; ".join( ret['comments'] ) )
            sys.stdout.flush()
            continue

        yeartimes, widths = ret['times'], ret['latdata']

        if first_entry: 

            # Define fields in NetCDF dataset. 

            d.createDimension( 'region', len(regions) )
            d.createDimension( 'str', max( [ len(region['name']) for region in regions ] ) )
            d_regions = d.createVariable( 'regions', 'c', dimensions=('region','str') )
            d_regions.setncatts( { 'description': "Name of region over which u=0 is computed" } )

            #  Region names. 

            for i, r in enumerate(regions):
                d_regions[i,:len(r['name'])] = r['name']

            first_entry = False

        #  Data for each key goes in a group. 

        g = d.createGroup( model )

        #  Group attributes identifying CMIP6 run. 

        g.setncatts( { 
                'model': model, 
                'frequency': scenario['table_id'], 
                'scenario': scenario['experiment_id'], 
                'realization': scenario['member_id']
            } )

        #  Group time dimension. 

        g.createDimension( 'time', None )

        g_years = g.createVariable( 'years', 'i4', dimensions=('time',) )
        g_years.setncatts( { 'description': 'Year of time interval',
                             'units': 'year', '_FillValue':np.int32(fill_value) } )

        g_months = g.createVariable( 'months', 'i4', dimensions=('time',) )
        g_months.setncatts( { 'description': 'Month of time interval', 'units':'month',
                              'range':np.array([1,12],dtype='i'), '_FillValue':np.int32(fill_value) } )

        g_lats = g.createVariable( 'lats', 'f4', dimensions=('time','region') )
        g_lats.setncatts( {'description': 'Latitude of the uwind=0 line',
                            'units': 'degrees north', '_FillValue':np.float32(fill_value)} )

        #  Write times to output. 

        g_years[:] = np.int32( yeartimes )
        g_months[:] = np.int32( yeartimes * 12 ) - np.int32( yeartimes ) * 12 + 1

        #  Write latitude nulls to output. 

        for iregion, r in enumerate(widths): 
            g_lats[:,iregion] = r['lats']

    d.close()

    print( f'Copying {tmpfile} to {outputfile}' )
    sys.stdout.flush()

    outputfile_dir = os.path.dirname( outputfile )
    if outputfile_dir != "": 
        os.makedirs( outputfile_dir, exist_ok=True )
    shutil.copy( tmpfile, outputfile )
    os.remove(tmpfile)


def main(): 

    parser = argparse.ArgumentParser( prog="""Process tropical width data from CMIP6""" )

    parser.add_argument( "output", type=str, help='The name of the NetCDF output file' )

    parser.add_argument( "--clobber", "-c", dest='clobber', default=False, action="store_true", 
            help='Clobber previously existing output file; do not clobber by default' )

    args = parser.parse_args()

    t0 = time()

    process_cmip6( args.output, clobber=args.clobber, fill_value=-999 )

    t1 = time()
    dt = t1 - t0
    print( f'total elapsed time = {int(dt/60):4d}m {int(dt)%60:2d}s' )
    sys.stdout.flush()


if __name__ == "__main__": 
    main()

