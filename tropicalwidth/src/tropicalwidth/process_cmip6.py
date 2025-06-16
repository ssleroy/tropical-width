#  process.py executable
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
from .libtropicalwidth import default_dataroot, regions


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

def CMIP6tropicalWidth( recs, fill_value=-999e7 ): 

    ret = { 'status': "success", 'messages': [], 'comments': [] }

    #  Initialize latitude save structure. 

    month_multiplier = 1/12.0
    out = [ { 'region': region['name'], 'lats': [] } for region in regions ]
    times = []

    for irec, rec in enumerate( recs ): 

        try: 
            print( f'  Opening {rec["path"]}' )
            sys.stdout.flush()
            d = Dataset( rec['path'], 'r' )
        except: 
            ret['status'] = "fail"
            ret['messages'].append( "UnreadableFile" )
            ret['comments'].append( f"Could not open file {rec['path']}" )
            return ret 

        #  Check input file. 

        varnames = list( d.variables.keys() )
        for v in [ 'lon', 'lat', 'uas' ]: 
            if v not in varnames: 
                ret['status'] = "fail"
                ret['messages'].append( 'InvalidFile' )
                ret['comments'].append( f'Could not find variable "{v}" in file {rec["path"]}' )
                d.close()
                return ret

        #  Longitude-latitude grid and ocean mask. 

        if irec == 0: 

            # print( '    Creating region masks' )
            # sys.stdout.flush()

            lats = d.variables['lat'][:]
            x = np.deg2rad( d.variables['lon'][:] )
            lons = np.rad2deg( np.arctan2( np.sin(x), np.cos(x) ) )

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
                    dlons2 = np.deg2rad( lons2 - region['lonbounds'][0] ) 
                    dlons2 = np.arctan2( -np.sin(dlons2), -np.cos(dlons2) ) + np.pi
                    dlon = np.rad2deg( region['lonbounds'][1] - region['lonbounds'][0] ) 
                    dlon = np.arctan2( -np.sin(dlon), -np.cos(dlon) ) + np.pi
                    regionmask = np.logical_and( regionmask, dlons2 <= dlon )

                #  Ocean only? 

                if region['oceanmask']: 
                    regionmask = np.logical_and( regionmask, oceanmask )

                regionmasks[iregion,:,:] = regionmask

        #  Define times in the file. 

        year_min, mo_min = int(rec['monthrange'][:4]), int(rec['monthrange'][4:6])
        year_max, mo_max = int(rec['monthrange'][7:11]), int(rec['monthrange'][11:])
        rectimes = np.arange(year_min + (mo_min-0.5) * month_multiplier, 
                                year_max + mo_max * month_multiplier,
                                month_multiplier)

        #  Loop over times in the file. 

        # iterator = tqdm( enumerate(rectimes), desc="    Time" )
        iterator = enumerate(rectimes)

        for irectime, rectime in iterator: 

            if rectime in times: 
                continue

            times.append( rectime )
            u = d.variables['uas'][irectime,:,:]

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
                    # TODO: I'm not clear how this finds the null. 
                    lat = lats[idx]*(1-slope) + lats[idx+1]*(slope)

                out[iregion]['lats'].append( lat )

        #  Next record/file. 

        d.close()

    #  Convert times and lats to ndarray and masked array. 

    if ret['status'] == "success": 
        times = np.array( times )
        for outrec in out: 
            x = np.array( outrec['lats'] )
            outrec['lats'] = np.ma.masked_where( x==fill_value, x )

        ret.update( { 'times': times, 'latdata': out } )

    return ret


################################################################################
#  Execution. 
################################################################################

def process_cmip6( outputfile:str, dataroot:str=default_dataroot, clobber=False, fill_value=-999 ): 
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

    print( "Generating " + outputfile )
    sys.stdout.flush()

    outputfile_dir = os.path.dirname( outputfile )
    if outputfile_dir != "": 
        os.makedirs( outputfile_dir, exist_ok=True )

    d = Dataset( outputfile, 'w', format='NETCDF4' )

    #  Get a listing of all uas (surface air zonal wind) files. Parse the file names. 

    allrecs = []

    for root, subdirs, files in os.walk( os.path.join( dataroot, "cmip6" ) ): 
        subdirs.sort()
        files.sort()

        for file in files: 
            # Check that file has correct format (is a uas file).
            m = re.search( r'^uas_(\w+)_(\S+)_(\w+)_(\w+)_(\w+)_(\d{6}-\d{6})\.nc$', file )
            if m: 
                rec = { 
                       'path': "/".join([root, file ]), 
                       'key': "_".join( [ m.group(1), m.group(2), m.group(3), m.group(4) ] ), 
                       'frequency': m.group(1), 
                       'model': m.group(2), 
                       'scenario': m.group(3), 
                       'realization': m.group(4), 
                       'grid': m.group(5), 
                       'monthrange': m.group(6)
                    }
                allrecs.append( rec )

    keys = sorted( list( { rec['key'] for rec in allrecs } ) )
    first_entry = True

    for ikey, key in enumerate(keys): 

        print( f'Computations for analysis key "{key}"' )
        sys.stdout.flush()

        recs = [ rec for rec in allrecs if rec['key']==key ]
        ret = CMIP6tropicalWidth( recs, fill_value=fill_value )

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

        g = d.createGroup( key )

        #  Group attributes identifying CMIP6 run. 

        g.setncatts( { 
                'model': recs[0]['model'], 
                'frequency': recs[0]['frequency'], 
                'scenario': recs[0]['scenario'], 
                'realization': recs[0]['realization']
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


def main(): 

    parser = argparse.ArgumentParser( prog="""Process tropical width data from CCMP""" )

    parser.add_argument( "output", type=str, help='The name of the NetCDF output file' )

    parser.add_argument( "--dataroot", default=default_dataroot, 
            help=f'The root path for the data; the default is "{default_dataroot}".' )

    parser.add_argument( "--clobber", "-c", dest='clobber', default=False, action="store_true", 
            help='Clobber previously existing output file; do not clobber by default' )

    args = parser.parse_args()

    t0 = time()

    process_cmip6( args.output, dataroot=args.dataroot, clobber=args.clobber, fill_value=-999 )

    t1 = time()
    dt = t1 - t0
    print( f'total elapsed time = {int(dt/60):4d}m {int(dt)%60:2d}s' )
    sys.stdout.flush()


if __name__ == "__main__": 
    main()

