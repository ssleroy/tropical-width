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
from .libtropicalwidth import Model, FilePtr, TimePtr, default_dataroot, \
        tropicalWidth, averageWind


################################################################################
#  Error handling.
################################################################################

class Error(Exception):
    pass

class TropicalWidthProcessError(Error): 
    def __init__(self,expression,message=''):
        self.expression = expression
        self.message = message


################################################################################
#  Execution. 
################################################################################

def process( timerange:{tuple,list}, outputfile:str, dataroot:str="/fg", clobber=False ): 
    """Process CCMP data, looking for the u=0 latitude in both hemispheres under four
    scenarios: all CCMP, ocean-only CCMP, data-constrained CCMP, ocean-only and data-
    constrained CCMP. The output is written to a NetCDF file."""

    if len( timerange ) != 2: 
        raise TropicalWidthProcessError( "InvalidArgument", "timerange must be a 2-tuple/list of datatime" )

    if not isinstance(timerange[0],datetime) or not isinstance(timerange[1],datetime): 
        raise TropicalWidthProcessError( "InvalidArgument", "timerange must be a 2-tuple/list of datatime" )

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

    tmpfile = "tmp.nc"
    print( "Generating " + outputfile )
    sys.stdout.flush()

    d = Dataset( tmpfile, 'w', format='NETCDF4_CLASSIC' )

    methods = { 'All':{'description':'All wind data in CCMP used'}, \
      'Dataonly':{'description':'Only the wind data constrained by at least one data point are used'} }

    methodNames = list( methods.keys() )
    nmethods = len( methodNames )

    first = True

    for imethod, methodName in enumerate(methodNames): 

        method = methods[methodName]
        print( f'Computations for analysis method "{methodName}"' )
        sys.stdout.flush()
        if re.search( 'Dataonly', methodName ): 
            m = Model( 'daily', dataroot=dataroot )
            n = 1
        else: 
            m = Model( 'monthly', dataroot=dataroot )
            n = 0

        months, regions = tropicalWidth( timerange, model=m, n=n )
        nregions = len( regions )

        if first: 
            d.createDimension( 'time', None )
            d.createDimension( 'method', nmethods )
            d.createDimension( 'region', nregions )
            d.createDimension( 'str1', max( [ len(name) for name in methodNames ] ) )
            d.createDimension( 'str2', max( [ len(region['name']) for region in regions ] ) )

            d_years = d.createVariable( 'years', 'i', dimensions=('time',) )
            d_years.setncatts( { 'description':'Year of time interval', 'units':'year' } )

            d_months = d.createVariable( 'months', 'i', dimensions=('time',) )
            d_months.setncatts( { 'description':'Month of time interval', 'units':'month', 'range':np.array([1,12],dtype='i') } )

            d_methods = d.createVariable( 'methods', 'c', dimensions=('method','str1') )
            d_methods.setncatts( { 'description':"Method of analysis scheme used to compute edge of Hadley cell" } )

            d_regions = d.createVariable( 'regions', 'c', dimensions=('region','str2') )
            d_regions.setncatts( { 'description':"Name of region over which u=0 is computed" } )

            d_lats = d.createVariable( 'lats', 'f', dimensions=('time','region','method') )
            d_lats.setncatts( {'description':'Latitude of the uwind=0 line', 'units':'degrees north'} )

            for itime, month in enumerate(months): 
                d_years[itime] = month.year
                d_months[itime] = month.month

            for i in range(nmethods): 
                d_methods[i,:len(methodNames[i])] = methodNames[i]

            for i, r in enumerate(regions): 
                d_regions[i,:len(r['name'])] = r['name']

            first = False

        for iregion, r in enumerate(regions): 
            d_lats[:,iregion,imethod] = r['lats']

        m.close()

    d.close()

    print( f'Copying {tmpfile} to {outputfile}' )
    sys.stdout.flush()

    outputfile_dir = os.path.dirname( outputfile )
    if outputfile_dir != "": 
        os.makedirs( outputfile_dir, exist_ok=True )
    shutil.copy( tmpfile, outputfile )


def main(): 

    parser = argparse.ArgumentParser( prog="""Process tropical width data from CCMP""" )

    parser.add_argument( "monthrange", type=str, help="""The range of months (inclusive) over which to process 
            CCMP tropical widths, format "YYYY-MM:YYYY-MM" for the begin and end months""" )

    parser.add_argument( "output", type=str, help='The name of the NetCDF output file' )

    parser.add_argument( "--dataroot", default=default_dataroot, 
            help=f'The root path for the data; the default is "{default_dataroot}".' )

    parser.add_argument( "--clobber", "-c", dest='clobber', default=False, action="store_true", 
            help='Clobber previously existing output file; do not clobber by default' )

    args = parser.parse_args()

    m = re.search( "^(\d{4}-\d{2}):(\d{4}-\d{2})$", args.monthrange )
    if m: 
        t = datetime.fromisoformat( m.group(2)+"-01" ) + timedelta(days=31)
        timerange = [ datetime.fromisoformat( m.group(1)+"-01" ), 
                     datetime(year=t.year,month=t.month,day=1) - timedelta(days=1) ]
    else: 
        print( 'monthrange has wrong format; must be "YYYY-MM:YYYY-MM"' )
        sys.stdout.flush()
        return 

    t0 = time()

    process( timerange, args.output, dataroot=args.dataroot, clobber=args.clobber )

    t1 = time()
    dt = t1 - t0
    print( f'total elapsed time = {int(dt/60):4d}m {int(dt)%60:2d}s' )
    sys.stdout.flush()


if __name__ == "__main__": 
    main()

