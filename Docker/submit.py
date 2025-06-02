#!/usr/bin/env python3

import os
import re
import sys
import argparse
from tropicalwidth.process import process 
from datetime import datetime
from time import time
# from tropicalwidth.libtropicalwidth import default_dataroot

default_dataroot = "./fg"

if __name__ == "__main__": 

    #  Define command line parser. 

    parser = argparse.ArgumentParser( description="Run a job on a workstation to process tropical width" )

    parser.add_argument( "yearrange", type=str, help="""The (inclusive) range of 
            years to process tropical width based on surface winds of CCMP, a 
            string of format "YYYY:YYYY" """ )
    parser.add_argument( "--dataroot", default=default_dataroot, 
            help=f'The root path for the data; the default is "{default_dataroot}"' )
    parser.add_argument( "--test", "-t", dest="test", default=False, action="store_true", 
            help='Run the program in test mode, generating messages indicating what jobs would be run' )
    parser.add_argument( "--clobber", "-c", dest="clobber", default=False, action="store_true",  
            help='Clobber previously existing output files; do not clobber by default' )

    #  Parse command line. 

    args = parser.parse_args()

    #  Compose list of jobs. 

    m = re.search( r'(\d{4}):(\d{4})$', args.yearrange )
    if m: 
        year0, year1 = int( m.group(1) ), int( m.group(2) )
    else: 
        print( 'Input yearrange must have format "YYYY:YYYY".' )
        exit()

    if args.test: 
        print( 'Jobs not being actually run...' )

    time_start = time()

    for year in range(year0,year1+1): 
        timerange = [ datetime(year,1,1), datetime(year,12,31) ]
        outputfile = os.path.join( args.dataroot, "output", f'tropicalwidth-{year:4d}.nc' )
        print( f'process( year={year:4d}, outputfile="{outputfile}", dataroot="{args.dataroot}" )' )
        sys.stdout.flush()
        if not args.test: 
            t0 = time()
            ret = process( timerange, outputfile, dataroot=args.dataroot, clobber=args.clobber )
            t1 = time()
            dt = t1 - t0
            print( f'  elapsed time = {int(dt/60):4d}m {int(dt)%60:2d}s' )

    time_end = time()
    dt = time_end - time_start
    if not args.test: 
        print( f'total elapsed time = {int(dt/60):4d}m {int(dt)%60:2d}s' )

