import os
import re
import sys
import argparse
from datetime import datetime, timedelta 
from netCDF4 import Dataset
import numpy as np
import boto3
from .libtropicalwidth import default_dataroot


#  Exception handling. 

class Error( Exception ): 
    pass

class ProcessOSCARError( Error ): 
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment


def compute_oscar_climatology( dataroot=default_dataroot, clobber=False ): 
    """Create climatologies of monthly average ocean surface currents from OSCAR data. 
    The data will be found in dataroot/oscar/YYYY/MM, where years YYYY and months MM will 
    be scanned for daily data files downloaded from the NASA PO-DAAC. If all days for the 
    month are found in the directory, then the daily data will be averaged into one monthly 
    file for that YYYY/MM.

    Set clobber=True if you wish to overwrite previously existing climatologies."""

    #  Path to OSCAR. 

    oscarroot = "/".join( [ dataroot, "oscar" ] )

    #  Defaults: time origin, temporary file. 

    dt0 = datetime( year=1990, month=1, day=1 )
    tmpfile = "tmp.nc"

    #  Walk through directory structure. 

    timerange = [ dt0, datetime.now() ]
    dt = timerange[0]

    while dt <= timerange[1]: 

        dt2 = dt + timedelta(days=31)
        dt2 = datetime( year=dt2.year, month=dt2.month, day=1 )
        ndays = ( dt2 - dt ).days

        ldir = os.path.join( dataroot, "oscar", f'{dt.year:4d}', f'{dt.month:02d}' )
        if not os.path.isdir( ldir ): 
            dt = dt2
            continue
        lpaths = [ os.path.join( ldir, f ) for f in os.listdir( ldir ) \
                if re.search( r'oscar_currents_[a-z]+_\d{8}.*\.nc', f ) ]

        if len(lpaths) != ndays: 
            print( 'Month {:} has only {:} files'.format( dt.strftime("%Y-%m"), len(lpaths) ) )
            sys.stdout.flush()
            dt = dt2
            continue

        opath = os.path.join( ldir, 'oscar_currents_{:}.nc4'.format( dt.strftime( "%Y%m" ) ) )

        if os.path.exists( opath ): 
            if clobber: 
                print( f'{opath} already exists; clobber' )
                sys.stdout.flush()
            else: 
                print( f'{opath} already exists; no clobber' )
                sys.stdout.flush()
                dt = dt2
                continue

        for ilpath, lpath in enumerate( lpaths ): 

            try: 
                d = Dataset( lpath, 'r' )
            except: 
                raise ProcessOSCARError( "UnreadableFile", f'Unreadable file: {lpath}' )

            if ilpath == 0: 

                #  Get metadata. 

                metadata = {}

                for var in [ 'lon', 'lat', 'u', 'v' ]: 
                    v = d.variables[var]
                    metadata[var] = { 'dtype': v.dtype, 'dimensions': tuple( [ dim for dim in v.dimensions if dim!='time' ] ), 'attrs': {} }
                    metadata[var]['attrs'] = { attr: v.getncattr(attr) for attr in v.ncattrs() } 

                #  Read coordinate grid. 

                lons = d.variables['lon'][:].squeeze()
                lats = d.variables['lat'][:].squeeze()

            #  Accumulate u, v field for surface current. 

            if ilpath == 0: 
                uc = d.variables['u'][:].squeeze()
                vc = d.variables['v'][:].squeeze()
            else: 
                uc += d.variables['u'][:].squeeze()
                vc += d.variables['v'][:].squeeze()

            d.close()

        #  Compute monthly average. 

        uc /= ndays
        vc /= ndays

        #  Create output file. 

        print( f'Creating {opath}' )
        sys.stdout.flush()

        d = Dataset( opath, 'w', format="NETCDF4" )

        #  Create dimensions. 

        d.createDimension( "longitude", lons.size )
        d.createDimension( "latitude", lats.size )

        #  Create variables. 

        for var, meta in metadata.items(): 
            v = d.createVariable( var, meta['dtype'], dimensions=meta['dimensions'] )
            v.setncatts( meta['attrs'] )

        v = d.createVariable( "time", np.int32 )
        v.setncatts( { 
                'description': "Time of climatology, number of days since {:}".format( dt0.strftime( "%Y-%m-%d" ) ), 
                'units': "days" } )

        #  Global attributes. 

        d.setncatts( { 
                'file_type': "OSCAR_monthly_climatology", 
                'year': np.int32( dt.year ), 
                'month': np.int32( dt.month ), 
                'creator_name': "Stephen Leroy (stephen.leroy@janusresearch.us)", 
                'origin': "NASA Physical Oceanography DAAC", 
                'doi': "10.5067/OSCAR-25F20", 
                'date_created': datetime.now().strftime( "%Y-%m-%d" ) } )

        #  Write data to output. 

        d.variables['lon'][:] = lons
        d.variables['lat'][:] = lats
        d.variables['time'].assignValue( ( dt - dt0 ).days )
        d.variables['u'][:] = uc
        d.variables['v'][:] = vc

        d.close()

        dt = dt2

    return


def main(): 

    #  Defaults. 

    parser = argparse.ArgumentParser( prog='process_oscar', description=
            """Create climatologies of monthly average ocean surface currents from OSCAR data. 
            The data will be found in dataroot/oscar/YYYY/MM, where years YYYY and months MM will 
            be scanned for daily data files downloaded from the NASA PO-DAAC. If all days for the 
            month are found in the directory, then the daily data will be averaged into one monthly 
            file for that YYYY/MM.""" )

    parser.add_argument( '--dataroot', default=default_dataroot, 
            help=f'The root data directory; "{default_dataroot}" by default' )
    parser.add_argument( '--clobber', default=False, action='store_true', 
            help='Clobber previously computed climatology; default is to skip when output file already exists' )

    args = parser.parse_args()

    ret = compute_oscar_climatology( args.dataroot, args.clobber )

    pass


if __name__ == "__main__": 
    main()

