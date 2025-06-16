import os
import re
import argparse
from netCDF4 import Dataset
from datetime import datetime, timedelta 
import numpy as np
from .libtropicalwidth import default_dataroot


#  Physical parameters. 

gravity = 9.80665           # J / kg / meter
Re = 6378.0e3               # m
rads = np.pi / 180
fill_value = -1.22e20

#  Reference time. 

reference_time = datetime( year=1980, month=1, day=1 )


def process_era5( yearrange, outputfile, dataroot=default_dataroot, clobber=False, 
                 streamfunction_plevel=500 ): 
    """Compute width of the Tropics according to the nulls in the mean meridional 
    streamfunction at 500 hPa. 

    yearrange           A two-tuple, two-element list, or 2-element ndarray of 
                        the year range over which to compute tropical width; the 
                        range is inclusive

    outputfile          The path to the output NetCDF file

    dataroot            The root directory for all data associated with the 
                        tropical width project

    clobber             Whether or not to clobber previously existing output 
                        file"""


    if os.path.exists( outputfile ): 
        if clobber: 
            print( f'{outputfile} already exists; clobbering' )
            os.unlink( outputfile )
        else: 
            print( f'{outputfile} already exists; skipping' )
            return

    print( f'Generating {outputfile}' )

    o = Dataset( outputfile, 'w', format='NETCDF4' )

    #  Define dimensions. 

    o.createDimension( 'time' )

    #  Define variables. 

    years = o.createVariable( 'years', 'i', ('time',) )
    years.setncatts( {
        'description': "Year of time interval", 
        'units': "year" 
        } )

    months = o.createVariable( 'months', 'i', ('time',) )
    months.setncatts( { 
        'description': "Month of time interval", 
        'units': "month"
        } )

    times = o.createVariable( 'times', 'd', ('time',) )
    times.setncatts( {
        'description': "time interval", 
        'units': "days since " + reference_time.strftime( "%d %B %Y" )
        } )

    latNH = o.createVariable( 'latNH', 'f', ('time',) )
    latNH.setncatts( { 
        'description': "Latitude of the extent of the Tropics into the Northern Hemisphere", 
        'units': 'degrees north' 
        } )

    latSH = o.createVariable( 'latSH', 'f', ('time',) )
    latSH.setncatts( { 
        'description': "Latitude of the extent of the Tropics into the Southern Hemisphere", 
        'units': "degrees north" 
        } )


    itime = 0

    #  Loop over year. 

    for year in range( yearrange[0], yearrange[1]+1 ): 

        era5_input_file = os.path.join( dataroot, "era5", 'vwnd.plevels.monthly.{:4d}.nc'.format(year) )
        print( f'Opening {era5_input_file}' )
        d = Dataset( era5_input_file, 'r' )

        if year == yearrange[0]: 

            #  Get coordinates. 

            lons = d.variables['longitude'][:]
            lats = d.variables['latitude'][:]
            levels = d.variables['level'][:]
            nlons, nlats, nlevels = lons.size, lats.size, levels.size
            ip = list(levels).index(streamfunction_plevel)

            w = np.zeros(nlevels)
            if levels[1] > levels[0]: 
                w[0] = ( levels[0] + levels[1] ) * 0.5
                w[1:ip] = ( levels[2:ip+1] - levels[0:ip-1] ) * 0.5
                w[ip] = ( levels[ip+1] - levels[ip] ) * 0.5
            else:
                w[-1] = ( levels[-1] + levels[-2] ) * 0.5
                w[ip+1:-1] = ( levels[ip:-2] - levels[ip+2:] ) * 0.5
                w[ip] = ( levels[ip] - levels[ip+1] ) * 0.5

        for imonth in range(12): 

            #  Loop over month of year. 

            t = datetime( year=year, month=imonth+1, day=1 )
            years[itime] = t.year
            months[itime] = t.month
            times[itime] = ( t - reference_time ).days

            #  Retrieve meridional wind for imonth and perform zonal average. Evaluate 
            #  streamfunction. 

            vbar = d.variables['v'][imonth,:,:,:].squeeze().mean(axis=2).T
            sfunc = np.inner( vbar, w ) * 2*np.pi * Re * np.cos(lats*rads) / gravity

            #  Search for NH Hadley cell edge. 

            m = np.logical_and( lats >= 20.0, lats <= 48.0 )
            m = np.logical_and( m[0:-1], sfunc[0:-1] * sfunc[1:] <= 0.0 )
            if np.any( m ): 
                ilat = np.argwhere(m)[0][0]
                tlat = sfunc[ilat] / ( sfunc[ilat] - sfunc[ilat+1] )
                lat = lats[ilat] * (1-tlat) + lats[ilat+1] * tlat
                latNH[itime] = lat
            else: 
                lat = None

            #  Search for SH Hadley cell edge. 

            m = np.logical_and( lats <= -20.0, lats >= -48.0 )
            m = np.logical_and( m[0:-1], sfunc[0:-1] * sfunc[1:] <= 0.0 )
            if np.any( m ): 
                ilat = np.argwhere(m)[0][0]
                tlat = sfunc[ilat] / ( sfunc[ilat] - sfunc[ilat+1] )
                lat = lats[ilat] * (1-tlat) + lats[ilat+1] * tlat
                latSH[itime] = lat
            else: 
                lat = None

            #  Increment itime and continue. 

            itime = itime + 1

        #  Next year. 

        d.close()

    #  Done. 

    o.close()
    print( f'Successfully created {outputfile}' )

    return


def main(): 

    parser = argparse.ArgumentParser( description="Process ERA5 for the width of the Tropics according to the mean meridional streamfunction." )

    parser.add_argument( 'yearrange', type=str, help='Range of years over which to compute the width of the Tropics as ' + \
            '"YYYY:YYYY"; the range is inclusive' )

    parser.add_argument( 'outputfile', type=str, help='Path of the output NetCDF file' )

    parser.add_argument( '--clobber', '-c', dest="clobber", default=False, action="store_true", 
            help="Clobber pre-existing output file; default is not to clobber" )

    args = parser.parse_args()

    m = re.search( r'^(\d{4}):(\d{4})$', args.yearrange )
    if not m: 
        print( f'The yearrange must have format "YYYY:YYYY".' )
        return
    yearrange = ( int( m.group(1) ), int( m.group(2) ) )

    ret = process_era5( yearrange, args.outputfile, clobber=args.clobber )

    return


if __name__ == "__main__": 
    main()
    pass

