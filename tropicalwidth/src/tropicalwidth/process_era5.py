import os
from netCDF4 import Dataset
from datetime import datetime, timedelta 
import numpy as np
import cartopy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import matplotlib
from .libtropicalwidth import default_dataroot

#  Physical parameters. 

gravity = 9.80665
Re = 6378.0e3
rads = np.pi / 180
fill_value = -1.22e20

#  Reference time. 

reference_time = datetime( year=1980, month=1, day=1 )


def process( t0, t1, outputfile, dataroot=default_dataroot ): 

    print( "Generating " + outputfile )

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


    firsttime = True
    itime = 0

    #  Loop over year. 

    for year in range( t0.year, t1.year ): 

        era5_input_file = os.path.join( dataroot, "era5", 'vwnd.plevels.monthly.{:4d}.nc'.format(year) )
        print( "Opening " + era5_input_file )
        d = Dataset( era5_input_file, 'r' )

        if firsttime: 

            #  Get coordinates. 

            lons = d.variables['longitude'][:]
            lats = d.variables['latitude'][:]
            levels = d.variables['level'][:]
            nlons, nlats, nlevels = lons.size, lats.size, levels.size
            ip = list(levels).index(500)

            w = np.zeros(nlevels)
            if levels[1] > levels[0]: 
                w[0] = ( levels[0] + levels[1] ) * 0.5
                w[1:ip] = ( levels[2:ip+1] - levels[0:ip-1] ) * 0.5
                w[ip] = ( levels[ip+1] - levels[ip] ) * 0.5
            else:
                w[-1] = ( levels[-1] + levels[-2] ) * 0.5
                w[ip+1:-1] = ( levels[ip:-2] - levels[ip+2:] ) * 0.5
                w[ip] = ( levels[ip] - levels[ip+1] ) * 0.5

            firsttime = False

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
    return


def main(): 

    parser = argparse.ArgumentParser( prog="process_era5", description='Compute the width of the ' + \
            'tropics according to the streamfunction definition from ERA5 meridional winds' )

    parser.add_argument( "yearrange", type=str, help='Range of years over which to ' + \ 
            'download ERA5 data, format "YYYY:YYYY", inclusive' )

    parser.add_argument( "output", type=str, help='Path to the output file' )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", default=default_dataroot, 
            help="Root of all data for the tropical width analysis project; " + \ 
                f'the default is "{default_dataroot}"' )

    args = parser.parse_args()

    m = re.search( r'^(\d{4}):(\d{4})$', args.yearrange )
    if not m:  
        print( 'Be sure that the yearrange has the format "YYYY:YYYY".' )
        exit()

    year0, year1 = int( m.group(1) ), int( m.group(2) )
    t0 = datetime( year=year0, month=1, day=1 )
    t1 = datetime( year=year1, month=12, day=31 )

    if not os.path.exists( era5file ): 
        process( t0, t1, args.outputfile, dataroot=dataroot )

