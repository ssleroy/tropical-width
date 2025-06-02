import os
from netCDF4 import Dataset
from datetime import datetime, timedelta 
import numpy as np
import cartopy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import matplotlib
from resources import dataroot, era5file
from tropicalwidth.libtropicalwidth import bucket
import subprocess


#  Physical parameters. 

gravity = 9.80665
Re = 6378.0e3
rads = np.pi / 180
fill_value = -1.22e20

#  Reference time. 

reference_time = datetime( year=1980, month=1, day=1 )


def process( t0, t1 ): 

    print( "Generating " + era5file )

    o = Dataset( era5file, 'w', format='NETCDF4' )

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

    for year in range( t0.year, t1.year+1 ): 

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


def trends( t0, t1 ): 

    print( 'Trends for {:4d} through {:4d}:'.format(t0.year,t1.year) )

    d = Dataset( era5file, 'r' )
    times = d.variables['times'][:]

    seasons = [ ('DJF',-1), ('MAM',2), ('JJA',5), ('SON',8) ]

    year0 = t0.year
    year1 = t1.year
    nyears = year1 - year0 + 1

    #  Trends in NH. 

    for season, offsetmonths in seasons: 

        y = np.zeros( nyears, dtype='f' ) + fill_value
        lats = d.variables['latNH'][:]

        for iyear, year in enumerate( np.arange( year0, year1+1 ) ): 
            t = datetime( year=year, month=1, day=15 ) + timedelta( days=offsetmonths*30 )
            t = datetime( year=t.year, month=t.month, day=1 )
            ta = ( t - reference_time ).days
            i = np.logical_and( times >= ta, times <= ta+63 )
            if np.logical_not( lats[i].mask.sum() ) > 0: 
                y[iyear] = lats[i].mean()

        y = np.ma.masked_where( np.logical_or( y==fill_value, np.isnan(y) ), y )
        dt = np.ma.masked_where( y.mask, np.arange( y.size ) )
        dt = dt - dt.mean()
        sumdt2 = ( dt**2 ).sum()
        trend = ( dt * y ).sum() / sumdt2
        sigma2 = ( ( y - y.mean() - trend * dt )**2 ).sum() / ( y.size - 2 )
        uncertainty = np.sqrt( sigma2 / sumdt2 )

        print( '  trend for {}(NH) = {:5.2f} ({:4.2f}) degs/decade'.format( season, trend*10, uncertainty*10 ) )

    #  Trends in SH. 

    for season, offsetmonths in seasons: 

        y = np.zeros( nyears, dtype='f' ) + fill_value
        lats = d.variables['latNH'][:]

        for iyear, year in enumerate( np.arange( year0, year1+1 ) ): 
            t = datetime( year=year, month=1, day=15 ) + timedelta( days=offsetmonths*30 )
            t = datetime( year=t.year, month=t.month, day=1 )
            ta = ( t - reference_time ).days
            i = np.logical_and( times >= ta, times <= ta+63 )
            if np.logical_not( lats[i].mask.sum() ) > 0: 
                y[iyear] = lats[i].mean()

        y = np.ma.masked_where( np.logical_or( y==fill_value, np.isnan(y) ), y )
        dt = np.ma.masked_where( y.mask, np.arange( y.size ) )
        dt = dt - dt.mean()
        sumdt2 = ( dt**2 ).sum()
        trend = ( dt * y ).sum() / sumdt2
        sigma2 = ( ( y - y.mean() - trend * dt )**2 ).sum() / ( y.size - 2 )
        uncertainty = np.sqrt( sigma2 / sumdt2 )

        print( '  trend for {}(SH) = {:5.2f} ({:4.2f}) degs/decade'.format( season, trend*10, uncertainty*10 ) )

    d.close()


def psvariability( t0, t1 ): 

    seasons = [ ('DJF',-1), ('MAM',2), ('JJA',5), ('SON',8) ]

    #  Initialize plot. 

    matplotlib.rc( 'font', family='Times New Roman', weight='normal', size=8 )
    fig = plt.figure(figsize=(5.5,4)) 
    proj = cartopy.crs.PlateCarree( central_longitude=180 )
    axes = []
    iaxis = 0

    year0, year1 = t0.year, t1.year

    for season, monthsoffset in seasons: 

        #  Establish year range. 

        timeseries = []

        for year in range( year0, year1+1 ): 
            t = datetime( year=year, month=1, day=1 ) + timedelta( days=monthsoffset*30 )
            t = datetime( year=t.year, month=t.month, day=1 )

            for imonth in range(3): 
                era5_input_file = os.path.join( dataroot, "era5", "pressure.surface.monthly.{:4d}.nc".format(t.year) )
                d = Dataset( era5_input_file, 'r' )
                if 'lons' not in locals(): 
                    lons = d.variables['longitude'][:]
                    lats = d.variables['latitude'][:]
                    nlons, nlats = lons.size, lats.size
                    lats2 = lats[ np.indices((nlats,nlons))[0,:,:] ]
                    lons2 = lons[ np.indices((nlats,nlons))[1,:,:] ]
                if imonth==0: seasonalavg = np.zeros( (nlons,nlats), dtype='d' )
                seasonalavg = seasonalavg + d.variables['sp'][t.month-1,:,:].squeeze().T
                d.close()
                t += timedelta(days=31)
                t = datetime( year=t.year, month=t.month, day=1 )
            timeseries.append( seasonalavg/3.0 )

        timeseries = np.array( timeseries )
        variability = np.sqrt( ( timeseries**2 ).mean(axis=0) - (timeseries.mean(axis=0))**2 )
        timeseries = None

        ix, iy = len(axes) % 2, len(axes) // 2
        position = [ 0.02 + 0.50*ix, 0.22 + (1-iy)*0.41, 0.46, 0.30 ]
        ax = plt.subplot( 2, 2, len(axes)+1, projection=proj )
        ax.set_position( position )
        ax.coastlines( linewidth=0.5 )
        ax.set_title( '({}) {}'.format( chr(len(axes)+ord('a')), season ), loc='left' )
        clevels = np.arange(0,6.1,0.5)
#       cyclic_variability, cyclic_lons = add_cyclic_point(variability, axis=0), add_cyclic_point(lons)
        CS = ax.contourf( lons, lats, 0.01 * variability.T, \
            clevels, cmap=plt.cm.YlGn, extend='max', transform=proj )
        ax.contour( lons, lats, 0.01 * variability.T, [lev for lev in CS.levels if lev==int(lev)], \
            linewidths=0.5, colors='#808080' )

        axes.append( ax )

    #  Colorbar. 

    ax = fig.add_axes( [0.10,0.12,0.80,0.03] )
    cbar = fig.colorbar( CS, cax=ax, orientation='horizontal' )
    cbar.set_ticks( np.arange(0,6.1,1) )
    cbar.set_label( 'Variability [hPa]' )


    epsfile = 'psvariability.eps'
    print( "Generating " + epsfile )
    plt.savefig( epsfile, format='eps' )


if __name__ == "__main__": 

    t0 = datetime(year=1995,month=1,day=1)
    t1 = datetime(year=2024,month=12,day=31)

    if not os.path.exists( era5file ): 
        process( t0, t1 )

    # trends( t0, t1 )

    #t0.set(year=1980,month=12,day=1)
    #t1.set(year=2017,month=12,day=1)
    #trends(t0,t1)

    # psvariability( t0, t1 )


