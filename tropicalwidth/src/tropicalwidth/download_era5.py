import os
import re
import argparse
import cdsapi
from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta, timezone
from .libtropicalwidth import default_dataroot

#  ERA5 download client. 

cds = cdsapi.Client()

#  Epoch for standard format. 

epoch = datetime( year=1900, month=1, day=1 )


#  Exception handling. 

class Error( Exception ): 
    pass

class ucar2cdsError( Error ):
    def __init__( self, message, comment ): 
        self.message = message
        self.comment = comment


def transform( cdsfile, outputfile ): 
    """Translate a file downloaded from the Copernicus Climate Data Store (CDS) 
    containing ERA5 data into a standard format NetCDF file."""

    d = Dataset( cdsfile, 'r' )
    e = Dataset( outputfile, 'w', format="NETCDF4" )

    #  Get time values. 

    v = d.variables['valid_time']
    m = re.search( r'(\w+) since (\d{4})-(\d{2})-(\d{2})', v.getncattr("units") )
    if m: 
        input_units = m.group(1).lower()
        year, month, day = int( m.group(2) ), int( m.group(3) ), int( m.group(4) )
        input_epoch = datetime( year=year, month=month, day=day )
        times = [ input_epoch + timedelta( **{ input_units: int(t) } ) for t in v[:] ]
    else: 
        raise ucar2cdsError( "InvalidFormat", 'Unable to parse "units" attribute of "valid_time" variable' )

    #  Define dimensions. 

    e.createDimension( "time", d.dimensions['valid_time'].size )
    e.createDimension( "longitude", d.dimensions['longitude'].size )
    e.createDimension( "latitude", d.dimensions['latitude'].size )
    if "pressure_level" in d.dimensions.keys(): 
        ndims = 4
        e.createDimension( "level", d.dimensions['pressure_level'].size )
    else: 
        ndims = 3

    #  Define coordinate variables. 

    v = e.createVariable( "time", np.int32, dimensions=("time",) )
    v.setncatts( { 
            'units': "hours since " + epoch.strftime( "%Y-%m-%d %H:%M:%S.0" ), 
            'long_name': "time", 
            'calendar': "gregorian" } )

    v = e.createVariable( "longitude", np.float32, dimensions=("longitude",) )
    v.setncatts( { 
            'units': "degrees_east", 
            'long_name': "longitude" } )

    v = e.createVariable( "latitude", np.float32, dimensions=("latitude",) )
    v.setncatts( { 
            'units': "degrees_north", 
            'long_name': "latitude" } )

    if ndims == 4: 
        v = e.createVariable( "level", np.float32, dimensions=("level",) )
        v.setncatts( { 
                'units': "millibars", 
                'long_name': "pressure_level" } )

    #  Define data variable. 

    for vname, vdata_input in d.variables.items(): 
        if len( vdata_input.shape ) == ndims: 
            found = True
            break 
        else: 
            found = False

    if found: 
        #  Define compression. 
        vals = vdata_input[:]
        vmin, vmax = vals.min(), vals.max()
    else: 
        d.close()
        e.close()
        os.unlink( outputfile )
        raise ucar2cdsError( "NoDataVariable", f'Unable to find a data variable in file {cdsfile}' )

    if ndims == 3: 
        vdata_output = e.createVariable( vname, np.float32, dimensions=("time","latitude","longitude") )
        vdata_output.setncatts( { 
                'long_name': vdata_input.getncattr("long_name"), 
                'standard_name': vdata_input.getncattr("standard_name"), 
                'units': vdata_input.getncattr("units") } )
    elif ndims == 4: 
        vdata_output = e.createVariable( vname, np.float32, dimensions=("time","level","latitude","longitude") )
        vdata_output.setncatts( { 
                'long_name': vdata_input.getncattr("long_name"), 
                'standard_name': vdata_input.getncattr("standard_name"), 
                'units': vdata_input.getncattr("units") } )

    #  Global attributes. 

    e.setncatts( { 
            'history': "{:} by cds2stp.py: cds2std {:} {:}".format( 
                datetime.now().astimezone().strftime( "%Y-%m-%d %H:%M:%S %Z" ), 
                cdsfile, outputfile )
            } )

    #  Coordinate data values. 

    outlons = np.arange( 0.00, 360.00, 0.25, dtype=np.float32 )
    outlats = np.arange( 90.00, -90.01, -0.25, dtype=np.float32 )

    #  Flip pressure coordinate? 

    if ndims == 4: 
        pflip = ( d.variables['pressure_level'][1] < d.variables['pressure_level'][0] )

    #  Is a longitude shift needed? 

    ilons = np.argwhere( d.variables['longitude'][:] == outlons[0] ).squeeze() + np.arange(outlons.size,dtype=np.int32)
    ilons[ilons >= outlons.size] -= outlons.size

    #  Flip latitudes? 

    latflip = ( d.variables['latitude'][1] > d.variables['latitude'][0] )

    #  Write coordinate data values. 

    e.variables['longitude'][:] = outlons
    e.variables['latitude'][:] = outlats

    if ndims == 4: 
        if pflip: 
            e.variables['level'][:] = np.flip( d.variables['pressure_level'][:] )
        else: 
            e.variables['level'][:] = d.variables['pressure_level'][:]

    e.variables['time'][:] = [ ( t - epoch ) / timedelta(hours=1) for t in times ]

    #  Write data field. 

    if ndims == 3: 
        if latflip: 
            vdata_output[:] = np.flip( vdata_input[:,:,ilons], axis=1 )
        else: 
            vdata_output[:] = vdata_input[:,:,ilons]
    elif ndims == 4: 
        if latflip: 
            if pflip: 
                ovals = np.flip( np.flip( vdata_input[:,:,:,ilons], axis=2 ), axis=1 )
            else: 
                ovals = np.flip( vdata_input[:,:,:,ilons], axis=2 )
        else: 
            if pflip: 
                ovals = np.flip( vdata_input[:,:,:,ilons], axis=1 )
            else: 
                ovals = vdata_input[:,:,:,ilons]
        vdata_output[:] = ovals 

    #  Done. 

    d.close()
    e.close()
    
    print( f'Output written to {outputfile}' )
    return outputfile


def download_v( year, dataroot=default_dataroot, clobber=False ): 
    """Download ERA5 monthly average meridional winds on pressure levels 
    for a specified year."""

    era5root = os.path.join( dataroot, "era5" )

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"vwnd.plevels.monthly.{year:4d}.nc" )
    tmpfile = f"tmp_vwnd.plevels.monthly.{year:4d}.nc" 

    if os.path.exists( localfile ): 
        if clobber: 
            print( f'{localfile} alread exists; clobbering' )
            os.unlink( localfile )
        else: 
            print( f'{localfile} alread exists; skipping' )
            return localfile

    head, tail = os.path.split( localfile )
    if head != "": 
        os.makedirs( head, exist_ok=True )
    print( f'Downloading {tail}' )

    #  Define the download definition dictionary. 

    ddict = {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'v_component_of_wind',
        'pressure_level': [
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000'
        ],
        'year': f'{year:4d}',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12'
        ],
        'time': '00:00'
    }

    #  Download. 

    cds.retrieve( 'reanalysis-era5-pressure-levels-monthly-means',
                 ddict, tmpfile )

    if not os.path.exists( tmpfile ):  
        print( 'Unsuccessful. Be sure you have the correct tokens for access to the ' + \
                'Copernicus Data Store in ~/.cdsapirc. See https://cds.climate.copernicus.eu/how-to-api.' )
        ret = ""
    else: 
        ret = transform( tmpfile, localfile )
        os.unlink( tmpfile )
    return ret


def download_ps( year, dataroot=default_dataroot, clobber=False ): 
    """Download ERA5 monthly average surface pressure for a specified year."""

    era5root = os.path.join( dataroot, "era5" )

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"pressure.surface.monthly.{year:4d}.nc" )
    tmpfile = f"tmp_pressure.surface.monthly.{year:4d}.nc"

    if os.path.exists( localfile ): 
        if clobber: 
            print( f'{localfile} alread exists; clobbering' )
            os.unlink( localfile )
        else: 
            print( f'{localfile} alread exists; skipping' )
            return localfile

    head, tail = os.path.split( localfile )
    if head != "": 
        os.makedirs( head, exist_ok=True )
    print( f'Downloading {tail}' )

    #  Define the download definition dictionary. 

    ddict = {
        'format': 'netcdf',
        'variable': 'surface_pressure',
        'product_type': 'monthly_averaged_reanalysis',
        'year': f'{year:4d}',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12'
        ],
        'time': '00:00'
    }

    #  Download. 

    cds.retrieve( 'reanalysis-era5-single-levels-monthly-means', 
                 ddict, tmpfile )

    if not os.path.exists( tmpfile ):  
        print( 'Unsuccessful. Be sure you have the correct tokens for access to the ' + \
                'Copernicus Data Store in ~/.cdsapirc. See https://cds.climate.copernicus.eu/how-to-api.' )
        ret = ""
    else: 
        ret = transform( tmpfile, localfile )
        os.unlink( tmpfile )
    return ret

    return localfile


def download_us( year, dataroot=default_dataroot, clobber=False ): 
    """Download ERA5 monthly average surface air zonal wind for a specified year."""

    era5root = os.path.join( dataroot, "era5" )

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"uwnd.surface.monthly.{year:4d}.nc" )
    tmpfile = f"tmp_uwnd.surface.monthly.{year:4d}.nc"

    if os.path.exists( localfile ): 
        if clobber: 
            print( f'{localfile} alread exists; clobbering' )
            os.unlink( localfile )
        else: 
            print( f'{localfile} alread exists; skipping' )
            return localfile

    head, tail = os.path.split( localfile )
    if head != "": 
        os.makedirs( head, exist_ok=True )
    print( f'Downloading {tail}' )

    #  Define the download definition dictionary. 

    ddict = {
        'format': 'netcdf',
        'variable': '10m_u_component_of_wind',
        'product_type': 'monthly_averaged_reanalysis',
        'year': f'{year:4d}',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12'
        ],
        'time': '00:00'
    }

    #  Download. 

    cds.retrieve( 'reanalysis-era5-single-levels-monthly-means', 
                 ddict, tmpfile )

    if not os.path.exists( tmpfile ):  
        print( 'Unsuccessful. Be sure you have the correct tokens for access to the ' + \
                'Copernicus Data Store in ~/.cdsapirc. See https://cds.climate.copernicus.eu/how-to-api.' )
        ret = ""
    else: 
        ret = transform( tmpfile, localfile )
        os.unlink( tmpfile )
    return ret

    return localfile


def main(): 

    parser = argparse.ArgumentParser( description="Download ERA5 files needed for " + \
            "analyzing the width of the Tropics and trends in that width." )

    parser.add_argument( "yearrange", type=str, help='Range of years over which to ' + \
            'download ERA5 data, format "YYYY:YYYY", inclusive' )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", default=default_dataroot, 
            help="Root of all data for the tropical width analysis project; " + \
                f'the default is "{default_dataroot}"' )

    parser.add_argument( "--clobber", "-c", dest="clobber", default=False, action="store_true", 
            help="Clobber pre-existing download; default is not to clobber" )

    args = parser.parse_args()

    m = re.search( r'^(\d{4}):(\d{4})$', args.yearrange )
    if not m: 
        print( 'Be sure that the yearrange has the format "YYYY:YYYY".' )
        exit()

    year0, year1 = int( m.group(1) ), int( m.group(2) )
    for year in range( year0, year1+1 ): 
        ret1 = download_v( year, dataroot=args.dataroot, clobber=args.clobber )
        ret2 = download_ps( year, dataroot=args.dataroot, clobber=args.clobber )
        ret3 = download_us( year, dataroot=args.dataroot, clobber=args.clobber )

    return 


if __name__ == "__main__": 
    main()
    pass


