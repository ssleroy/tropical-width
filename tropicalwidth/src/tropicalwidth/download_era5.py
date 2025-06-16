import os
import re
import argparse
import cdsapi
from .libtropicalwidth import default_dataroot

#  ERA5 download client. 

cds = cdsapi.Client()


def download_v( year, dataroot=default_dataroot, clobber=False ): 
    """Download ERA5 monthly average meridional winds on pressure levels 
    for a specified year."""

    era5root = os.path.join( dataroot, "era5" )

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"vwnd.plevels.monthly.{year:4d}.nc" )
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
                 ddict, localfile )

    if not os.path.exists( localfile ):  
        print( 'Unsuccessful. Be sure you have the correct tokens for access to the ' + \
                'Copernicus Data Store in ~/.cdsapirc. See https://cds.climate.copernicus.eu/how-to-api.' )
        ret = ""
    else: 
        ret = localfile
    return ret


def download_ps( year, dataroot=default_dataroot, clobber=False ): 
    """Download ERA5 monthly average surface pressure for a specified year."""

    era5root = os.path.join( dataroot, "era5" )

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"pressure.surface.monthly.{year:4d}.nc" )
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
                 ddict, localfile )

    return localfile


def download_us( year, dataroot=default_dataroot, clobber=False ): 
    """Download ERA5 monthly average surface air zonal wind for a specified year."""

    era5root = os.path.join( dataroot, "era5" )

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"uwnd.surface.monthly.{year:4d}.nc" )
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
                 ddict, localfile )

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

    yearrange = [ int( m.group(1) ), int( m.group(2) ) ]
    for year in range( yearrange[0], yearrange[1]+1 ): 
        ret1 = download_v( year, dataroot=args.dataroot, clobber=args.clobber )
        ret2 = download_ps( year, dataroot=args.dataroot, clobber=args.clobber )
        ret3 = download_us( year, dataroot=args.dataroot, clobber=args.clobber )

    return 


if __name__ == "__main__": 
    main()
    pass

