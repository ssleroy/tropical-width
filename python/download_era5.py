import os
import pdb

projectroot = os.path.join( os.path.expanduser("~"), "Projects", "TropicalWidth" )
gitroot = os.path.join( projectroot, "tropical-width" )
dataroot = os.path.join( projectroot, "Data" )
era5root = os.path.join( dataroot, "era5" )

#  ERA5 download client. 

import cdsapi
cds = cdsapi.Client()

def download_v( year ): 
    """Download ERA5 monthly average meridional winds on pressure levels 
    for a specified year."""

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"vwnd.plevels.monthly.{year:4d}.nc" )
    if os.path.exists( localfile ): 
        return localfile

    head, tail = os.path.split( localfile )
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

    return localfile


def download_ps( year ): 
    """Download ERA5 monthly average surface pressure for a specified year."""

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"pressure.surface.monthly.{year:4d}.nc" )
    if os.path.exists( localfile ): 
        return localfile

    head, tail = os.path.split( localfile )
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


def download_us( year ): 
    """Download ERA5 monthly average surface air zonal wind for a specified year."""

    #  Define the local file name. 

    localfile = os.path.join( era5root, f"uwnd.surface.monthly.{year:4d}.nc" )
    if os.path.exists( localfile ): 
        return localfile

    head, tail = os.path.split( localfile )
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


if __name__ == "__main__": 
    # pdb.set_trace()

    for year in range( 1989, 2023 ): 
        ret = download_v( year )
        print( f'Obtained {ret}' )
        ret = download_ps( year )
        print( f'Obtained {ret}' )
        ret = download_us( year )
        print( f'Obtained {ret}' )

