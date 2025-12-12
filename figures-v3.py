# Linear Regression Analysis and Figures, revision
# 
# Generate figures for a paper on widening of the Tropics. The central analysis will be on 
# latitudes of nulls in zonal mean zonal surface winds. It will also include nulls in the 
# streamfunction in atmospheric analyses. 
# 
# 1. Import packages, environment variables, and define plot defaults
# 2. Define linear regression functions
#     * Linear regression
#     * Expand linear regression
# 3. Linear regression analyses
#     * PDO timeseries
#     * AMO timeseries
#     * CCMP trend analysis
#     * ERA5 trend analysis
#     * CMIP6 trend analyses
#     * Inter-monthly variability analysis
#     * Trend in total width of Tropics
# 5. Generate figures
#     * PDO timeseries
#     * AMO timeseries
#     * Timeseries of u=0 latitude
#     * Trends by season
#     * Zonal mean zonal wind climatology
#     * CCMP and ERA5 timeseries, full globe
#     * CCMP and ERA5 timeseries, full globe, deseasonalized
#     * Table of trends
#     * Table of PDO
#     * Table of AMO
#     * CCMP sounding density maps
#     * Sounding density timeseries
#     * Spatial variance of zonal wind
#     * Timeseries of index regional winds
#     * Regression of index regional winds onto the 500-hPa geopotential field
#    

## 1. Import packages, environment variables, and define plot defaults
 
# Import packages. AWS S3 client. Assume authentication in the environment, either 
# by environment variables or by service permissions. 

import os
import re
import json
import requests
from time import time
import numpy as np
from netCDF4 import Dataset, MFDataset
from datetime import datetime, timedelta
import subprocess
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cartopy.crs as ccrs
from tropicalwidth.libtropicalwidth import Model, averageWind

#  Physical constants. 

gravity = 9.80665   #  J/kg/m
Re = 6378.135      # km

#  AWS S3 client. 

import boto3
from tropicalwidth.libtropicalwidth import bucket
s3 = boto3.client( "s3" )

#  Default matplotlib settings. 

axeslinewidth = 0.5
plt.rcParams.update( {
  'font.family': "Helvetica", 
  'font.size': 9, 
  'font.weight': "normal", 
  'text.usetex': True, 
  'xtick.major.width': axeslinewidth, 
  'xtick.minor.width': axeslinewidth, 
  'ytick.major.width': axeslinewidth, 
  'ytick.minor.width': axeslinewidth, 
  'axes.linewidth': axeslinewidth } )

#  Environment variables. 

DATAROOT = os.getenv( "DATAROOT" )
if DATAROOT is None: 
    print( "Be sure to define the environment variables DATAROOT." )
    exit()

ERA5ANALYSIS = os.getenv( "ERA5ANALYSIS" )
if ERA5ANALYSIS is None: 
    ERA5ANALYSIS = "tropicalwidth_era5.nc"

CMIP6ANALYSIS = os.getenv( "CMIP6ANALYSIS" )
if CMIP6ANALYSIS is None: 
    CMIP6ANALYSIS = "tropicalwidth_cmip6.nc"

print( f'  DATAROOT={DATAROOT}\nERA5ANALYSIS={ERA5ANALYSIS}\nCMIP6ANALYSIS={CMIP6ANALYSIS}' )

#  Define seasons. 

seasons = [ { 'name': "All", 'monthrange': [1,12] }, 
            { 'name': "DJF", 'monthrange': [12,2] }, 
            { 'name': "MAM", 'monthrange': [3,5] }, 
            { 'name': "JJA", 'monthrange': [6,8] }, 
            { 'name': "SON", 'monthrange': [9,11] } 
          ]


## 2. Linear regression functions

def regress( times, vals, pdo=None, amo=None, noautocorrelation=False ):
    """Perform a linear regression analysis on a monthly timeseries of values (vals). 
    The time coordinates for those values are given by times, each of element of which 
    has units of years and points to the middle of the month (year+(month-0.5)/12). The 
    assumption is that the annual cycle is removed. Values are regressed against a line 
    and an intercept: normal linear regression. 

    If "pdo" is set to pdo as defined earlier in this notebook, then regression is also 
    performed against the PDO index timeseries "rindices". To do this, set "pdo=pdo". 

    If "amo" is set to amo as defined earlier in this notebook, then regression is also 
    performed against the AMO index timeseries "rindices". To do this, set "amo=amo". 

    By default, serial autocorrelation is considered in the error analysis. by setting 
    "noautocorrelation" to True, serial autocorrelation is ignored. This option should 
    be used when considering seasonal rather than all-year time series. 

    The output is a dictionary with the best fit values of linear trend ( Units(vals) 
    per year ) and the uncertainty covariance matrix."""

    #  How many months in a year? 

    medianyear = int( times.mean() )
    ii = np.argwhere( np.logical_and( times > medianyear, times < medianyear+1 ) )
    nmonths = ii.size

    #  Just the unmasked values. 

    try: 
        if vals.mask.any(): 
            igood = np.argwhere( np.logical_not( vals.mask ) ).squeeze()
        else: 
            igood = np.arange( vals.size )
    except: 
        igood = np.arange( vals.size )

    timereference = times[igood].mean()

    #  What to account for? PDO? AMO? 

    labels = [ "bias", "trend" ]
    if pdo is not None: 
        labels.append( "pdo" )
    if amo is not None: 
        labels.append( "amo" )

    fp = np.zeros( (igood.size,len(labels)), dtype=np.float32 )

    for ilabel, label in enumerate( labels ): 

        if label == "bias": 
            fp[:,ilabel] = 1.0

        elif label == "trend": 
            x = times[igood] - timereference
            fp[:,ilabel] = x

        elif label == "pdo": 
            #  Extract relevant times. 
            pvals = [] 
            for time in times[igood]:
                ii = np.argwhere( np.abs( pdo['times'] - time ) < 0.5 / 12 )[0] 
                pvals.append( pdo['rindices'][ii] )
            #  Remove the mean. 
            pvals = np.array( pvals ).flatten()
            fp[:,ilabel] = pvals - pvals.mean()

        elif label == "amo": 
            #  Extract relevant times. 
            pvals = [] 
            for time in times[igood]:
                ii = np.argwhere( np.abs( amo['times'] - time ) < 0.5 / 12 )[0] 
                pvals.append( amo['rindices'][ii] )
            #  Remove the mean. 
            pvals = np.array( pvals ).flatten()
            fp[:,ilabel] = pvals - pvals.mean()

    #  Best fit coefficients. 

    Ainv = np.linalg.inv( fp.T @ fp ) 
    coeffs = Ainv @ fp.T @ vals[igood]

    #  Calculate residuals and standard deviation. 

    residuals = vals[igood] - fp @ coeffs
    var = ( residuals**2 ).sum() / ( residuals.size - nmonths - fp.shape[1] )
    if noautocorrelation: 
        phi = 0.0
    else: 
        phi = ( residuals[:-1] * residuals[1:] ).sum() / ( residuals.size - nmonths - fp.shape[1] ) / var

    stddev = np.sqrt( var * (1+phi)/(1-phi) )

    #  Uncertainty covariancematrix. 

    cov = Ainv * stddev**2

    #  Return dictionary. 

    ret = { 'labels': labels, 'coefficients': coeffs, 'uncertainty': cov, 'timereference': timereference }

    return ret


def expand( times, coeffs, covariance, timereference, pdo=None ):
    """Expand a fit to timeseries analysis generated by the function regress. """

    #  Form temporal fingerprints. 

    x = times - timereference

    if pdo is None:
        fp = np.zeros( (x.size,2), dtype=np.float32 )
        fp[:,0] = 1.0
        fp[:,1] = x
    else:
        fp = np.zeros( (x.size,3), dtype=np.float32 )
        fp[:,0] = 1.0
        fp[:,1] = x
        pvals = [] 

        for time in times:
            ii = np.argwhere( np.abs( pdo['times'] - time ) < 0.5 / 12 )[0] 
            pvals.append( pdo['rindices'][ii] )
        fp[:,2] = np.array( pvals ).flatten()

    #  Best fit coefficients. 

    fit = ( fp @ coeffs ).squeeze()

    #  Error bars. 

    ebars = np.sqrt( ( fp @ covariance @ fp.T ).diagonal() )
    
    #  Return dictionary. 

    ret = { 'times': times, 'fit': fit, 'ebars': ebars }
    
    return ret


## 3. Linear regression analyses

def get_pdo_timeseries(): 
    """Retrieve the latest index timeseries for the Pacific Decadal Oscillation (PDO)
    and convolve it with an exponential response function with timescale of 5 months."""

    print( 'get_pdo_timeseries' )
    time0 = time()

    pdo_file = "ersst.v5.pdo.dat"
    resp = requests.get( 'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat' )
    with open( pdo_file, 'wb' ) as f: 
        f.write( resp.content )

    with open( pdo_file, 'r' ) as f: 
        lines = f.readlines()

    lines = lines[2:]
    years = []
    months = []
    indices = []

    #  Parse file contents. 

    for line in lines: 
        ss = line.strip().split()
        year = int( ss[0] )
        indices += [ float(s) for s in ss[1:] ]
        years += [ year for i in range(12) ]
        months += [ i+1 for i in range(12) ]

    #  Convert to numpy arrays. 

    x = np.array( indices )
    indices = np.ma.masked_where( x == 99.99, x )
    years = np.array( years )
    months = np.array( months )

    #  Convolve with five-month response function. 

    rf = np.exp( -np.arange(48) / 5.0 )
    rf /= rf.sum()
    rindices = np.convolve( indices, rf )[:indices.size]

    #  Store in dictionary. 

    pdo = { 'years': years, 'months': months, 'indices': indices, 'rindices': rindices }
    pdo['times'] = np.array( [ year + (month-0.5)/12.0 for year, month in zip( years, months ) ] )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

    return pdo

def get_amo_timeseries(): 
    """Retrieve the latest index timeseries for the Atlantic Multidecadal Oscillation (AMO)
    and convolve it with an exponential response function with timescale of 5 months."""

    print( 'get_amo_timeseries' )
    time0 = time()

    amo_file = "csu_amo.csv"
    resp = requests.get( 'https://tropical.colostate.edu/Forecast/downloadable/csu_amo.csv' )
    with open( amo_file, 'wb' ) as f: 
        f.write( resp.content )

    with open( amo_file, 'r' ) as f: 
        lines = f.readlines()

    lines = lines[1:]
    years = []
    months = []
    indices = []

    #  Parse file contents. 

    for line in lines: 
        ss = line.strip().split( "," )
        year = int( ss[0] )
        lindices = [ float(s) for s in ss[1:] if re.search( r'[0-9.]+', s ) ]
        indices += lindices
        n = len( lindices )
        years += [ year for i in range(n) ]
        months += [ i+1 for i in range(n) ]

    #  Convert to numpy arrays. 

    x = np.array( indices )
    indices = np.ma.masked_where( x == 99.99, x )
    years = np.array( years )
    months = np.array( months )

    #  Convolve with five-month response function. 

    rf = np.exp( -np.arange(48) / 5.0 )
    rf /= rf.sum()
    rindices = np.convolve( indices, rf )[:indices.size]

    #  Store in dictionary. 

    amo = { 'years': years, 'months': months, 'indices': indices, 'rindices': rindices }
    amo['times'] = np.array( [ year + (month-0.5)/12.0 for year, month in zip( years, months ) ] )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

    return amo


def compute_ccmp_linear_regressions( outputfile="ccmp_trends.dat", jsonsavefile='ccmp_analyses.json', pdo=None, amo=None ): 
    """Open CCMP data files and perform linear regression by region, by season and all-months. 
    Results are stored locally in list **ccmp_analyses**. A truncated version is stored in 
    "ccmp_analyses.json". Also, easily read ASCII tables of regression results are saved in 
    "ccmp_trends.dat", "ccmp_pdo.dat", and "ccmp_amo.dat". """

    print( 'compute_ccmp_linear_regressions' )
    time0 = time()

    ldir = os.path.join( DATAROOT, "output" )
    files = [ os.path.join( ldir, f ) for f in os.listdir(ldir) \
            if re.search( r"^tropicalwidth.\d{4}.nc$", f ) ]
    files.sort()

    d = MFDataset( files, 'r' )
    years = d.variables['years'][:]
    months = d.variables['months'][:]
    times = years + (months-0.5)/12

    x = d.variables['methods'][:]
    nmethods = x.shape[0]
    methods = []
    for i in range(nmethods):
        m = np.logical_not( x[i,:].mask )
        methods.append( bytes( x[i,m] ).decode() )

    x = d.variables['regions'][:]
    nregions = x.shape[0]
    regions = []
    for i in range(nregions):
        m = np.logical_not( x[i,:].mask )
        regions.append( bytes( x[i,m] ).decode() )
    ccmp_regions = regions

    ccmp_yearrange = [ years.min(), years.max() ]

    print( '  methods = ' + ", ".join( methods ) )
    print( '  regions = ' + ", ".join( regions ) )

    ccmp_analyses = []

    for imethod, method in enumerate( methods ):
        for iregion, region in enumerate( regions ):
            lats = d.variables['lats'][:,iregion,imethod]

            #  Compute annual cycle. 

            annualcycle = np.zeros( 12, dtype='f' )
            for imonth in range(12):
                ii = ( months == imonth+1 )
                annualcycle[imonth] = lats[ii].mean()

            annualcycle -= annualcycle.mean()

            #  Compute departures from annual cycle. 

            dlats = lats - annualcycle[months-1]

            for season in seasons:

                print( f'  method={method}, region={region}, season={season["name"]}' )

                #  Select months by season. 

                ii0, ii1 = ( months >= season['monthrange'][0] ), ( months <= season['monthrange'][1] )
                if season['monthrange'][1] > season['monthrange'][0]:
                    ii = np.argwhere( np.logical_and( ii0, ii1 ) ).squeeze()
                else:
                    ii = np.argwhere( np.logical_or( ii0, ii1 ) ).squeeze()
                y = dlats[ii]
                x = times[ii]

                #  Linear regression. 

                ret = regress( x, y, pdo=pdo, amo=amo, noautocorrelation=(season != "All") )
                iintercept = ret['labels'].index("bias")
                itrend = ret['labels'].index("trend")

                #  Store results. 

                rec = {
                    'method': method,
                    'region': region,
                    'season': season['name'],
                    'lats': lats[ii],                 #  ...with annual cycle included...
                    'times': x,
                    'time_reference': ret['timereference'], 
                    'annualcycle': annualcycle,
                    'intercept_index': iintercept, 
                    'trend_index': itrend, 
                    'labels': ret['labels'], 
                    'coefficients': ret['coefficients'], 
                    'uncertainty_covariance': ret['uncertainty'], 
                    'intercept': ret['coefficients'][iintercept], 
                    'intercept_uncertainty': np.sqrt( ret['uncertainty'][iintercept,iintercept] ), 
                    'trend': ret['coefficients'][itrend],
                    'trend_uncertainty': np.sqrt( ret['uncertainty'][itrend,itrend] )
                }

                if pdo: 
                    i = ret['labels'].index("pdo")
                    rec.update( { 
                        'pdo_index': i, 
                        'pdo': ret['coefficients'][i],
                        'pdo_uncertainty': np.sqrt( ret['uncertainty'][i,i] ) 
                    } )

                if amo: 
                    i = ret['labels'].index("amo")
                    rec.update( { 
                        'amo_index': i, 
                        'amo': ret['coefficients'][i],
                        'amo_uncertainty': np.sqrt( ret['uncertainty'][i,i] ) 
                    } )

                ccmp_analyses.append( rec )

    d.close()

    out = []

    for a in ccmp_analyses: 

        rec = { 
                'method': a['method'],
                'region': a['region'],
                'season': a['season'],
                'intercept': float( a['intercept'] ), 
                'intercept_uncertainty': float( a['intercept_uncertainty'] ), 
                'trend': float( a['trend'] ),
                'trend_uncertainty': float( a['trend_uncertainty'] ) }

        if pdo: 
            rec.update( {
                'pdo': float( a['pdo'] ),
                'pdo_uncertainty': float( a['pdo_uncertainty'] ) }  )

        if amo: 
            rec.update( {
                'amo': float( a['amo'] ),
                'amo_uncertainty': float( a['amo_uncertainty'] ) }  )

        out.append( rec )

    print( f'  Saving data to {jsonsavefile}' )
    with open( jsonsavefile, 'w' ) as f:
        json.dump( out, f, indent='  ' )

    #  Write trends and their uncertainties in user-friendly table format. 

    lines = [ '{:12s}  {:^80s}'.format( "Region", "Seasonal trends (uncertainty) [deg/decade]" ) ]
    lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )

    for hemisphere in [ "N", "S" ]:
        for region in regions:
            if region[0] != hemisphere:
                continue
            line = f'{region:12s}'
            for season in seasons:
                rec = [ rec for rec in ccmp_analyses if rec['region']==region and
                    rec['season']==season['name'] and rec['method']=="All" ][0]
                line += f'  {rec["trend"]*10:6.3f} ({rec["trend_uncertainty"]*10:5.3f})'
            lines.append( line )

    #  To an ASCII output file. 

    print( f'  Saving CCMP trend data to {outputfile}.' )
    with open( outputfile, 'w' ) as f:
        f.write( "\n".join( lines ) + "\n" )

    #  Write PDO estimates and their uncertainties in user-friendly table format. 

    if pdo: 

        lines = [ '{:12s}  {:^80s}'.format( "Region", "Seasonal PDO (uncertainty) [degrees lat per PDO index]" ) ]
        lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )

        for hemisphere in [ "N", "S" ]:
            for region in regions:
                if region[0] != hemisphere:
                    continue
                line = f'{region:12s}'
                for season in seasons:
                    rec = [ rec for rec in ccmp_analyses if rec['region']==region and
                        rec['season']==season['name'] and rec['method']=="All" ][0]
                    line += f'  {rec["pdo"]:6.3f} ({rec["pdo_uncertainty"]:5.3f})'
                lines.append( line )

        #  To an ASCII output file. 

        file = "ccmp_pdo.dat"
        print( f'  Saving CCMP PDO estimates to {file}.' )
        with open( file, 'w' ) as f:
            f.write( "\n".join( lines ) + "\n" )

    #  Write AMO estimates and their uncertainties in user-friendly table format. 

    if amo: 

        lines = [ '{:12s}  {:^80s}'.format( "Region", "Seasonal AMO (uncertainty) [degrees lat per AMO index]" ) ]
        lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )

        for hemisphere in [ "N", "S" ]:
            for region in regions:
                if region[0] != hemisphere:
                    continue
                line = f'{region:12s}'
                for season in seasons:
                    rec = [ rec for rec in ccmp_analyses if rec['region']==region and
                        rec['season']==season['name'] and rec['method']=="All" ][0]
                    line += f'  {rec["amo"]:6.3f} ({rec["amo_uncertainty"]:5.3f})'
                lines.append( line )

        #  To an ASCII output file. 

        file = "ccmp_amo.dat"
        print( f'  Saving CCMP PDO estimates to {file}.' )
        with open( file, 'w' ) as f:
            f.write( "\n".join( lines ) + "\n" )

    #  Done. 

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

    return ccmp_analyses


## ERA5 linear regression

def compute_era5_linear_regression( outputfile='era5_trends.dat', pdo=None, amo=None ): 
    """Get ERA5 data. Results are stored locally in list era5_analyses. Also, easily 
    read ASCII tables of regression results are saved in "era5_trends.dat", "era5_pdo.dat", 
    and "era_amo.dat"."""

    print( 'compute_era5_linear_regression' )
    time0 = time()

    inputfile = ERA5ANALYSIS
    yearrange = [ 1995, 2024 ]

    print( f'  Reading {inputfile}' )
    e = Dataset( inputfile, 'r' )

    years = e.variables['years'][:]
    months = e.variables['months'][:]
    times = years + ( months - 0.5 )/12.0

    regions = [ "NH", "SH" ]
    nregions = len( regions )
    era5_regions = regions

    era5_analyses = []

    for iregion, region in enumerate( regions ): 
        if region == "NH": 
            lats = e.variables['latNH'][:]
        elif region == "SH": 
            lats = e.variables['latSH'][:]

        #  Compute annual cycle. 

        annualcycle = np.zeros( 12, dtype='f' )
        for imonth in range(12): 
            ii = np.logical_and( months == imonth+1, years >= yearrange[0], years <= yearrange[1] )
            annualcycle[imonth] = lats[ii].mean()

        annualcycle -= annualcycle.mean()

        #  Compute departures from annual cycle. 

        dlats = lats - annualcycle[months-1]

        for season in seasons: 

            print( f'  region={region}, season={season["name"]}' )

            #  Select months by season. 

            ii0 = np.logical_and( np.logical_and( months >= season['monthrange'][0], years >= yearrange[0] ), \
                    years <= yearrange[1] )
            ii1 = np.logical_and( np.logical_and( months <= season['monthrange'][1], years >= yearrange[0] ), \
                    years <= yearrange[1] )

            if season['monthrange'][1] > season['monthrange'][0]: 
                ii = np.argwhere( np.logical_and( ii0, ii1 ) ).squeeze()
            else: 
                ii = np.argwhere( np.logical_or( ii0, ii1 ) ).squeeze() 

            y = dlats[ii]
            x = np.ma.array( times[ii] )
            x.mask = y.mask

            #  Linear regression. 

            ret = regress( x, y, pdo=pdo, amo=amo, noautocorrelation=(season != "All") )
            iintercept = ret['labels'].index("bias")
            itrend = ret['labels'].index("trend")

            #  Store results. 

            rec = { 
                'region': region, 
                'season': season['name'], 
                'lats': lats[ii],              # ...including annual cycle...
                'annualcycle': annualcycle, 
                'times': times[ii], 
                'labels': ret['labels'], 
                'time_reference': ret['timereference'], 
                'intercept_index': iintercept, 
                'trend_index': itrend, 
                'coefficients': ret['coefficients'], 
                'uncertainty_covariance': ret['uncertainty'], 
                'intercept': ret['coefficients'][iintercept], 
                'intercept_uncertainty': np.sqrt( ret['uncertainty'][iintercept,iintercept] ), 
                'trend': ret['coefficients'][itrend],
                'trend_uncertainty': np.sqrt( ret['uncertainty'][itrend,itrend] )
                }

            if pdo: 
                i = ret['labels'].index( "pdo" )
                rec.update( { 
                    'pdo_index': i, 
                    'pdo': ret['coefficients'][i],
                    'pdo_uncertainty': np.sqrt( ret['uncertainty'][i,i] ) } )

            if amo: 
                i = ret['labels'].index( "amo" )
                rec.update( { 
                    'amo_index': i, 
                    'amo': ret['coefficients'][i],
                    'amo_uncertainty': np.sqrt( ret['uncertainty'][i,i] ) } )

            era5_analyses.append( rec )

    e.close()

    #  Save trends and their uncertainties in ASCII format table. 

    lines = [ '{:12s}  {:^80s}'.format( "Region", "Seasonal trends (uncertainty) [deg/decade]" ) ]
    lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )   

    for hemisphere in [ "N", "S" ]: 
        for region in regions: 
            if region[0] != hemisphere: 
                continue
            line = f'{region:12s}'
            for season in seasons: 
                rec = [ rec for rec in era5_analyses if rec['region']==region and 
                       rec['season']==season['name'] ][0]
                line += f'  {rec["trend"]*10:6.3f} ({rec["trend_uncertainty"]*10:5.3f})'
            lines.append( line )

    print( f'  Saving ERA5 trend data to {outputfile}.' )
    with open( outputfile, 'w' ) as f: 
        f.write( "\n".join( lines ) + "\n" )

    #  Write PDO estimates and their uncertainties in user-friendly table format. 

    if pdo: 

        lines = [ '{:12s}  {:^80s}'.format( "Region", "Seasonal PDO (uncertainty) [degrees lat per PDO index]" ) ]
        lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )

        for hemisphere in [ "N", "S" ]:
            for region in regions:
                if region[0] != hemisphere:
                    continue
                line = f'{region:12s}'
                for season in seasons:
                    rec = [ rec for rec in ccmp_analyses if rec['region']==region and
                           rec['season']==season['name'] and rec['method']=="All" ][0]
                    line += f'  {rec["pdo"]:6.3f} ({rec["pdo_uncertainty"]:5.3f})'
                lines.append( line )

        #  To an ASCII output file. 

        file = "era5_pdo.dat"
        print( f'  Saving ERA5 PDO estimates to {file}.' )
        with open( file, 'w' ) as f:
            f.write( "\n".join( lines ) + "\n" )

    #  Write AMO estimates and their uncertainties in user-friendly table format. 

    if amo: 

        lines = [ '{:12s}  {:^80s}'.format( "Region", "Seasonal AMO (uncertainty) [degrees lat per AMO index]" ) ]
        lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )

        for hemisphere in [ "N", "S" ]:
            for region in regions:
                if region[0] != hemisphere:
                    continue
                line = f'{region:12s}'
                for season in seasons:
                    rec = [ rec for rec in ccmp_analyses if rec['region']==region and
                           rec['season']==season['name'] and rec['method']=="All" ][0]
                    line += f'  {rec["amo"]:6.3f} ({rec["amo_uncertainty"]:5.3f})'
                lines.append( line )

        #  To an ASCII output file. 

        file = "era5_amo.dat"
        print( f'  Saving ERA5 PDO estimates to {file}.' )
        with open( file, 'w' ) as f:
            f.write( "\n".join( lines ) + "\n" )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

    return era5_analyses


### CMIP6 linear regression

def compute_cmip6_linear_regression( outputfile='cmip6_trends.dat', jsonsavefile="cmip6_analyses.json", pdo=None, amo=None ): 
    """Read in analyses of latitudes of streamfunction=0 latitudes analyzed from CMIP6 
    models. Results are stored locally in list cmip6_analyses. A truncated version is 
    stored in "cmip6_analyses.json". Also, easily read ASCII tables of regression results 
    are saved in "cmip6_trends.dat", "cmip6_pdo.dat", and "cmip6_amo.dat"."""

    print( 'compute_cmip6_linear_regression' )
    time0 = time()

    datafile = CMIP6ANALYSIS
    print( f'  Opening {datafile}' )

    d = Dataset( datafile, 'r' )

    #  Get regions. 

    x = d.variables['regions'][:]
    nregions = x.shape[0]
    regions = []
    for i in range(nregions): 
        m = np.logical_not( x[i,:].mask )
        regions.append( bytes( x[i,m] ).decode() )
    cmip6_regions = regions

    #  Define year range for linear regression. 

    cmip6_yearrange = [ 1985, 2014 ]

    #  Get unique models. 

    allgroups = []

    for groupname, group in d.groups.items(): 
        allgroups.append( { 
                'name': groupname, 
                'model': group.getncattr( "model" ), 
                'frequency': group.getncattr( "frequency" ), 
                'scenario': group.getncattr( "scenario" ), 
                'realization': group.getncattr( "realization" )
        } )

    models = sorted( list( set( [ g['model'] for g in allgroups ] ) ) )
    print( '  Unique models = ' + ", ".join( models ) )

    #  Loop over models. 

    cmip6_analyses = []

    for model in models: 

        print( f'  Analysis for model {model}' )
    
        #  Select a single realization. 

        gname = sorted( [ g['name'] for g in allgroups if g['model']==model ] )[0]
        g = d.groups[gname]

        for iregion, region in enumerate( regions ): 
            years = g.variables['years'][:]
            ii = np.argwhere( np.logical_and( years >= cmip6_yearrange[0], years <= cmip6_yearrange[1] ) ).squeeze()
            years, months = g.variables['years'][ii], g.variables['months'][ii]
            times = years + ( months - 0.5 ) / 12
            lats = g.variables['lats'][ii,iregion]
    
            #  Compute annual cycle. 

            annualcycle = np.zeros( 12, dtype='f' )
            for imonth in range(12): 
                ii = ( months == imonth+1 )
                annualcycle[imonth] = lats[ii].mean()

            annualcycle -= annualcycle.mean()
    
            #  Compute departures from annual cycle. 

            dlats = lats - annualcycle[months-1]
     
            for season in seasons: 

                #  Select months by season. 
        
                ii0, ii1 = ( months >= season['monthrange'][0] ), ( months <= season['monthrange'][1] )
                if season['monthrange'][1] > season['monthrange'][0]: 
                    ii = np.argwhere( np.logical_and( ii0, ii1 ) ).squeeze()
                else: 
                    ii = np.argwhere( np.logical_or( ii0, ii1 ) ).squeeze()
                y = dlats[ii]
                x = times[ii]

                #  Linear regression. 

                ret = regress( x, y, pdo=pdo, amo=amo, noautocorrelation=(season != "All") )
                iintercept = ret['labels'].index("bias")
                itrend = ret['labels'].index("trend")

                #  Store results. 

                rec = { 
                    'model': model, 
                    'region': region, 
                    'season': season['name'], 
                    'lats': lats[ii],                 #  ...with annual cycle included...
                    'times': x, 
                    'time_reference': ret['timereference'], 
                    'annualcycle': annualcycle, 
                    'intercept_index': iintercept, 
                    'trend_index': itrend, 
                    'coefficients': ret['coefficients'], 
                    'uncertainty_covariance': ret['uncertainty'], 
                    'intercept': ret['coefficients'][iintercept], 
                    'intercept_uncertainty': np.sqrt( ret['uncertainty'][iintercept,iintercept] ), 
                    'trend': ret['coefficients'][itrend],
                    'trend_uncertainty': np.sqrt( ret['uncertainty'][itrend,itrend] ) 
                } 

                if pdo: 
                    i = ret['labels'].index( "pdo" )
                    rec.update( { 
                        'pdo_index': i, 
                        'pdo': ret['coefficients'][i],
                        'pdo_uncertainty': np.sqrt( ret['uncertainty'][i,i] ) } )

                if amo: 
                    i = ret['labels'].index( "amo" )
                    rec.update( { 
                        'amo_index': i, 
                        'amo': ret['coefficients'][i],
                        'amo_uncertainty': np.sqrt( ret['uncertainty'][i,i] ) } )

                cmip6_analyses.append( rec )

    d.close()

    #  Saving computations to output JSON file. 

    out = []
    for a in cmip6_analyses: 
        rec = {
                'model': a['model'],
                'region': a['region'],
                'season': a['season'],
                'intercept': float( a['intercept'] ), 
                'intercept_uncertainty': float( a['intercept_uncertainty'] ), 
                'trend': float( a['trend'] ),
                'trend_uncertainty': float( a['trend_uncertainty'] ) } 

        if pdo: 
            rec.update( { 
                'pdo': float( a['pdo'] ),
                'pdo_uncertainty': float( a['pdo_uncertainty'] ) } )

        if amo: 
            rec.update( { 
                'amo': float( a['amo'] ),
                'amo_uncertainty': float( a['amo_uncertainty'] ) } )

        out.append( rec )

    print( f'  Saving data to {jsonsavefile}' )
    with open( jsonsavefile, 'w' ) as f: 
        json.dump( out, f, indent='  ' )
    
    #  Write in user-friendly table format. 

    lines = []

    for model in models: 
        lines.append( f'Model: {model}' )
        lines.append( '{:12s}  {:^80s}'.format( "Region", "Seasonal trends (uncertainty) [deg/decade]" ) )
        lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )

        for hemisphere in [ "N", "S" ]: 
            for region in regions: 
                if region[0] != hemisphere: 
                    continue
                line = f'{region:12s}'
                for season in seasons: 
                    rec = [ rec for rec in cmip6_analyses if rec['region']==region and 
                            rec['season']==season['name'] and rec['model']==model ][0]
                    line += f'  {rec["trend"]*10:6.3f} ({rec["trend_uncertainty"]*10:5.3f})'
                lines.append( line )
        lines.append( "" )

    #  To an ASCII output file. 

    print( f'  Saving CMIP6 model trend data to {outputfile}.' )
    with open( outputfile, 'w' ) as f: 
        f.write( "\n".join( lines ) + "\n" )

    #  Write PDO estimates and their uncertainties in user-friendly table format. 

    if pdo: 
    
        lines = []

        for model in models: 
            lines.append( f'Model: {model}' )
            lines.append( '{:12s}  {:^80s}'.format( "Region", "Seasonal PDO (uncertainty) [degrees lat per PDO index]" ) )
            lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )

            for hemisphere in [ "N", "S" ]: 
                for region in regions: 
                    if region[0] != hemisphere: 
                        continue
                    line = f'{region:12s}'
                    for season in seasons: 
                        rec = [ rec for rec in cmip6_analyses if rec['region']==region and 
                                rec['season']==season['name'] and rec['model']==model ][0]
                        line += f'  {rec["pdo"]:6.3f} ({rec["pdo_uncertainty"]:5.3f})'
                    lines.append( line )
            lines.append( "" )

        #  To an ASCII output file. 

        file = 'cmip6_pdo.dat'
        print( f'  Saving CMIP6 PDO estimates to {file}.' )
        with open( file, 'w' ) as f:
            f.write( "\n".join( lines ) + "\n" )

    #  Write AMO estimates and their uncertainties in user-friendly table format. 

    if amo: 
    
        lines = []

        for model in models: 
            lines.append( f'Model: {model}' )
            lines.append( '{:12s}  {:^80s}'.format( "Region", "Seasonal AMO (uncertainty) [degrees lat per AMO index]" ) )
            lines.append( " "*14 + "  ".join( [ f'{season["name"]:^14s}' for season in seasons ] ) )

            for hemisphere in [ "N", "S" ]: 
                for region in regions: 
                    if region[0] != hemisphere: 
                        continue
                    line = f'{region:12s}'
                    for season in seasons: 
                        rec = [ rec for rec in cmip6_analyses if rec['region']==region and 
                                rec['season']==season['name'] and rec['model']==model ][0]
                        line += f'  {rec["amo"]:6.3f} ({rec["amo_uncertainty"]:5.3f})'
                    lines.append( line )
            lines.append( "" )

        #  To an ASCII output file. 

        file = 'cmip6_amo.dat'
        print( f'  Saving CMIP6 PDO estimates to {file}.' )
        with open( file, 'w' ) as f:
            f.write( "\n".join( lines ) + "\n" )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

    return cmip6_analyses


### Analyze inter-monthly variability in zonal winds 

def compute_zonal_wind_variability(): 

    print( 'compute_zonal_wind_variability' )
    time0 = time()

    nzonal = 15
    first = True

    indexregions = [ 
        { 'name': "North Atlantic", 'longituderange': [306,336], 'latituderange': [24,36] }, 
        { 'name': "North Pacific", 'longituderange': [180,230], 'latituderange': [24,36] }, 
        { 'name': "Northern Subtropical Front", 'longituderange': [0,359.99], 'latituderange': [24,36] }, 
        { 'name': "South Atlantic", 'longituderange': [-30,6], 'latituderange': [-36,-24] }, 
        { 'name': "South Pacific", 'longituderange': [215,276], 'latituderange': [-36,-24] }, 
        { 'name': "Southern Subtropical Front", 'longituderange': [0,359.99], 'latituderange': [-36,-24] }, 
    ]

    print( "  Computations" )

    for year in range( 1995, 2025 ): 
        print( f'    Year {year}' )

        for imonth in range(12): 
            month = imonth + 1

            #  CCMP monthly mean zonal winds. 

            lpath = os.path.join( DATAROOT, "ccmp", f'Y{year:4d}', f'M{month:02d}', 
                            f'CCMP_Wind_Analysis_{year:4d}{month:02d}_monthly_mean_V03.1_L4.nc' )
            if not os.path.exists( lpath ): 
                rpath = os.path.join( "ccmp", f'Y{year:4d}', f'M{month:02d}', 
                            f'CCMP_Wind_Analysis_{year:4d}{month:02d}_monthly_mean_V03.1_L4.nc' )
                os.makedirs( os.path.dirname( lpath ), exist_ok=True )
                s3.download_file( bucket, rpath, lpath )

            #  Get CCMP u field. 

            d = Dataset( lpath, 'r' )
            ccmp_u = d.variables['u'][:].squeeze()
            if first: 
                ccmp_lons, ccmp_lats = d.variables['longitude'][:], d.variables['latitude'][:]
            d.close()

            #  OSCAR monthly mean surface currents. 

            lpath = os.path.join( DATAROOT, "oscar", f'{year:4d}', f'{month:02d}', f'oscar_currents_{year:4d}{month:02d}.nc4' )
            if not os.path.exists( lpath ): 
                rpath = os.path.join( "oscar", f'{year:4d}', f'{month:02d}', f'oscar_currents_{year:4d}{month:02d}.nc4' )
                os.makedirs( os.path.dirname( lpath ), exist_ok=True )
                s3.download_file( bucket, rpath, lpath )

            #  Get OSCAR u field. 

            d = Dataset( lpath, 'r' )
            oscar_u = d.variables['u'][:].squeeze().T
            if first: 
                oscar_lons, oscar_lats = d.variables['lon'][:], d.variables['lat'][:]
            d.close()

            #  Create output dictionary. 

            if first: 
                pps = { 'oscar_lons': oscar_lons, 'oscar_lats': oscar_lats, 
                        'mean_state': np.zeros( (12,oscar_lats.size,oscar_lons.size), np.float32 ), 
                        'spatial_variance': np.zeros( (12,oscar_lats.size,oscar_lons.size), np.float32 ), 
                        'nmonths': np.zeros( 12, np.int32 ) }

            #  Define index regions. 

            if first: 
                for r in indexregions: 
                    dlons = oscar_lons - r['longituderange'][0]
                    ii = dlons < 0.0
                    if ii.sum() > 0: dlons[ii] += 360
                    dlon = r['longituderange'][1] - r['longituderange'][0]
                    if dlon < 0: dlon += 360
                    ilons = np.argwhere( dlons <= dlon ).squeeze()
                    dlats = oscar_lats - r['latituderange'][0]
                    dlat = r['latituderange'][1] - r['latituderange'][0]
                    ilats = np.argwhere( dlats <= dlat ).squeeze()
                    dd = np.abs( oscar_lats[1] - oscar_lats[0] )
                    latweights = np.sin( np.deg2rad( oscar_lats[ilats] + 0.5*dd ) ) - np.sin( np.deg2rad( oscar_lats[ilats] - 0.5*dd ) )
                    latweights /= latweights.sum()
                    r.update( { 'ilons': ilons, 'ilats': ilats, 'latweights': latweights } )
                    r.update( { 'years': [], 'months': [], 'values': [] } )

            if first: 
                first = False 

            #  Stagger CCMP u-wind onto OSCAR lon/lat grid. 

            ccmp_ua = 0.5 * ( ccmp_u[:-1,:] + ccmp_u[1:,:] )
            ccmp_ub = np.zeros( (oscar_lats.size,oscar_lons.size), np.float32 )
            ccmp_ub[:,:-1] = 0.5 * ( ccmp_ua[:,:-1] + ccmp_ua[:,1:] )
            ccmp_ub[:,-1] = 0.5 * ( ccmp_ua[:,0] + ccmp_ua[:,-1] )

            #  Compose sum. 

            u = ccmp_ub * 1.0
            iocean = np.logical_not( oscar_u.mask ).squeeze() 
            u[iocean] = u[iocean] + oscar_u[iocean]

            #  Load into storage dictionary. 

            pps['mean_state'][imonth,:,:] += u
            pps['spatial_variance'][imonth,:,:] += u**2
            pps['nmonths'][imonth] += 1

            #  Compute regional averages. 

            for r in indexregions: 
                r['years'].append( year )
                r['months'].append( month )
                r['values'].append( ( u[:,r['ilons']].mean(axis=1)[r['ilats']] * r['latweights'] ).sum() )

    #  Normalize. 

    print( '  Normalizing timeseries' )
    for imonth in range(12): 
        pps['mean_state'][imonth,:,:] /= pps['nmonths'][imonth]
        pps['spatial_variance'][imonth,:,:] /= pps['nmonths'][imonth]
        pps['spatial_variance'][imonth,:,:] -= pps['mean_state'][imonth,:,:]**2

    print( '  Defining times' )
    for r in indexregions: 
        r['years'] = np.array( r['years'] )
        r['months'] = np.array( r['months'] )
        nyears = int( len( r['values'] ) / 12 )
        x = np.array( r['values'], dtype=np.float32 ).reshape( nyears, 12 )
        r['values'] = ( x - x.mean( axis=0 ) )

    print( '  Data stored in pps and indexregions' )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

    return pps, indexregions


### Regression of index regional winds onto 500 hPa geopotential

def compute_indexregion_regressions( indexregions, era5file="geopotential.500hPa.monthly.1995-2024.nc" ): 
    """The index regional zonal wind timeseries is regressed onto the 500 hPa 
    geopotential field monthly averages as obtained from ERA5. 

    Arguments

    indexregions        computed by compute_zonal_wind_variability
    era5file            the input file containing a timeseries of the 500-hPa geopotential of ERA5.
    """

    print( 'compute_indexregion_regressions' )
    time0 = time()

    lpath = os.path.join( DATAROOT, "era5", era5file )

    if not os.path.exists( lpath ): 
        rpath = f'era5/{era5file}'
        os.makedirs( os.path.dirname( lpath ), exist_ok=True )
        s3.download_file( bucket, rpath, lpath )

    #  Open file, get lons and lats, interpret times. 

    d = Dataset( lpath, 'r' )

    era5_lons = d.variables['longitude'][:]
    era5_lats = d.variables['latitude'][:]

    v = d.variables['valid_time']
    att = v.getncattr( "units" )
    m = re.search( r'(\d{4}-\d{2}-\d{2})', att )
    t0 = datetime.strptime( m.group(1), '%Y-%m-%d' )
    dts = [ t0 + timedelta(seconds=int(s)) for s in v[:] ]
    era5_years = np.array( [ t.year for t in dts ], dtype=np.int32 )
    era5_months = np.array( [ t.month for t in dts ], dtype=np.int32 )
    ntimes = era5_years.size 
    nyears = int( ntimes / 12 )

    #  Read in geopotential field. Compute mean annual cycle. 

    z = d.variables['z'][:].reshape( nyears, 12, era5_lats.size, era5_lons.size )
    za = z.mean(axis=0)
    d.close()

    #  Remove annual cycle from timeseries. 

    zd = z.transpose( (0,2,3,1) ) - za.transpose( (1,2,0) ).reshape( era5_lats.size, era5_lons.size, 12 )
    zd = zd.transpose( (1,2,0,3) ).reshape( era5_lats.size, era5_lons.size, ntimes )

    #  Regression analysis. 

    print( '  Creating era5_indexregion_regressions' )
    era5_indexregion_regressions = []

    for r in indexregions: 

        print( "    " + r['name'] )

        #  Matchup times. 

        i0 = [ i for i in range(ntimes) if era5_years[i]==r['years'][0] and era5_months[i]==r['months'][0] ][0]
        i1 = [ i for i in range(ntimes) if era5_years[i]==r['years'][-1] and era5_months[i]==r['months'][-1] ][0]
        itimes = np.arange( i0, i1+1, dtype=np.int32 )

        #  Regression. 

        x = r['values'].flatten()
        y = (zd[:,:,itimes] * x ).sum(axis=2) / np.sqrt( ( x**2 ).sum() )
        era5_indexregion_regressions.append( { 
                'name': r['name'], 
                'longituderange': r['longituderange'], 
                'latituderange': r['latituderange'], 
                'lons': era5_lons, 
                'lats': era5_lats, 
                'regression': y } )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

    return era5_indexregion_regressions


### Trend in total width of Tropics
 
def compute_total_width_trend( ccmp_analyses, pdo=None ): 
    """Compute the trend in the total width of the Tropics based on CCMP."""

    print( 'compute_total_width_trend' )
    time0 = time()

    #  Extract northern hemisphere and southern hemisphere. 

    NH = [ a for a in ccmp_analyses if a['method']=="All" and a['region']=="NH" and a['season']=="All" ][0]
    SH = [ a for a in ccmp_analyses if a['method']=="All" and a['region']=="SH" and a['season']=="All" ][0]

    #  Timeseries of width in degrees. 

    y = NH['lats'] - SH['lats']
    y = y.reshape( int(y.size/12), 12 ) 

    x = NH['times']
    dx = x - x.mean()

    #  Remove annual cycle. 

    annualcycle = NH['annualcycle'] - SH['annualcycle'] 
    y = ( y - annualcycle ).reshape( y.size )

    #  Linear regression. 

    ret = regress( x, y, pdo=pdo, amo=amo )
    i = ret['labels'].index( "trend" )
    trend = ret['coefficients'][i]
    trend_uncertainty = np.sqrt( ret['uncertainty'][i,i] )

    print( f'  Trend in width of Tropics, all seasons, global: {trend*10:5.3f} ({trend_uncertainty*10:5.3f}) deg/decade' )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

    return trend, trend_uncertainty


## Generate figures

### PDO timeseries
 
def plot_pdo_timeseries( pdo, outputfile="pdo_timeseries.pdf" ): 
    """This figure is a timeseries of the raw PDO index and the same index subjected to an 
    exponential convolution of the same timeseres contained as "rindices" in pdo."""

    print( 'plot_pdo_timeseries' )
    time0 = time()

    fig = plt.figure( figsize=(6,2) )
    ax = fig.add_axes( [0.08,0.19,0.91,0.78] )

    ax.set_xlim( 1900, 2025 )
    ax.set_xticks( np.arange(1900, 2025, 20 ) )
    ax.xaxis.set_minor_locator( MultipleLocator(5) )
    ax.set_xlabel( "Year" )

    ax.set_ylim( -4, 4.01 )
    ax.set_yticks( np.arange( -4.0, 4.01, 2 ) )
    ax.yaxis.set_minor_locator( MultipleLocator(0.5) )
    ax.set_ylabel( "PDO Index" )

    ax.grid( color="#C0C0C0" )
    ax.fill_between( [1995,2025], [-4,-4], [4,4], color="#E0E0E0" )

    ax.plot( pdo['times'], pdo['indices'], color="#000000", lw=0.5, label="Principal component" )
    ax.plot( pdo['times'], pdo['rindices'], color="#FF2020", lw=1.5, label="PDO index" )

    ax.legend( loc="upper right" )

    print( f'  Generating {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )

### AMO timeseries
 
def plot_amo_timeseries( amo, outputfile="amo_timeseries.pdf" ): 
    """This figure is a timeseries of the raw AMO index and the same index subjected to an 
    exponential convolution of the same timeseres contained as "rindices" in amo."""

    print( 'plot_amo_timeseries' )
    time0 = time()

    fig = plt.figure( figsize=(6,2) )
    ax = fig.add_axes( [0.08,0.19,0.91,0.78] )

    ax.set_xlim( 1940, 2025 )
    ax.set_xticks( np.arange(1940, 2025, 20 ) )
    ax.xaxis.set_minor_locator( MultipleLocator(5) )
    ax.set_xlabel( "Year" )

    ax.set_ylim( -4, 4.01 )
    ax.set_yticks( np.arange( -4.0, 4.01, 2 ) )
    ax.yaxis.set_minor_locator( MultipleLocator(0.5) )
    ax.set_ylabel( "AMO Index" )

    ax.grid( color="#C0C0C0" )
    ax.fill_between( [1995,2025], [-4,-4], [4,4], color="#E0E0E0" )

    ax.plot( amo['times'], amo['indices'], color="#000000", lw=0.5, label="Principal component" )
    ax.plot( amo['times'], amo['rindices'], color="#FF2020", lw=1.5, label="AMO index" )

    ax.legend( loc="lower left" )

    print( f'  Generating {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Timeseries of u=0 latitude

def plot_unull_timeseries( ccmp_analyses, outputfile="ccmp_tropicalwidth_timeseries.pdf" ): 
    """Unadulterated version of all-season, global determination of edge of the 
    Tropics, defined by the _u=0_ latitude."""

    print( 'plot_unull_timeseries' )
    time0 = time()

    Nregion, Sregion = "N Oceans", "S Oceans"

    fig = plt.figure( figsize=(6,4) )

    #  Northern Hemisphere. 

    yearrange = [ int( ccmp_analyses[0]['times'].min() ), int( ccmp_analyses[0]['times'].max() ) ]
    xticks = np.arange( yearrange[0], yearrange[1]+1.01, dtype='i' )
    imajor = ( xticks/4.0 == np.int32( xticks/4.0 ) )

    ax = fig.add_axes( [0.07,0.57,0.91,0.40] )
    ax.set_xlim( yearrange[0]-1, yearrange[1]+2 )
    ax.set_xticks( xticks[imajor] )
    ax.set_xticklabels( [] )

    ax.set_ylim( 20, 45 )
    yticks = np.arange(20,46,5).astype('i')
    ax.set_yticks( yticks )
    ax.set_yticklabels( [ r'{:2d}$^\circ$N'.format(ytick) for ytick in yticks ] )
    ax.set_ylabel( '' )
    ax.grid( which='both', axis='both', color='#808080', linestyle='--', lw=0.1 )

    methods = [ rec['method'] for rec in ccmp_analyses if rec['region']==Nregion and rec['season']=="All" ]

    region = Nregion
    for imethod, method in enumerate(methods): 
        rec = [ rec for rec in ccmp_analyses if rec['region']==region and rec['method']==method and rec['season']=="All" ][0]
        ax.plot( rec['times'], rec['lats'], label=method, linewidth=0.5 )

    #  Southern Hemisphere. 

    ax = fig.add_axes( [0.07,0.08,0.91,0.40] )
    ax.set_xlim( yearrange[0]-1, yearrange[1]+2 )
    ax.set_xticks( xticks[imajor] )
    ax.set_xticklabels( [ str(xtick) for xtick in xticks[imajor] ] )
    # ax.set_xticks( xticks, minor=True )
    # ax.set_xticklabels( [] )

    ax.set_ylim( -45, -20 )
    yticks = np.arange(-45,-19,5).astype('i')
    ax.set_yticks( yticks )
    ax.set_yticklabels( [ r'{:2d}$^\circ$S'.format(np.abs(ytick)) for ytick in yticks ] )
    ax.set_ylabel( '' )
    ax.grid( which='both', axis='both', color='#808080', linestyle='--', lw=0.1 )

    region = Sregion
    for imethod, method in enumerate(methods): 
        rec = [ rec for rec in ccmp_analyses if rec['region']==region and rec['method']==method and rec['season']=="All" ][0]
        ax.plot( rec['times'], rec['lats'], label=method, linewidth=0.5 )

    ax.legend( ncol=2, loc='lower center' )

    #  RMS difference, with and without data, for each hemisphere. 

    rec0 = [ rec for rec in ccmp_analyses if rec['region']==Nregion and rec['method']=="Dataonly" ][0]
    rec1 = [ rec for rec in ccmp_analyses if rec['region']==Nregion and rec['method']=="All" ][0]

    diff = rec0['lats'] - rec1['lats']
    bias = diff.mean()
    rms = np.sqrt( ( (diff-bias)**2 ).sum() / (diff.size-1) )
    print( f'  {Nregion} bias = {bias:5.2f} degs, {Nregion} rms = {rms:5.3f} degs' )

    rec0 = [ rec for rec in ccmp_analyses if rec['region']==Sregion and rec['method']=="Dataonly" and rec['season']=="All" ][0]
    rec1 = [ rec for rec in ccmp_analyses if rec['region']==Sregion and rec['method']=="All" and rec['season'] == "All" ][0]

    diff = rec0['lats'] - rec1['lats']
    bias = diff.mean()
    rms = np.sqrt( ( (diff-bias)**2 ).sum() / (diff.size-1) )
    print( f'  {Sregion} bias = {bias:5.2f} degs, {Sregion} rms = {rms:5.3f} degs' )

    #  Close. 

    print( "  Generating " + outputfile )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Trends by season

def plot_trends_by_season( ccmp_analyses, outputfile="ccmp_seasonal_timeseries.pdf" ): 
    """Plot of timeseries of tropical edges, north and south, according to season."""

    print( 'plot_trends_by_season' )
    time0 = time()

    fig = plt.figure( figsize=(6,4) )

    method = "All"
    analyses = ccmp_analyses

    yearrange = [ int( ccmp_analyses[0]['times'].min() ), int( ccmp_analyses[0]['times'].max() ) ]

    xlim = [ yearrange[0]-1, yearrange[1]+2 ]
    years = np.arange( yearrange[0], yearrange[1]+0.1, dtype=np.int32 )
    xticks = np.arange( min( [ y for y in years if y % 4 == 0 ] ), years.max()+0.01, 4, dtype='i' )
    xminor = years

    cmap = plt.get_cmap( 'tab20' )

    #  Loop over hemispheres. 

    for ihemisphere, hemisphere in enumerate( [ "N", "S" ] ): 

        ax = fig.add_axes( [0.07,0.57-ihemisphere*0.49,0.81,0.40] )
        region = f'{hemisphere}H'

        ax.set_xlim( *xlim )
        ax.set_xticks( xticks )
        ax.set_xticklabels( [ str(xtick) for xtick in xticks ] )
        ax.xaxis.set_minor_locator( MultipleLocator(1) )

        if hemisphere == "N": 
            ax.set_ylim( 20, 45 )
            yticks = np.arange(20,46,5).astype('i')
            ax.set_xticklabels( [] )
        elif hemisphere == "S": 
            ax.set_ylim( -45, -20 )
            yticks = np.arange(-45,-19,5).astype('i')

        ax.set_yticks( yticks )
        ax.set_yticklabels( [ r'{:2d}$^\circ${:}'.format(np.abs(ytick),hemisphere) for ytick in yticks ] )
        ax.set_ylabel( '' )
        ax.grid( which='both', axis='both', color='#808080', linestyle='--', lw=0.1 )

        for iseason, season in enumerate(seasons):           
            if season['name'] == "All": 
                continue
            rec = [ rec for rec in analyses if rec['region']==region and rec['method']==method \
                        and rec['season']==season['name'] ][0]
            color = cmap( ( iseason + 0.5 ) / 20.0 )
            imonths = np.int32( rec['times'] * 12 ) - np.int32( rec['times'] ) * 12

            #  Calculate seasonal average. 

            ii = np.arange( 12, dtype='i' )
            ii1, ii2 = ( ii+1 >= season['monthrange'][0] ), ( ii+1 <= season['monthrange'][1] )
            if season['monthrange'][0] <= season['monthrange'][1]: 
                ii = np.argwhere( np.logical_and(ii1,ii2) ).squeeze()
            else: 
                ii = np.argwhere( np.logical_or(ii1,ii2) ).squeeze()

            seasonal_anomaly = rec['annualcycle'][ii].mean() - rec['annualcycle'].mean()
            clats = rec['lats'] - rec['annualcycle'][imonths] + seasonal_anomaly
            lw = 1.0
            ax.scatter( rec['times'], clats, label=f'{season["name"]}', color=color, s=0.2 )

        ax.legend( ncol=1, bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small" )

    #  Close. 

    print( "  Generating " + outputfile )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Zonal mean zonal wind climatology
 
def plot_wind_climatology( outputfile="ccmp_zonal_mean_zonal_wind.pdf" ): 
    """Plot long-term zonal mean zonal surface air wind as a function of latitude, average over 1995-2024."""

    print( 'plot_wind_climatology' )
    time0 = time()

    # First, download needed OSCAR files. 

    for year in range( 1995, 2025 ): 
        for month in range(1,13): 
            rpath = f'oscar/{year:4d}/{month:02d}/oscar_currents_{year:4d}{month:02d}.nc4'
            lpath = f'{DATAROOT}/oscar/{year:4d}/{month:02d}/oscar_currents_{year:4d}{month:02d}.nc4'
            if not os.path.exists( lpath ): 
                print( f'  Downloading {lpath}' )
                os.makedirs( os.path.dirname( lpath ), exist_ok=True )
                s3.download_file( bucket, rpath, lpath )

    # Create plot. 

    m = Model( model='monthly', dataroot=DATAROOT )
    ccmp_yearrange = [ 1995, 2024 ]
    timerange = [ datetime.fromisoformat( f'{ccmp_yearrange[0]:4d}-01-01'), \
                datetime.fromisoformat( f'{ccmp_yearrange[1]:4d}-12-31' ) ]
    lats, ubar = averageWind( timerange, m )

    #  Set up plot. 

    fig = plt.figure( figsize=(4,2.5) )

    ax = fig.add_axes( [0.15,0.12,0.80,0.85] )
    ax.set_xlim( -90, 90 )
    ax.set_xticks( np.arange( -90, 91, 30 ) )
    ax.set_xticklabels( [ r'90$^\circ$S', r'60$^\circ$S', r'30$^\circ$S', r'Eq', r'30$^\circ$N', r'60$^\circ$N', r'90$^\circ$N' ] )
    ax.xaxis.set_minor_locator( MultipleLocator(10) )
    ax.invert_xaxis()
    ax.set_xlabel( '' )

    ax.set_ylim( -10, 10 )
    yticks = np.arange( -10, 11, 5, dtype='i' )
    ax.set_yticks( yticks )
    ax.set_yticklabels( [ str(ytick) for ytick in yticks ] )
    ax.yaxis.set_minor_locator( MultipleLocator(1) )
    ax.set_ylabel( 'u [m s$^{-1}$]' )

    ax.plot( [-90,90], [0,0], 'k--', lw=0.2 )
    for season in [ 'DJF', 'JJA' ]: 
        u = np.ma.masked_where( ubar[season]==0.0, ubar[season] )
        ax.plot( lats, u, lw=1, label=season )

    ax.legend( fontsize='small' )

    print( "  Generating " + outputfile )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### CCMP and ERA5 timeseries, full globe

def plot_ccmp_era5_timeseries( ccmp_analyses, era5_analyses, outputfile="ccmp_era5_timeseries.pdf" ): 

    print( 'plot_ccmp_era5_timeseries' )
    time0 = time()

    fig = plt.figure( figsize=(6,4) )

    #  Northern Hemisphere. 

    region = "NH"

    yearrange = int( ccmp_analyses[0]['times'].min() ), int( ccmp_analyses[0]['times'].max() )
    xticks = np.arange( yearrange[0], yearrange[1]+1.01 )
    imajor = ( xticks/4.0 == np.int32( xticks/4.0 ) )

    ax = fig.add_axes( [0.07,0.57,0.91,0.40] )
    ax.set_xlim( yearrange[0]-1, yearrange[1]+2 )
    ax.set_xticks( xticks[imajor] )
    ax.set_xticklabels( [] )

    ax.set_ylim( 20, 45 )
    yticks = np.arange(20,46,5).astype('i')
    ax.set_yticks( yticks )
    ax.set_yticklabels( [ r'{:2d}$^\circ$N'.format(ytick) for ytick in yticks ] )
    ax.set_ylabel( '' )
    ax.grid( which='both', axis='both', color='#808080', linestyle='--', lw=0.1 )

    ccmp = [ rec for rec in ccmp_analyses if rec['region']==region and rec['method']=="All" and rec['season']=="All" ][0]
    era5 = [ rec for rec in era5_analyses if rec['region']==region and rec['season']=="All" ][0]

    ax.plot( ccmp['times'], ccmp['lats'], label="CCMP", linewidth=1.0 )
    ax.plot( era5['times'], era5['lats'], label="ERA5", linewidth=1.0 )

    #  Southern Hemisphere. 

    region = "SH"

    ax = fig.add_axes( [0.07,0.08,0.91,0.40] )
    ax.set_xlim( yearrange[0]-1, yearrange[1]+2 )
    ax.set_xticks( xticks[imajor] )
    # ax.set_xticks( xticks, minor=True )
    ax.set_xticklabels( [ str(int(xtick)) for xtick in xticks[imajor] ] )

    ax.set_ylim( -45, -20 )
    yticks = np.arange(-45,-19,5).astype('i')
    ax.set_yticks( yticks )
    ax.set_yticklabels( [ r'{:2d}$^\circ$S'.format(np.abs(ytick)) for ytick in yticks ] )
    ax.set_ylabel( '' )
    ax.grid( which='both', axis='both', color='#808080', linestyle='--', lw=0.1 )

    ccmp = [ rec for rec in ccmp_analyses if rec['region']==region and rec['method']=="All" and rec['season']=="All" ][0]
    era5 = [ rec for rec in era5_analyses if rec['region']==region and rec['season']=="All" ][0]

    ax.plot( ccmp['times'], ccmp['lats'], label="CCMP", linewidth=1.0 )
    ax.plot( era5['times'], era5['lats'], label="ERA5", linewidth=1.0 )

    ax.legend( ncol=2, loc='lower center' )

    print( f'  Saving to {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### CCMP and ERA5 timeseries, full globe, deseasonalized

def plot_ccmp_era5_timeseries_deseasonalized( ccmp_analyses, era5_analyses, outputfile="ccmp_era5_deseasonalize_timeseries.pdf", pdo=None ): 
    """Also include seasonal cycle and linear regression envelopes for CCMP timeseries."""

    print( 'plot_ccmp_era5_timeseries_deseasonalized' )
    time0 = time()

    fig = plt.figure( figsize=(6,3) )
    colors = { 'ccmp': "#0040FF", 'era5': "#40FF00" }
    yearrange = [ 1995, 2024 ]
    regionsuffix = "H"

    for ihemisphere, hemisphere in enumerate( [ "N", "S" ] ): 

        regionname = f'{hemisphere}{regionsuffix}'
        ccmp_rec = [ rec for rec in ccmp_analyses if rec['region']==regionname and rec['method']=="All" ][0]
        if regionname in [ rec['region'] for rec in era5_analyses ]: 
            era5_rec = [ rec for rec in era5_analyses if rec['region']==regionname ][0]
        else: 
            era5_rec = None
        ax = fig.add_axes( [0.07,0.57-ihemisphere*0.49,0.60,0.40] )

        xticks = np.arange( yearrange[0], yearrange[1]+1.1, 5, dtype='i' )
        ax.set_xlim( yearrange[0], yearrange[1]+1 )
        ax.set_xticks( xticks )
        ax.xaxis.set_minor_locator( MultipleLocator(1) )
        ax.yaxis.set_minor_locator( MultipleLocator(1) )

        if hemisphere == "N": 
            ax.set_xticklabels( [] )
            ax.set_ylim( 25, 40 )
            yticks = np.arange(25,41,5).astype('i')
        else: 
            ax.set_xticklabels( [ str(xtick) for xtick in xticks ] )
            ax.set_ylim( -40, -25 )
            yticks = np.arange(-40,-24,5).astype('i')
        
        ax.set_yticks( yticks )
        ax.set_yticklabels( [ r'{:2d}$^\circ${:}'.format(np.abs(ytick),hemisphere) for ytick in yticks ] )
        ax.set_ylabel( '' )
        ax.grid( which='both', axis='both', color='#808080', linestyle='--', lw=0.1 )

        for model, analysis in zip( ["ccmp","era5"], [ccmp_rec,era5_rec] ): 
            if analysis is None: continue
            times = analysis['times']
            imonths = np.int32( times * 12 ) - np.int32( times ) * 12
            clats = analysis['lats'] * 1.0
            for imonth in range(12): 
                ii = np.argwhere( imonths == imonth ).squeeze()
                clats[ii] -= analysis['annualcycle'][imonth]
            if model == "ccmp": 
                label = r'$\bar{u}(\phi,\ \mathrm{surface})=0$ (CCMP)'
            elif model == "era5": 
                label = r'$\psi(\phi,500\ \mathrm{hPa})=0$ (ERA5)'
            ax.plot( times, clats, lw=1.0, color=colors[model], label=label )

        #  Produce the fit envelope. 

        ret = expand( ccmp_rec['times'], ccmp_rec['coefficients'], ccmp_rec['uncertainty_covariance'], 
                     ccmp_rec['time_reference'], pdo=pdo )
        ax.fill_between( ccmp_rec['times'], ret['fit']-ret['ebars'], ret['fit']+ret['ebars'], color="#C0C0C0" )
    
        if ihemisphere == 1: 
            ax.legend( loc="lower left", fontsize="small", ncols=1 )
        
        #  Annual cycle. 
    
        ax = fig.add_axes( [0.75,0.57-ihemisphere*0.49,0.23,0.40] )

        ax.set_xlim( -0.5, 11.5 )
        ax.set_xticks( np.arange(0,12,3,dtype='i') )
        ax.xaxis.set_minor_locator( MultipleLocator(1) )
        if ihemisphere == 0: 
            ax.set_xticklabels( [] )
        else: 
            ax.set_xticklabels( "Jan Apr Jul Oct".split() )

        yticks = np.arange(-10,10.1,5).astype('i')
        ax.set_ylim( -10, 10 )
        ax.set_yticks( yticks )
        ax.set_yticklabels( [ r'{:3d}$^\circ$'.format(ytick) for ytick in yticks ] )
        ax.yaxis.set_minor_locator( MultipleLocator(1) )
        ax.set_ylabel( '' )

        ax.plot( [-1,13], [0,0], ls="--", lw=0.1, color="#404040" )

        for model, analysis in zip( ["ccmp","era5"], [ccmp_rec,era5_rec] ): 
            if analysis is None: continue
            cycle = analysis['annualcycle']
            ax.plot( cycle-cycle.mean(), color=colors[model], label=model.upper() )

    print( f'  Saving to {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Table of trends

def plot_table_of_trends( ccmp_analyses, cmip6_analyses, outputfile="table_of_trends.pdf" ): 
    """Display a graphical version of the table of trend analyses, CCMP and CMIP6, 
    northern and southern hemispheres, by season and region."""

    print( 'plot_table_of_trends' )
    time0 = time()

    fig = plt.figure( figsize=(6.5,4.0) )

    suffixes = [ "H", "Pacific", "Atlantic" ]
    hemispheres = [ "N", "S" ]
    method = "All" 

    xlim = [ -0.5, len(seasons)-0.5 ]
    xticks = np.arange( len(seasons), dtype='i' )

    ylim = [ -1.8, 1.8 ]
    yticks = np.arange( -1.0, 1.001, 1, dtype='f' )
    yminor = 0.2

    superframe = np.array( [ 0.065, 0.35, 0.925, 0.63 ] )
    subframe = np.array( [ 0.01, 0.01, 0.98, 0.98 ] )

    models = sorted( list( set( [ rec['model'] for rec in cmip6_analyses ] ) ) )
    nmodels = len( models )
    dx = 0.6
    cmap = plt.get_cmap( 'gist_ncar' )
    colors = [ cmap( 0.1 + 0.9*(imodel+0.5)/nmodels ) for imodel in range(nmodels) ]

    nx, ny = len(suffixes), len(hemispheres)

    for ihemisphere, hemisphere in enumerate( hemispheres ): 
        for isuffix, suffix in enumerate( suffixes ): 

            pos = ( subframe + np.array( [ isuffix, ny-ihemisphere-1, 0, 0 ] ) ) / np.array( [ nx, ny, nx, ny ] )
            pos = np.array([superframe[0],superframe[1],0,0]) + \
                    np.array([superframe[2],superframe[3],superframe[2],superframe[3]]) * pos
            ax = fig.add_axes( pos )

            ax.set_xlim( *xlim )
            ax.set_ylim( *ylim )

            for ytick in yticks: 
                ax.plot( xlim, [ytick,ytick], color="#808080", lw=0.5, ls='--' )

            #  Define region. 

            if suffix == "H": 
                region = f'{hemisphere}{suffix}'
            else: 
                region = f'{hemisphere} {suffix}'

            ax.text( -0.2, ylim[0]+(ylim[1]-ylim[0])*0.9, region, fontsize="small", color="#0000C0" )

            if isuffix == 0: 
                ax.set_yticks( yticks )
                ax.set_yticklabels( [ str(int(ytick)) for ytick in yticks ] )
                ax.yaxis.set_minor_locator( MultipleLocator(yminor) )
                ax.set_ylabel( r'Trend [$^\circ$ dec$^{-1}$]' )
            else: 
                ax.set_yticks( [] )

            if ihemisphere == 0: 
                ax.set_xticks( [] )
            else: 
                ax.set_xticks( xticks )
                ax.set_xticklabels( [ season['name'] for season in seasons ], rotation=-70 )

            #  Plot bar for each season in CCMP analyses. 

            for iseason, season in enumerate(seasons): 
                rec = [ rec for rec in ccmp_analyses if rec['method']==method \
                        and rec['season']==season['name'] and rec['region']==region ][0]
                x = iseason * np.array([1,1]) - 0.5 * dx
                y = rec['trend'] + rec['trend_uncertainty'] * np.array([-1,1])
                ax.plot( x, y*10, lw=2, color='k' )

            #  Plot bar for each season in CMIP6 analyses. 

            for imodel, model in enumerate( models ): 
                for iseason, season in enumerate(seasons): 
                    rec = [ rec for rec in cmip6_analyses if rec['model']==model \
                            and rec['season']==season['name'] and rec['region']==region ][0]
                    x = iseason * np.array([1,1]) + ( (imodel+1)/nmodels - 0.5 ) * dx
                    y = rec['trend'] + rec['trend_uncertainty'] * np.array([-1,1])
                    if iseason==0: 
                        ax.plot( x, y*10, lw=1, color=colors[imodel], label=model )                    
                    else: 
                        ax.plot( x, y*10, lw=1, color=colors[imodel] )
            if ihemisphere==1 and isuffix==1: 
                ax.legend( ncol=5, bbox_to_anchor=(0.5,-0.35), loc="upper center", fontsize=6, frameon=False )

    print( f'  Saving to {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Table of PDO
 
def plot_table_of_pdo( ccmp_analyses, cmip6_analyses, outputfile="table_of_pdo.pdf" ): 
    """Display a graphical version of the table of PDO analyses, CCMP and CMIP6, northern 
    and southern hemispheres, by season and region."""

    print( 'plot_table_of_pdo' )
    time0 = time()

    suffixes = [ "H", "Pacific", "Atlantic" ]
    hemispheres = [ "N", "S" ]
    method = "All" 

    xlim = [ -0.5, len(seasons)-0.5 ]
    xticks = np.arange( len(seasons), dtype='i' )

    ylim = [ -1.8, 1.8 ]
    yticks = np.arange( -1.0, 1.001, 1, dtype='f' )
    yminor = 0.2

    superframe = np.array( [ 0.065, 0.35, 0.925, 0.63 ] )
    subframe = np.array( [ 0.01, 0.01, 0.98, 0.98 ] )

    models = sorted( list( set( [ rec['model'] for rec in cmip6_analyses ] ) ) )
    nmodels = len( models )
    dx = 0.6
    cmap = plt.get_cmap( 'gist_ncar' )
    colors = [ cmap( 0.1 + 0.9*(imodel+0.5)/nmodels ) for imodel in range(nmodels) ]

    nx, ny = len(suffixes), len(hemispheres)

    fig = plt.figure( figsize=(6.5,4.0) )

    for ihemisphere, hemisphere in enumerate( hemispheres ): 
        for isuffix, suffix in enumerate( suffixes ): 

            pos = ( subframe + np.array( [ isuffix, ny-ihemisphere-1, 0, 0 ] ) ) / np.array( [ nx, ny, nx, ny ] )
            pos = np.array([superframe[0],superframe[1],0,0]) + np.array([superframe[2],superframe[3],superframe[2],superframe[3]]) * pos
            ax = fig.add_axes( pos )

            ax.set_xlim( *xlim )
            ax.set_ylim( *ylim )

            for ytick in yticks: 
                ax.plot( xlim, [ytick,ytick], color="#808080", lw=0.5, ls='--' )

            #  Define region. 

            if suffix == "H": 
                region = f'{hemisphere}{suffix}'
            else: 
                region = f'{hemisphere} {suffix}'

            ax.text( -0.2, ylim[0]+(ylim[1]-ylim[0])*0.9, region, fontsize="small", color="#0000C0" )

            if isuffix == 0: 
                ax.set_yticks( yticks )
                ax.set_yticklabels( [ str(int(ytick)) for ytick in yticks ] )
                ax.yaxis.set_minor_locator( MultipleLocator(yminor) )
                ax.set_ylabel( r'PDO [$^\circ$ index$^{-1}$]' )
            else: 
                ax.set_yticks( [] )

            if ihemisphere == 0: 
                ax.set_xticks( [] )
            else: 
                ax.set_xticks( xticks )
                ax.set_xticklabels( [ season['name'] for season in seasons ], rotation=-70 )

            #  Plot bar for each season in CCMP analyses. 

            for iseason, season in enumerate(seasons): 
                rec = [ rec for rec in ccmp_analyses if rec['method']==method and rec['season']==season['name'] and rec['region']==region ][0]
                x = iseason * np.array([1,1]) - 0.5 * dx
                y = rec['pdo'] + rec['pdo_uncertainty'] * np.array([-1,1])
                ax.plot( x, y, lw=2, color='k' )

            #  Plot bar for each season in CMIP6 analyses. 

            for imodel, model in enumerate( models ): 
                for iseason, season in enumerate(seasons): 
                    rec = [ rec for rec in cmip6_analyses if rec['model']==model and rec['season']==season['name'] and rec['region']==region ][0]
                    x = iseason * np.array([1,1]) + ( (imodel+1)/nmodels - 0.5 ) * dx
                    y = rec['pdo'] + rec['pdo_uncertainty'] * np.array([-1,1])
                    if iseason==0: 
                        ax.plot( x, y, lw=1, color=colors[imodel], label=model )                    
                    else: 
                        ax.plot( x, y, lw=1, color=colors[imodel] )
            if ihemisphere==1 and isuffix==1: 
                ax.legend( ncol=5, bbox_to_anchor=(0.5,-0.35), loc="upper center", fontsize=6, frameon=False )

    print( f'  Saving to {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Table of AMO
 
def plot_table_of_amo( ccmp_analyses, cmip6_analyses, outputfile="table_of_amo.pdf" ): 
    """Display a graphical version of the table of AMO analyses, CCMP and CMIP6, northern 
    and southern hemispheres, by season and region."""

    print( 'plot_table_of_amo' )
    time0 = time()

    suffixes = [ "H", "Pacific", "Atlantic" ]
    hemispheres = [ "N", "S" ]
    method = "All" 

    xlim = [ -0.5, len(seasons)-0.5 ]
    xticks = np.arange( len(seasons), dtype='i' )

    ylim = [ -1.8, 1.8 ]
    yticks = np.arange( -1.0, 1.001, 1, dtype='f' )
    yminor = 0.2

    superframe = np.array( [ 0.065, 0.35, 0.925, 0.63 ] )
    subframe = np.array( [ 0.01, 0.01, 0.98, 0.98 ] )

    models = sorted( list( set( [ rec['model'] for rec in cmip6_analyses ] ) ) )
    nmodels = len( models )
    dx = 0.6
    cmap = plt.get_cmap( 'gist_ncar' )
    colors = [ cmap( 0.1 + 0.9*(imodel+0.5)/nmodels ) for imodel in range(nmodels) ]

    nx, ny = len(suffixes), len(hemispheres)

    fig = plt.figure( figsize=(6.5,4.0) )

    for ihemisphere, hemisphere in enumerate( hemispheres ): 
        for isuffix, suffix in enumerate( suffixes ): 

            pos = ( subframe + np.array( [ isuffix, ny-ihemisphere-1, 0, 0 ] ) ) / np.array( [ nx, ny, nx, ny ] )
            pos = np.array([superframe[0],superframe[1],0,0]) + np.array([superframe[2],superframe[3],superframe[2],superframe[3]]) * pos
            ax = fig.add_axes( pos )

            ax.set_xlim( *xlim )
            ax.set_ylim( *ylim )

            for ytick in yticks: 
                ax.plot( xlim, [ytick,ytick], color="#808080", lw=0.5, ls='--' )

            #  Define region. 

            if suffix == "H": 
                region = f'{hemisphere}{suffix}'
            else: 
                region = f'{hemisphere} {suffix}'

            ax.text( -0.2, ylim[0]+(ylim[1]-ylim[0])*0.9, region, fontsize="small", color="#0000C0" )

            if isuffix == 0: 
                ax.set_yticks( yticks )
                ax.set_yticklabels( [ str(int(ytick)) for ytick in yticks ] )
                ax.yaxis.set_minor_locator( MultipleLocator(yminor) )
                ax.set_ylabel( r'AMO [$^\circ$ index$^{-1}$]' )
            else: 
                ax.set_yticks( [] )

            if ihemisphere == 0: 
                ax.set_xticks( [] )
            else: 
                ax.set_xticks( xticks )
                ax.set_xticklabels( [ season['name'] for season in seasons ], rotation=-70 )

            #  Plot bar for each season in CCMP analyses. 

            for iseason, season in enumerate(seasons): 
                rec = [ rec for rec in ccmp_analyses if rec['method']==method and rec['season']==season['name'] and rec['region']==region ][0]
                x = iseason * np.array([1,1]) - 0.5 * dx
                y = rec['amo'] + rec['amo_uncertainty'] * np.array([-1,1])
                ax.plot( x, y, lw=2, color='k' )

            #  Plot bar for each season in CMIP6 analyses. 

            for imodel, model in enumerate( models ): 
                for iseason, season in enumerate(seasons): 
                    rec = [ rec for rec in cmip6_analyses if rec['model']==model and rec['season']==season['name'] and rec['region']==region ][0]
                    x = iseason * np.array([1,1]) + ( (imodel+1)/nmodels - 0.5 ) * dx
                    y = rec['amo'] + rec['amo_uncertainty'] * np.array([-1,1])
                    if iseason==0: 
                        ax.plot( x, y, lw=1, color=colors[imodel], label=model )                    
                    else: 
                        ax.plot( x, y, lw=1, color=colors[imodel] )
            if ihemisphere==1 and isuffix==1: 
                ax.legend( ncol=5, bbox_to_anchor=(0.5,-0.35), loc="upper center", fontsize=6, frameon=False )

    print( f'  Saving to {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### CCMP sounding density maps

def plot_ccmpcounts( outputfile="ccmpcounts.pdf" ): 
    """Maps of CCMP sounding density for three epochs: 1995-2004, 2005-2014, 2015-2024."""

    print( 'plot_ccmpcounts' )
    time0 = time()

    # First, get CCMP counts file. 

    countsfile = "ccmpcounts.nc"
    if not os.path.exists( countsfile ): 
        print( f'  Downloading {countsfile}' )
        s3.download_file( bucket, f'ccmp/{countsfile}', countsfile )

    # Generate plot. 

    d = Dataset( countsfile, 'r' )
    lons = d.variables['longitude'][:]
    lats = d.variables['latitude'][:]

    #  Compute area elements. 

    dlon = 2 * np.pi / lons.size

    sinmidlats = np.zeros( lats.size+1 )
    sinmidlats[1:-1] = np.sin( np.deg2rad( 0.5 * ( lats[1:] + lats[:-1] ) ) )
    if lats[0] < lats[1]: 
        sinmidlats[0], sinmidlats[-1] = -1.0, 1.0
    else: 
        sinmidlats[0], sinmidlats[-1] = 1.0, -1.0
    dsinmidlats = np.abs( sinmidlats[1:] - sinmidlats[:-1] )
    da = dsinmidlats * dlon * Re**2

    #  Contour plots. 

    fig = plt.figure( figsize=(4,6) )
    levels = np.arange( 0.0, 1.01, 0.02 ) * 4000
    cmap = plt.get_cmap( 'Oranges' )
    years = d.variables['year'][:]

    for i in range(3): 
    
        yearrange = 1995 + 10*i + np.array( [0,9] )
        ii = np.argwhere( np.logical_and( years >= yearrange[0], years <= yearrange[1] ) ).squeeze()
        m = d.variables['nobs'][ii,:,:].mean(axis=0)
        m = ( m.T * ( 100.0 )**2 / da ).T

        ax = fig.add_axes( [0.01,(0.05+2-i)/3,0.79,0.90/3], projection=ccrs.PlateCarree() )
        ax.coastlines()
        ax.gridlines(draw_labels=False)
        ax.text( -175, 94, "({:}) {:4d} through {:4d}".format( chr( ord('a')+i ), yearrange[0], yearrange[1] ) )
        last_ax = ax.contourf( lons, lats, m, levels=levels, cmap=cmap, extend="max" )

        if i == 1: 
            cax = fig.add_axes( [0.83,0.1,0.03,0.8] )
            fig.colorbar( last_ax, cax=cax, orientation="vertical", extend="max", 
                         ticks=np.arange(0,4000.1,1000), label="Soundings per (100 km)$^2$ per month" )
        
    print( f'  Saving to {outputfile}' )
    fig.savefig( outputfile )

    d.close()

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Sounding density timeseries

def plot_sounding_density_timeseries( outputfile="ccmp_midlat_density_timeseries.pdf" ): 

    print( 'plot_sounding_density_timeseries' )
    time0 = time()

    countsfile = "ccmpcounts.nc"
    d = Dataset( countsfile, 'r' )
    lons = d.variables['longitude'][:]
    lats = d.variables['latitude'][:]
    years = d.variables['year'][:]
    months = d.variables['month'][:]

    #  Justify longitudes. 

    ii = ( lons >= 180 )
    lons[ii] -= 360

    #  Compute area elements. 

    dlon = 2 * np.pi / lons.size

    sinmidlats = np.zeros( lats.size+1 )
    sinmidlats[1:-1] = np.sin( np.deg2rad( 0.5 * ( lats[1:] + lats[:-1] ) ) )
    if lats[0] < lats[1]: 
        sinmidlats[0], sinmidlats[-1] = -1.0, 1.0
    else: 
        sinmidlats[0], sinmidlats[-1] = 1.0, -1.0
    dsinmidlats = np.abs( sinmidlats[1:] - sinmidlats[:-1] )
    da = dsinmidlats * dlon * Re**2

    #  Select North Pacific zone. 

    regions = [ 
        { 'name': "North Pacific", 'latrange': [ 25, 40 ], 'lonrange': [ 165, -135 ] }, 
        { 'name': "North Atlantic", 'latrange': [ 25, 40 ], 'lonrange': [ -75, -15 ] } ]    

    ilats = np.argwhere( np.logical_and( lats >= 25, lats <= 40 ) ).squeeze()

    for region in regions: 
        if region['lonrange'][0] < region['lonrange'][1]: 
            ilons = np.argwhere( np.logical_and( region['lonrange'][0] <= lons, lons <= region['lonrange'][1] ) ).squeeze()
        else: 
            ilons = np.argwhere( np.logical_or( region['lonrange'][0] <= lons, lons <= region['lonrange'][1] ) ).squeeze()
        m = d.variables['nobs'][:,ilats,ilons].mean(axis=2) @ ( 100.0**2/(ilons.size*da[ilats]) )
        region.update( { 'density': m } )
    
    d.close()

    #  Compose plot. 

    fig = plt.figure( figsize=(5,2.5) )
    ax = fig.add_axes( [0.10,0.09,0.86,0.87] )
    ax.set_xlim( 1995, 2025 )
    ax.set_xticks( np.arange(1995,2025.01,5) )
    ax.set_xticks( np.arange(1995,2025.01,1), minor=True )
    ax.set_ylim( 0, 500 )
    ax.set_yticks( np.arange(0,500.1,100) )
    ax.set_yticks( np.arange(0,500.1,20), minor=True )
    ax.set_ylabel( "Sounding density [ (100 km)$^{-2}$ month$^{-1}$ ]" )

    for region in regions: 
        ax.plot( years + (months-0.5)/12, region['density'], lw=0.5, label=region['name'] )

    ax.legend()

    print( f'  Creating {outputfile}' )
    plt.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Spatial variance of zonal wind

def plot_spatial_variability( pps, indexregions, outputfile="spatial_variability.pdf" ): 

    print( 'plot_spatial_variability' )
    time0 = time()

    nlevels = 32
    levels = 16 * np.arange(nlevels+1) / nlevels
    ticks = np.arange( 0, levels.max() + 0.01, 4 )

    fig = plt.figure( figsize=(5,4.5) )

    for iseason, season in enumerate( [ "DJF", "JJA" ] ): 

        ss = [ s for s in seasons if s['name'] == season ][0]
        if ss['monthrange'][0] < ss['monthrange'][1]: 
            imonths = np.arange( ss['monthrange'][0]-1, ss['monthrange'][1], dtype=np.int32 )
        else: 
            imonths = np.arange( ss['monthrange'][0]-1, ss['monthrange'][1]+12, dtype=np.int32 ) % 12
        z = pps['spatial_variance'][imonths,:,:].mean(axis=0)

        ax = fig.add_axes( [ 0.01, 0.51-0.5*iseason, 0.82, 0.42 ], projection=ccrs.PlateCarree( central_longitude=180 ) )
        ax.coastlines( lw=0.4, color="#C0C0C0" )
        last = ax.contourf( pps['oscar_lons'], pps['oscar_lats'], z, levels=levels, 
                       extend="max", cmap='viridis', transform=ccrs.PlateCarree() )
        if season == "DJF": 
            text = '({:}) Boreal winter'.format( chr( ord('a') + iseason ) )
        elif season == "JJA": 
            text = '({:}) Austral winter'.format( chr( ord('a') + iseason ) )
        else: 
            text = '({:})'.format( chr( ord('a') + iseason ) )
        ax.text( -178, 94, text )

        for r in indexregions: 
            lons, lats = r['longituderange'], r['latituderange']
            if not re.search( r'Subtropical Front', r['name'] ): 
                x = [ lons[0], lons[1], lons[1], lons[0], lons[0] ]
                y = [ lats[0], lats[0], lats[1], lats[1], lats[0] ]
                ax.plot( x, y, lw=1.0, color='#FF0000', transform=ccrs.PlateCarree() )

    #  Colorbar. 

    ax = fig.add_axes( [0.85,0.10,0.02,0.80] )
    fig.colorbar( last, ax, orientation="vertical", ticks=ticks, label="Monthly variability in $u_s$ [m$^2$/s$^2$]" )

    print( f'  Saving to {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Timeseries of index regional winds

def plot_index_regional_winds_timeseries( indexregions, outputfile="index_regional_winds.pdf" ): 

    print( 'plot_index_regional_winds_timeseries' )
    time0 = time()

    fig = plt.figure( figsize=(6.5,4) )
    nx, ny = 3, 2

    for iplot, r in enumerate(indexregions): 
        ix, iy = ( iplot % nx ), ny-1 - int(iplot/nx)
        pos = [ 0.08 + ix*0.305, 0.10 + iy*0.46, 0.27, 0.38 ]
        ax = fig.add_axes( pos )
        ax.set_xlim( 1995, 2025 )
        ax.set_xticks( np.arange( 1995, 2025.1, 5 ) )
        ax.xaxis.set_minor_locator( MultipleLocator(1) )
        if iy==0: 
            # ax.set_xlabel( 'Year' )
            ax.xaxis.set_tick_params( which="major", rotation=-60 )
        else: 
            ax.set_xticklabels( [] )

        ax.set_ylim( -2, 2 )
        ax.set_yticks( np.arange( -2, 2.01, 1 ) )
        ax.yaxis.set_minor_locator( MultipleLocator(0.2) )
        if ix==0: 
            ax.set_ylabel( r'Regional $\Delta u_s$ [m/s]' )
        else: 
            ax.set_yticklabels( [] )

        times = r['years'] + ( r['months'] - 0.5 )/12
        ax.plot( times, r['values'].flatten(), lw=1 )

        for y in [-1,0,1]: 
            ax.plot( [1995,2025], [y,y], lw=0.5, ls="--", color="#303030" )
        ax.text( 1995.2, 2.08, '({:}) {:}'.format( chr(ord('a')+iplot), r['name'] ) )

    print( f'  Generating {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


### Regression of index regional winds onto the 500-hPa geopotential field

def plot_era5_index_regressions( indexregions, outputfile="era5_index_regressions.pdf" ): 

    print( 'plot_era5_index_regressions' )
    time0 = time()

    fig = plt.figure( figsize=(6.5,2) )
    levels = np.arange( -600, 600.1, 50 )
    ticks = np.arange( levels.min(), levels.max()+0.01, 200 )
    nx, ny = 3, 2
    for iplot, r in enumerate( era5_indexregion_regressions) : 
        ix, iy = ( iplot % nx ), ny-1 - int(iplot/nx)
        label = '({:}) {:}'.format( chr(ord('a')+iplot), r['name'] )
        ax = fig.add_axes( [ 0.01+0.30*ix, 0.02+0.51*iy, 0.25, 0.40 ], projection=ccrs.PlateCarree(central_longitude=180) )
        ax.coastlines( lw=0.25, color="#008080" )
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                    xlocs=np.arange(-180,180,60), ylocs=np.arange(-90,90,30), 
                    linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        ax.text( 2-180, 98, label, fontsize=7 )
        last = ax.contourf( r['lons'], r['lats'], r['regression']/gravity, 
                    levels=levels, extend="both", cmap="seismic", transform=ccrs.PlateCarree() )

        lons, lats = r['longituderange'], r['latituderange']
        x = [ lons[0], lons[1], lons[1], lons[0], lons[0] ]
        y = [ lats[0], lats[0], lats[1], lats[1], lats[0] ]
        ax.plot( x, y, lw=1.0, color='#00FF00', transform=ccrs.PlateCarree() )

    #  Colorbar. 

    ax = fig.add_axes( [0.88,0.06,0.015,0.80] )
    fig.colorbar( last, ax, orientation="vertical", ticks=ticks, label=r'$\Phi$(500 hPa) onto $u_s$ [m]' )

    print( f'  Saving to {outputfile}' )
    fig.savefig( outputfile )

    dtime = int( time() - time0 )
    print( f'  Elapsed time = {dtime} secs\n' )


def amo_trend_covariance( ccmp_analyses, cmip6_analyses, amo, outputfile="amo_trend_covariance.pdf" ): 
    """Plot error covariance ellipses for trend-AMO detections based on CCMP and CMIP6 models."""

    #  First, compute the trend in AMO rindices over the requisite time interval. 
    #  The units of amo_trend will be AMO index / yr. 

    amo_times = amo['years'] + ( amo['months'] - 0.5 ) / 12
    ii = np.argwhere( np.logical_and( amo_times > 1995, amo_times < 2025 ) ).flatten()
    x = amo_times[ii]
    y = amo['rindices'][ii]
    dx = x - x.mean()
    amo_trend = ( y * dx ).sum() / ( dx * dx ).sum()

    #  Set up figure and axes. 

    fig = plt.figure( figsize=(4,4) )
    ax = fig.add_axes( [ 0.10, 0.10, 0.88, 0.88 ] )

    ticks, ml = np.arange( -4, 4.1, 2 ), MultipleLocator( 1 )

    ax.set_xlim( ticks.min(), ticks.max() )
    ax.set_xticks( ticks )
    ax.xaxis.set_minor_locator( ml )
    ax.set_xlabel( r'Trend [$^\circ$ dec$^{-1}$' )

    ax.set_ylim( ticks.min(), ticks.max() )
    ax.set_yticks( ticks )
    ax.yaxis.set_minor_locator( ml )
    ax.set_ylabel( r'AMO [$^\circ$ dec$^{-1}$' )

    #  CCMP uncertainty covariance. 

    rec = None
    for a in ccmp_analyses: 
        if a['method']=="All" and a['season']=="All" and a['region']=="N Atlantic": 
            rec = a
            break 
    itrend, iamo = rec['labels'].index("trend"), rec['labels'].index("amo")
    cov = rec['covariance'][itrend,iamo]
    trend, amo = rec['trend'], rec['amo']

    #  Eigendecompose. 

    vals, vecs = np.linalg.eig( cov )
    theta = np.arange( 0.0, 1.00001, 0.01 ) * 2 * np.pi

    x = trend
    y = amo

    x += np.sqrt(vals[0]) * vecs[0,0] * np.cos(theta) + np.sqrt(vals[1]) * vecs[0,1] * np.cos(theta)
    y += np.sqrt(vals[1]) * vecs[1,0] * np.cos(theta) + np.sqrt(vals[1]) * vecs[1,1] * np.cos(theta)

    #  Plot ellipse. 

    ax.plot( x / 10.0, y / amo_trend / 10.0, lw=2.0, color="k" )

    #  Continue here. 

def execute_all(): 

    time0 = time()

    if True : 
        pdo = get_pdo_timeseries() 
    else: 
        pdo = None

    if False : 
        amo = get_amo_timeseries() 
    else: 
        amo = None

    ccmp_analyses = compute_ccmp_linear_regressions( outputfile="ccmp_trends.dat", jsonsavefile='ccmp_analyses.json', pdo=pdo, amo=amo ) 
    era5_analyses = compute_era5_linear_regression( outputfile='era5_trends.dat', pdo=pdo, amo=amo ) 
    cmip6_analyses = compute_cmip6_linear_regression( outputfile='cmip6_trends.dat', jsonsavefile="cmip6_analyses.json", pdo=pdo, amo=amo ) 
    pps, indexregions = compute_zonal_wind_variability() 
    era5_indexregion_regressions = compute_indexregion_regressions( indexregions, era5file="geopotential.500hPa.monthly.1995-2024.nc" ) 
    trend, uncertainty = compute_total_width_trend( ccmp_analyses, pdo=pdo, amo=amo ) 

    if pdo is not None: 
        plot_pdo_timeseries( pdo, outputfile="pdo_timeseries.pdf" ) 
    if amo is not None: 
        plot_amo_timeseries( amo, outputfile="amo_timeseries.pdf" ) 
    plot_unull_timeseries( ccmp_analyses, outputfile="ccmp_tropicalwidth_timeseries.pdf" ) 
    plot_trends_by_season( ccmp_analyses, outputfile="ccmp_seasonal_timeseries.pdf" ) 
    plot_wind_climatology( outputfile="ccmp_zonal_mean_zonal_wind.pdf" ) 
    plot_ccmp_era5_timeseries( ccmp_analyses, era5_analyses, outputfile="ccmp_era5_timeseries.pdf" ) 
    plot_ccmp_era5_timeseries_deseasonalized( ccmp_analyses, era5_analyses, pdo=pdo, amo=amo, outputfile="ccmp_era5_deseasonalize_timeseries.pdf" ) 
    plot_table_of_trends( ccmp_analyses, cmip6_analyses, outputfile="table_of_trends.pdf" ) 
    if pdo is not None: 
        plot_table_of_pdo( ccmp_analyses, cmip6_analyses, outputfile="table_of_pdo.pdf" ) 
    if amo is not None: 
        plot_table_of_amo( ccmp_analyses, cmip6_analyses, outputfile="table_of_amo.pdf" ) 
    plot_ccmpcounts( outputfile="ccmpcounts.pdf" ) 
    plot_sounding_density_timeseries( outputfile="ccmp_midlat_density_timeseries.pdf" ) 
    plot_spatial_variability( pps, indexregions, outputfile="spatial_variability.pdf" ) 
    plot_index_regional_winds_timeseries( indexregions, outputfile="index_regional_winds.pdf" ) 
    plot_era5_index_regressions( indexregions, outputfile="era5_index_regressions.pdf" ) 

    dtime = int( time() - time0 )
    minutes, seconds = int( dtime / 60 ), ( dtime % 60 )
    print( f'Elapsed time = {minutes} mins, {seconds:2d} secs\n' )


def execute_select(): 

    time0 = time()

    pdo = None
    amo = get_amo_timeseries() 

    ccmp_analyses = compute_ccmp_linear_regressions( outputfile="ccmp_trends.dat", jsonsavefile='ccmp_analyses.json', pdo=pdo, amo=amo ) 
    cmip6_analyses = compute_cmip6_linear_regression( outputfile='cmip6_trends.dat', jsonsavefile="cmip6_analyses.json", pdo=pdo, amo=amo ) 

    plot_table_of_trends( ccmp_analyses, cmip6_analyses, outputfile="table_of_trends.pdf" ) 
    plot_table_of_amo( ccmp_analyses, cmip6_analyses, outputfile="table_of_amo.pdf" ) 

    dtime = int( time() - time0 )
    minutes, seconds = int( dtime / 60 ), ( dtime % 60 )
    print( f'Elapsed time = {minutes} mins, {seconds:2d} secs\n' )


if __name__ == "__main__": 
    execute_select()
    pass

