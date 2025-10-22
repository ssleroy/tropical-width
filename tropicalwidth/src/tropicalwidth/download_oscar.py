import os
import re
from copy import copy 
import argparse
import earthaccess
from datetime import datetime, timedelta
from .libtropicalwidth import default_dataroot


#  Earthdata authentication. 

auth = earthaccess.login( persist=True )

#  Short names for OSCAR data sets. 

fnl_name = "OSCAR_L4_OC_FINAL_V2.0"
nrt_name = "OSCAR_L4_OC_NRT_V2.0"


def download_oscar( year, month, dataroot=default_dataroot, clobber=False ): 

    #  Authenticate and set up earthaccess. 

    dtime0 = datetime( year=year, month=month, day=1 )
    dtime1 = dtime0 + timedelta(days=31)
    dtime1 = datetime( year=dtime1.year, month=dtime1.month, day=1 ) - timedelta(days=1)
    temporal = ( dtime0.strftime( "%Y-%m-%d" ), dtime1.strftime( "%Y-%m-%d" ) )

    try: 
        fnl = earthaccess.search_data( short_name=fnl_name, temporal=temporal )
    except: 
        fnl = []

    try: 
        nrt = earthaccess.search_data( short_name=nrt_name, temporal=temporal )
    except: 
        nrt = []

    fnl_recs = {}
    for f in fnl: 
        m = re.search( r'(\d{8})$', f['umm']['GranuleUR'] )
        if m: 
            fnl_recs.update( { m.group(1): { 'value': f, 'product': "fnl" } } )

    nrt_recs = {}
    for f in nrt: 
        m = re.search( r'(\d{8})$', f['umm']['GranuleUR'] )
        if m: 
            nrt_recs.update( { m.group(1): f } )
            nrt_recs.update( { m.group(1): { 'value': f, 'product': "nrt" } } )

    #  Merge fnl and nrt. fnl takes precedence. 

    merge = {}
    merge.update( nrt_recs )
    merge.update( fnl_recs )

    #  Separate fnl and nrt records. 

    dates = sorted( list( merge.keys() ) )

    nrt_download = []
    fnl_download = []

    for date in dates: 
        rec = merge[date]
        if rec['product'] == "nrt": 
            nrt_download.append( rec['value'] )
        elif rec['product'] == "fnl": 
            fnl_download.append( rec['value'] )

    #  Download data. 

    print( f'Downloading {len(nrt_download)+len(fnl_download)} files for {year:4d}-{month:02d}' )

    ldir = os.path.join( dataroot, "oscar", f'{year:4d}', f'{month:02d}' )
    os.makedirs( ldir, exist_ok=True )

    lpaths = []
    if len( nrt_download ) > 0: 
        lpaths += earthaccess.download( nrt_download, local_path=ldir )
    if len( fnl_download ) > 0: 
        lpaths += earthaccess.download( fnl_download, local_path=ldir )

    #  Done. 

    return lpaths


def main(): 

    parser = argparse.ArgumentParser( description="Download OSCAR surface current data. Be " + \
            "certain that your NASA Earthdata username and password are in order in your " + \
            "~\.netrc file for machine urs.earthdata.nasa.gov." )

    parser.add_argument( "monthrange", type=str, help='The range of months over which to ' + \
            'download OSCAR ocean surface current data; the format should be ' + \
            '"YYYY-MM:YYYY-MM", and the range is inclusive' )

    parser.add_argument( "--dataroot", "-d", dest="dataroot", default=default_dataroot,
            help="Root of all data for the tropical width analysis project; " + \
                f'the default is "{default_dataroot}"' )

    args = parser.parse_args()

    m = re.search( r'^(\d{4}-\d{2}):(\d{4}-\d{2})$', args.monthrange )
    if m: 
        yearmonth1 = m.group(1)
        yearmonth2 = m.group(2)
    else: 
        print( 'Be sure that the monthrange has format "YYYY-MM:YYYY-MM".' )
        return

    if yearmonth1 > yearmonth2: 
        print( "monthrange values must be in ascending order." )
        return

    yearmonth = copy( yearmonth1 )

    while yearmonth <= yearmonth2: 
        year, month = int( yearmonth[0:4] ), int( yearmonth[5:7] )
        ret = download_oscar( year, month, dataroot=default_dataroot )

        #  Next month. 

        month += 1
        if month > 12: 
            year += 1
            month = 1
        yearmonth = f'{year:4d}-{month:02d}'

    #  Done. 

    return


if __name__ == "__main__": 
    main()
    pass

