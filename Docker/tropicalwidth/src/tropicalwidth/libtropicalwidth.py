#
#  Module: libtropicalwidth
#  Version: 1.0
#  Date: March 12, 2025
#  Author: Stephen Leroy (sleroy@aer.com)

import os
import re
import sys
import requests
import copy
from tqdm import tqdm
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
from global_land_mask import globe
import boto3


################################################################################
#  Physical constants. 
################################################################################

#  Define oceanic regions in the northern and southern hemispheres. 

regions = [ 
           { 'name': 'NH', 'latbounds': [20, 48], 'oceanmask': False  }, 
           { 'name': 'N Oceans', 'latbounds': [20, 48], 'oceanmask': True  }, 
           { 'name': 'SH', 'latbounds': [-48, -20], 'oceanmask': False }, 
           { 'name': 'S Oceans', 'latbounds': [-48, -20], 'oceanmask': True }, 
           { 'name': 'N Pacific', 'lonbounds': [140,-120], 'latbounds':[20,48], 'oceanmask': True }, 
           { 'name': 'S Pacific', 'lonbounds': [160,-80], 'latbounds':[-48,-20], 'oceanmask': True }, 
           { 'name': 'N Atlantic', 'lonbounds':[-70,-15], 'latbounds':[20,48], 'oceanmask': True }, 
           { 'name': 'S Atlantic', 'lonbounds':[-40,10], 'latbounds':[-48,-20], 'oceanmask': True }
        ]

################################################################################
#  Other parameters. 
################################################################################

bucket = "aer-sleroy-tropical-width"
account = "aerdev"

default_dataroot = "/fg"
fill_value = -999.0
oscar_tmp = "oscar_tmp.nc"

################################################################################
#  AWS session, s3 client. 
################################################################################

session = boto3.Session( region_name="us-east-1" )
s3 = session.client( "s3" )

################################################################################
#  Exception handling. 
################################################################################

class Error(Exception):
    pass

class tropicalWidthError(Error):
    def __init__(self,expression,message=''):
        self.expression = expression
        self.message = message

class averageWindError(Error): 
    def __init__(self,expression,message=''):
        self.expression = expression
        self.message = message



################################################################################
#  Class Model. This defines the model that will be interpolated and stores
#  all buffered data. 
################################################################################

MAXOPENFILES = 2
MAXSTOREDTIMES = 8

class Model(): 

    def __init__( self, model='daily', version="v03.1", dataroot="/fg", clobberoceanmask=False, oscar=True ):
        """Define a model class. In this case, the model class refers to the temporal 
        resolution of the data, either 'daily' or 'monthly'."""

        self.model = model.lower()
        self.fileptrs = []
        self.timeptrs = []
        self.lons = []
        self.lats = []
        self.regions = []
        self.clobberoceanmask = clobberoceanmask
        self.oceanmask = None
        self.oscar = oscar

        #  Model-specific versioning, naming. 

        self.version = version
        self.modelnamestring = 'CCMP-' + model.lower()

        #  Set up communication metadata, model cycle time, and parameters 
        #  for defining model time. 

        if self.model=='daily': self.cycletime = 6
        if self.model=='monthly': self.cycletime = None
        self.rootpath = os.path.join( dataroot, "ccmp" )
        self.oceanroot = os.path.join( dataroot, "oscar" )

    def paths( self, time ): 

        remotehost = "http://data.remss.com"
        remotedir = "/".join( [ "ccmp", self.version, f'Y{time.year:04d}', f'M{time.month:02d}' ] )
        localdir = os.path.join( self.rootpath, f'Y{time.year:4d}', f'M{time.month:02d}' )
        version = self.version.upper()

        if ( self.model=='daily' ): 
            file = f"CCMP_Wind_Analysis_{time.year:4d}{time.month:02d}{time.day:02d}_{version}_L4.nc"
            rpath, lpath = "/".join( [remotehost,remotedir,file] ), os.path.join( localdir, file )
        if ( self.model=='monthly' ):
            file = f"CCMP_Wind_Analysis_{time.year:4d}{time.month:02d}_monthly_mean_{version}_L4.nc"
            rpath, lpath = "/".join( [remotehost,remotedir,file] ), os.path.join( localdir, file )

        return dict( rpath=rpath, lpath=lpath )


    def initialize( self, ptr ):
        """
        Obtain longitudes and latitudes.  The pointer given as an argument 
        can be either a FilePtr or a TimePtr.
        """

        if isinstance( ptr, FilePtr ): 
            p = ptr.p
        elif isinstance( ptr, TimePtr ): 
            p = ptr.fileptr.p
        if len(self.lons) > 0: 
            return

        #  Define longitude-latitude grid. 

        self.lons = ptr.lons
        self.lats = ptr.lats

        #  Ocean mask.

        s = os.getenv( "OCEANMASK" )
        if s == "" or s is None: 
            oceanmaskfile = 'oceanmask.nc'
        else: 
            oceanmaskfile = s

        #  Generate an ocean mask if one hasn't already been generated. 

        if os.path.isfile( oceanmaskfile ) and not self.clobberoceanmask : 

            d = Dataset( oceanmaskfile, 'r' )
            self.oceanmask = d.variables['oceanmask'][:,:]
            d.close()

        else:

            print( 'Creating an ocean mask.' )
            sys.stdout.flush()
            self.oceanmask = np.zeros( ( self.lats.size, self.lons.size ), dtype='b' )
            for ilat in tqdm( range(self.lats.size), desc="Latitude" ): 
                lat = self.lats[ilat]
                for ilon,lon in enumerate(self.lons): 
                    self.oceanmask[ilat,ilon] = globe.is_ocean( lat, lon )

            #  Store ocean mask to a file. 

            d = Dataset( oceanmaskfile, 'w', format='NETCDF3_CLASSIC' )
            d.createDimension( 'lon', size=len(self.lons) )
            d.createDimension( 'lat', size=len(self.lats) )
            d_lons = d.createVariable( 'lons', 'f', dimensions=('lon',) )
            d_lons.setncatts( { 'description':'Longitude', 'units':'degrees east' } )
            d_lats = d.createVariable( 'lats', 'f', dimensions=('lat',) )
            d_lats.setncatts( { 'description':'Latitude', 'units':'degrees north' } )
            d_oceanmask = d.createVariable( 'oceanmask', 'b', dimensions=('lat','lon') )
            d_oceanmask.setncatts( { 'description':'An ocean mask, 0 for land, 1 for water' } )
            d_lons[:] = self.lons[:]
            d_lats[:] = self.lats[:]
            d_oceanmask[:] = self.oceanmask[:]
            d.close()

        rlons = np.deg2rad( p.variables['longitude'][:] )
        self.lons = np.rad2deg( np.arctan2( np.sin(rlons), np.cos(rlons) ) )
        self.lats = p.variables['latitude'][:]

        #  Define region masks. 

        lats2 = self.lats[ np.indices( ( self.lats.size, self.lons.size ) )[0,:,:] ]
        lons2 = self.lons[ np.indices( ( self.lats.size, self.lons.size ) )[1,:,:] ]

        #  Loop over above-defined regions. 

        self.regions = copy.deepcopy( regions )

        for region in self.regions: 

            #  Mask by latitude (always). 

            regionmask = np.logical_and( region['latbounds'][0] < lats2, lats2 < region['latbounds'][1] )

            #  Mask by longitude? 

            if 'lonbounds' in region.keys(): 
                dlons2 = np.deg2rad( lons2 - region['lonbounds'][0] ) 
                dlons2 = np.arctan2( -np.sin(dlons2), -np.cos(dlons2) ) + np.pi
                dlon = np.rad2deg( region['lonbounds'][1] - region['lonbounds'][0] ) 
                dlon = np.arctan2( -np.sin(dlon), -np.cos(dlon) ) + np.pi
                regionmask = np.logical_and( regionmask, dlons2 <= dlon )

            #  Ocean only? 

            if region['oceanmask']: 
                regionmask = np.logical_and( regionmask, self.oceanmask )

            region.update( { 'regionmask': regionmask.astype('b').data } )

    def matchtimeptr(self,var,time):
        """
        Find and return the TimePtr already loaded into memory, the memory being
        referenced internally. If the time interval is not currently stored in 
        memory, return None.
        """

        if self.model=='daily': 
            itimeptr = [ i for i,p in enumerate(self.timeptrs) \
                if p.time.year==time.year and p.time.month==time.month \
                and p.time.day==time.day and p.time.hour==time.hour and p.varname==var ]
            if len(itimeptr) > 0: 
                timeptr = self.timeptrs.pop( itimeptr[0] )
                self.timeptrs.append( timeptr )
            else:
                timeptr = None
            return timeptr

        if self.model=='monthly': 
            itimeptr = [ i for i,p in enumerate(self.timeptrs) \
                if p.time.year==time.year and p.time.month==time.month and p.varname==var ]
            if len(itimeptr) > 0: 
                timeptr = self.timeptrs.pop( itimeptr[0] )
                self.timeptrs.append( timeptr )
            else:
                timeptr = None
            return timeptr

    def loadtime(self,var,time):
        """
        Load the variable 'var' for time increment [class datetime.datetime] into memory.
        """

        if self.model=='daily': 
            ifileptr = [ i for i,p in enumerate(self.fileptrs) for t in p.times \
                if time.year==t.year and time.month==t.month \
                    and time.day==t.day and time.hour==t.hour ]

        if self.model=='monthly': 
            ifileptr = [ i for i,p in enumerate(self.fileptrs) for t in p.times \
                if time.year==t.year and time.month==t.month ]

        if len(ifileptr) > 0: 
            fileptr = self.fileptrs[ ifileptr[0] ]
            if len(self.timeptrs) >= MAXSTOREDTIMES: del self.timeptrs[0] 
            self.timeptrs.append( TimePtr( self, var, fileptr, time ) )
            timeptr = self.timeptrs[-1]
        else: 
            timeptr = None
        return timeptr


    def matchfileptr(self,var,time): 
        """
        Find and return the FilePtr toward a file which is currently open. If 
        the file corresponding to time [class datetime.datetime] is not currently open, 
        return None.
        """

        if self.model=='daily': 
            ifileptr = [ i for i,p in enumerate(self.fileptrs) for t in p.times \
                if time.year==t.year and time.month==t.month \
                    and time.day==t.day and time.hour==t.hour ]

        if self.model=='monthly': 
            ifileptr = [ i for i,p in enumerate(self.fileptrs) for t in p.times \
                if time.year==t.year and time.month==t.month ]

        if len(ifileptr) > 0: 
            fileptr = self.fileptrs.pop( ifileptr[0] )
            self.fileptrs.append( fileptr )
            if len(self.timeptrs) >= MAXSTOREDTIMES: del self.timeptrs[0] 
            self.timeptrs.append( TimePtr( self, var, fileptr, time ) )
        else: 
            fileptr = None
        return fileptr

    def loadfile(self,time): 
        """
        Open a model file corresponding time interval time [class datetime.datetime]. 
        """

        lfile = self.paths(time)['lpath']
        if os.path.isfile(lfile): 
            if len(self.fileptrs) >= MAXOPENFILES: 
                self.fileptrs[0].close()
                del self.fileptrs[0]
            self.fileptrs.append( FilePtr( self, lfile, time=time ) )
            return lfile
        else:
            return None


    def getanalysis(self,time): 
        """
        This function obtains the model analysis file from the model data archive; 
        GESDISC for MERRA, etc. The file is stored locally for future use. 
        """

        paths = self.paths( time )
        resp = requests.get( paths['rpath'], stream=True )

        #  If remote file does not exist, raise an exception. 

        if resp.status_code != 200: raise tropicalWidthError( 'FileNotOnServer', \
            'Analysis {:} not on server.'.format( paths['rpath'] ) )

        #  If remote file exists, make a local path, read the image (netcdf), and write to disk. 

        localdir = os.path.split( paths['lpath'] )[0]
        os.makedirs( localdir, exist_ok=True )
        print( "Retrieving " + paths['lpath'] )
        sys.stdout.flush()
        img = resp.raw.read()
        with open(paths['lpath'],'wb') as f: f.write(img)
        resp.close()

        #  Return the path to the new local file. 

        return paths['lpath']


    def get_data( self, var, t ): 
        """
        Get data in the form of class timeptr for time t (class datetime.datetime) for variable
        var. Return None if unable to obtain the timeptr.
        """

        timeptr = None

        while timeptr is None: 

            #  Check whether the time pointer is currently in memory. 

            timeptr = self.matchtimeptr( var, t )
            if timeptr: break

            #  Load time pointer if the time is available. 

            timeptr = self.loadtime( var, t )
            if timeptr: break

            #  If time pointer is not available, then see if the corresponding 
            #  file is open. If the file is open, load the time interval. 

            fileptr = self.matchfileptr( var, t )
            if fileptr: continue

            #  If the file is not open, open it if it is available locally. 
            #  If the file is not available locally, retrieve it from GESDisc. 

            localfile = self.loadfile( t )
            if localfile: continue

            #  If the file doesn't exist locally, download it. 

            try: 
                localfile = self.getanalysis( t )
            except tropicalWidthError as e: 
                if e.expression=='FileNotOnServer': 
                    print( e.message )
                    break
            continue

        return timeptr

    def close(self): 
        for fileptr in self.fileptrs: fileptr.close()
        self.fileptrs = []
        self.timeptrs = []


class FilePtr():

    def __init__( self, model, modelfile, time=None ):
        """
        A file pointer defines an open model data file and contains
        information on what time intervals are in that file. Optional
        argument time should contain a class datetime.datetime that gives the 
        time corresponding to a file, to be used only in case of model
        "monthly". 
        """

        self.file = modelfile
        self.model, self.version = model.model, model.version

        self.p = Dataset( modelfile, 'r' )
        self.times = []
        if model.model=='monthly': 
            if time is None: 
                raise tropicalWidthError( 'InvalidArgument', \
                    'Time argument must be given for model="monthly".' )
            t = datetime( year=time.year, month=time.month, day=1 )
            self.times.append( t )
        else:
            t0 = datetime( year=1987, month=1, day=1 )
            times = [ t0 + timedelta(hours=float(hr)) for hr in self.p.variables['time'][:] ]
            for itime,tm in enumerate(times):
                t = datetime( year=tm.year, month=tm.month, day=tm.day, hour=tm.hour )
                self.times.append( t )
        x = np.deg2rad( self.p.variables['longitude'][:] )
        self.lons = np.rad2deg( np.arctan2(np.sin(x),np.cos(x)) )
        self.lats = self.p.variables['latitude'][:] 

        #  Get ocean surface currents. 

        if model.oscar: 

            if model.model=='monthly': 
                s3path = os.path.join( "oscar", f'{t.year:04d}', f'{t.month:02d}', 
                                 f'oscar_currents_{t.year:04d}{t.month:02d}.nc4' )
                ret = s3.list_objects_v2( Bucket=bucket, Prefix=s3path )

                if ret['KeyCount'] == 1: 
                    s3.download_file( bucket, s3path, oscar_tmp )
                    try: 
                        o = Dataset( oscar_tmp, 'r' )
                    except: 
                        raise tropicalWidthError( "UnreadableFile", f'Could not read file s3://{s3path}' )
                    lons = o.variables['lon'][:]
                    lats = o.variables['lat'][:]
                    u = o.variables['u'][:].T
                    v = o.variables['v'][:].T
                    o.close()
                    os.unlink( oscar_tmp )
                    found = True
                else: 
                    print( f'No OSCAR surface currents for {t.year:04d}-{t.month:02d}, s3path={s3path}' )
                    sys.stdout.flush()
                    found = False

            elif model.model=='daily': 
                prefix = os.path.join( "oscar", f'{t.year:04d}', f'{t.month:02d}', 
                                 f'oscar_currents_' )
                ret = s3.list_objects_v2( Bucket=bucket, Prefix=prefix )

                if ret['KeyCount'] > 0: 
                    ss = f'oscar_currents_[a-z]+_{t.year:04d}{t.month:02d}{t.day:02d}_subsetted.nc4'
                    keys = [ obj['Key'] for obj in ret['Contents'] if re.search( ss, obj['Key'] ) ]
                else: 
                    keys = []

                if len( keys ) == 1: 
                    s3path = keys[0]
                    s3.download_file( bucket, s3path, oscar_tmp )
                    try: 
                        o = Dataset( oscar_tmp, 'r' )
                    except: 
                        raise tropicalWidthError( "UnreadableFile", f'Could not read file s3://{s3path}' )
                    lons = o.variables['lon'][:]
                    lats = o.variables['lat'][:]
                    u = o.variables['u'][:].squeeze().T
                    v = o.variables['v'][:].squeeze().T
                    o.close()
                    os.unlink( oscar_tmp )
                    found = True
                else: 
                    print( f'No OSCAR surface currents for {t.year:04d}-{t.month:02d}-{t.day:02d}' )
                    sys.stdout.flush()
                    found = False

            self.oscar_u = np.ma.zeros( (self.lats.size,self.lons.size), np.float32 )
            self.oscar_v = np.ma.zeros( (self.lats.size,self.lons.size), np.float32 )

            if found: 

                for x, z in [ (u,self.oscar_u), (v,self.oscar_v) ]: 

                    #  Account for longitude shift. 

                    y = np.ma.zeros( (lats.size,lons.size), np.float32 )
                    y[:,:-1] = 0.5 * ( x[:,:-1] + x[:,1:] )
                    y[:,-1] = 0.5 * ( x[:,-1] + x[:,0] )

                    #  Account for latitude shift. 

                    z[1:-1,:] = 0.5 * ( y[:-1,:] + y[1:,:] )


    def close(self):
        self.p.close()
        self.file = None
        self.p = None
        self.times = []


class TimePtr():

    def __init__( self, model, varname, fileptr, time ):
        """
        Create a time pointer. A time pointer contains a three-dimensional 
        snapshot of the variable defined by varname for a particular time 
        defined by time [class datetime.datetime]. 
        """

        self.fileptr = fileptr
        self.time = time + timedelta(seconds=0)
        self.varname = varname

        if model.model=='daily': 
            itime = [ i for i,t in enumerate(fileptr.times) \
                if t.year==time.year and t.month==time.month and t.day==time.day and t.hour==time.hour ]
            if varname == 'u': 
                self.data = np.squeeze( fileptr.p.variables['uwnd'][itime[0],:,:] )
            elif varname == 'v': 
                self.data = np.squeeze( fileptr.p.variables['vwnd'][itime[0],:,:] )
            elif varname == 'n': 
                self.data = np.squeeze( fileptr.p.variables['nobs'][itime[0],:,:] )
            else: 
                raise tropicalWidthError( 'UnknownVariable', '{:} not in file.'.format(varname) )

        if model.model=='monthly': 
            itime = [ i for i,t in enumerate(fileptr.times) if t.year==time.year and t.month==time.month ]
            if varname not in { 'u', 'v', 'n' }: 
                raise tropicalWidthError( 'UnknownVariable', '{:} not in file.'.format(varname) )
            self.data = np.squeeze( fileptr.p.variables[varname][:,:] )

        if model.oscar: 
            if varname == 'u': 
                self.data += fileptr.oscar_u
            elif varname == 'v': 
                self.data += fileptr.oscar_v

        self.lons = fileptr.lons
        self.lats = fileptr.lats

    def close(self):
        self.fileptr = None
        self.time = None
        self.varname = None
        self.data = None


################################################################################
#  tropicalWidth
################################################################################

def tropicalWidth( timerange, model, n=0 ):
    """
    Generate a monthly timeseries of the width of the Tropics as established by 
    the u=0 line in both the northern and southern hemispheres. The function 
    returns a two-element tuple: the first is an array of datetime.datetime 
    class times designating the months, the second is a list of regions 
    containing the region definitions and an array of latitudes where u=0 
    for each region. The regions are defined in the model (instance of class 
    Model). 

    Set n the minimum number of observations in a grid cell of CCMP for it 
    to be considered in forming zonal averages. 
    """

    #  Check caller arguments. 

    m = model
    if ( m.model=='monthly' and n>0 ): 
        raise tropicalWidthError( 'InvalidCall', 'Cannot request using monthly average ' \
            + 'fields and screen by the number of observations' )

    #  Initialize. 

    months = []

    #  First month, last month. 

    t0 = datetime( year=timerange[0].year, month=timerange[0].month, day=1 )
    t1 = datetime( year=timerange[1].year, month=timerange[1].month, day=1 )
    tmonth = t0 + timedelta(seconds=0)

    #  Initialize regions. 

    if len( m.regions ) == 0 : 
        m.initialize( m.get_data( 'u', t0 ) )
        nlons, nlats = m.lons.size, m.lats.size 
        for r in m.regions: 
            r.update( { 'lats': [] } )

    #  Loop over month. 

    while ( tmonth <= t1 ): 

        #  Loop over all time elements for this month. 

        t = tmonth + timedelta(seconds=0)

        #  Refresh zonal average and time counter for month. 

        for r in m.regions: 
            r.update( { 'za': np.zeros( nlats, dtype='f' ), 'ntimes': np.zeros( nlats, dtype='i' ) } )

        #  Initialize regions. 

        while ( t.year==tmonth.year and t.month==tmonth.month ): 

            #  Is the time interval already available? 

            var1, var2 = 'u', 'n'
            timeptr1, timeptr2 = None, None
            fileptr1, fileptr2 = None, None

            timeptr1 = m.get_data( var1, t )

            #  The data fields should now be in timeptr1 and timeptr2. 

            if m.model=='daily': 
                timeptr2 = m.get_data( var2, t )
                if timeptr1 is None or timeptr2 is None: 
                    t += timedelta(hours=m.cycletime)
                    continue
            elif m.model=='monthly': 
                if timeptr1 is None: 
                    t += timedelta(days=31)
                    t = datetime( year=t.year, month=t.month, day=1 )
                    continue

            for region in m.regions: 

                #  Mask the data. 

                mask = region['regionmask']

                #  Mask by valid data. 

                if ( m.model=='daily' ): 
                    mask = np.logical_and( mask, ( timeptr2.data >= n ) )

                #  Execute zonal average time average. 

                ms = mask.sum(axis=1)
                good = ( ms > 0 )
                region['za'][good] += ( timeptr1.data * mask ).sum( axis=1 )[good] / ms[good]
                region['ntimes'][good] += 1

            #  Increment time. 

            if ( m.model=='daily' ): 
                t += timedelta( hours=model.cycletime )
            elif ( m.model=='monthly' ): 
                t += timedelta(days=31)
                t = datetime( year=t.year, month=t.month, day=1 )

        #  Normalize time. 

        for region in m.regions: 

            if m.model=='daily': 
                g = ( region['ntimes'] >= 20 )
            elif m.model=='monthly': 
                g = ( region['ntimes'] == 1)

            region['za'][g] /= region['ntimes'][g]
            region['za'] = np.ma.masked_where( np.logical_not(g), region['za'] )

            ii = np.argwhere( region['za'][:-1] * region['za'][1:] <= 0 )
            if ii.size == 0: 
                lat = fill_value
            else: 
                i = ii[0,0]
                ti = region['za'][i] / ( region['za'][i] - region['za'][i+1] )
                lat = m.lats[i]*(1-ti) + m.lats[i+1]*(ti)
            region['lats'].append( float( lat ) )

        #  Month. 

        tm = datetime( year=tmonth.year, month=tmonth.month, day=1 )
        months.append( tm )

        #  Update month counter. 

        tmonth += timedelta( days=31 )
        tmonth = datetime( year=tmonth.year, month=tmonth.month, day=1 )

    #  Close. 

    model.close()

    for region in m.regions: 
        x = np.array( region['lats'] )
        region['lats'] = np.ma.masked_where( x == fill_value, x )

    return months, m.regions


################################################################################
#  averageWind
################################################################################

def averageWind( timerange, model ):
    """Compute the average zonal mean zonal wind profile in latitude. A two-element 
    tuple is output. The first elements is the array of latitudes. The second 
    element is a dictionary with seasonal keys ( 'DJF', 'MAM', 'JJA', 'SON' )
    that point to arrays of zonal average zonal wind averaged over the time 
    period given by timerange."""

    #  Check caller arguments. 

    m = model

    if m.model != 'monthly': 
        raise averageWindError( 'Invalid model', \
            'The model data set should be a timeseries of monthly averages.' )

    #  Initialize. 

    ubar = { 'DJF': None, 'MAM': None, 'JJA': None, 'SON': None }

    #  First month, last month. 

    t0 = datetime( year=timerange[0].year, month=timerange[0].month, day=1 )
    t1 = datetime( year=timerange[1].year, month=timerange[1].month, day=1 )

    #  Loop over season. 

    firsttime = True

    for season in ubar.keys(): 

        if season=='DJF': startmonth = 12
        if season=='MAM': startmonth = 3
        if season=='JJA': startmonth = 6
        if season=='SON': startmonth = 9

        year0 = t0.year
        t = datetime( year=t1.year, month=startmonth, day=1 ) + timedelta(days=62)
        t = datetime( year=t.year, month=t.month, day=1 )
        if t.year <= t1.year: 
            year1 = t1.year
        else: 
            year1 = t1.year - 1

        nyears = 0

        for year in np.arange(year0,year1+1): 
            t = datetime( year=year, month=startmonth, day=1 )

            nmonths = 0
            for imonth in range(3): 

                #  Is the time interval already available? 

                var = 'u'
                timeptr = None
                fileptr = None

                timeptr = m.get_data( var, t )
                if timeptr is None: break

                #  Initialization: get the longitudes, latitudes. 

                m.initialize( timeptr )
                nlons, nlats = m.lons.size, m.lats.size

                #  Initialization. 

                if firsttime: 
                    mask = np.zeros( (nlats,nlons), dtype='i' ) + 1
                    mask = np.logical_and( mask, m.oceanmask )
                    firsttime = False

                if imonth==0: 
                    annualaverage = np.zeros( nlats )

                if year == year0: 
                    average = np.zeros( nlats )

                #  Execute zonal average time average. 

                za = np.zeros( nlats )
                n = mask.sum(axis=1)
                good = ( n > 0 )
                za[good] = ( timeptr.data * mask ).sum( axis=1 )[good] / n[good]

                #  Add to annual average. 

                annualaverage = annualaverage + za
                nmonths = nmonths + 1

                #  Increment time. 

                t += timedelta( days=31 )
                t = datetime( year=t.year, month=t.month, day=1 )

            #  Annual average. 

            if nmonths==3: 
                annualaverage = annualaverage / 3
                average = average + annualaverage
                nyears = nyears + 1

        #  Average over years. 

        ubar[season] = average / nyears

    #  Return. 

    return m.lats, ubar


################################################################################
#  tropicalWidth
################################################################################

def tropicalWidth( timerange, model, n=0 ):
    """
    Generate a monthly timeseries of the width of the Tropics as established by 
    the u=0 line in both the northern and southern hemispheres. The function 
    returns a two-element tuple: the first is an array of datetime.datetime 
    class times designating the months, the second is a list of regions 
    containing the region definitions and an array of latitudes where u=0 
    for each region. The regions are defined in the model (instance of class 
    Model). 

    Set n the minimum number of observations in a grid cell of CCMP for it 
    to be considered in forming zonal averages. 
    """

    #  Check caller arguments. 

    m = model
    if ( m.model=='monthly' and n>0 ): 
        raise tropicalWidthError( 'InvalidCall', 'Cannot request using monthly average ' \
            + 'fields and screen by the number of observations' )

    #  Initialize. 

    months = []

    #  First month, last month. 

    t0 = datetime( year=timerange[0].year, month=timerange[0].month, day=1 )
    t1 = datetime( year=timerange[1].year, month=timerange[1].month, day=1 )
    tmonth = t0 + timedelta(seconds=0)

    #  Initialize regions. 

    if len( m.regions ) == 0 : 
        m.initialize( m.get_data( 'u', t0 ) )
        nlons, nlats = m.lons.size, m.lats.size 
        for r in m.regions: 
            r.update( { 'lats': [] } )

    #  Loop over month. 

    while ( tmonth <= t1 ): 

        #  Loop over all time elements for this month. 

        t = tmonth + timedelta(seconds=0)

        #  Refresh zonal average and time counter for month. 

        for r in m.regions: 
            r.update( { 'za': np.zeros( nlats, dtype='f' ), 'ntimes': np.zeros( nlats, dtype='i' ) } )

        #  Initialize regions. 

        while ( t.year==tmonth.year and t.month==tmonth.month ): 

            #  Is the time interval already available? 

            var1, var2 = 'u', 'n'
            timeptr1, timeptr2 = None, None
            fileptr1, fileptr2 = None, None

            timeptr1 = m.get_data( var1, t )

            #  The data fields should now be in timeptr1 and timeptr2. 

            if m.model=='daily': 
                timeptr2 = m.get_data( var2, t )
                if timeptr1 is None or timeptr2 is None: 
                    t += timedelta(hours=m.cycletime)
                    continue
            elif m.model=='monthly': 
                if timeptr1 is None: 
                    t += timedelta(days=31)
                    t = datetime( year=t.year, month=t.month, day=1 )
                    continue

            for region in m.regions: 

                #  Mask the data. 

                mask = region['regionmask']

                #  Mask by valid data. 

                if ( m.model=='daily' ): 
                    mask = np.logical_and( mask, ( timeptr2.data >= n ) )

                #  Execute zonal average time average. 

                ms = mask.sum(axis=1)
                good = ( ms > 0 )
                region['za'][good] += ( timeptr1.data * mask ).sum( axis=1 )[good] / ms[good]
                region['ntimes'][good] += 1

            #  Increment time. 

            if ( m.model=='daily' ): 
                t += timedelta( hours=model.cycletime )
            elif ( m.model=='monthly' ): 
                t += timedelta(days=31)
                t = datetime( year=t.year, month=t.month, day=1 )

        #  Normalize time. 

        for region in m.regions: 

            if m.model=='daily': 
                g = ( region['ntimes'] >= 20 )
            elif m.model=='monthly': 
                g = ( region['ntimes'] == 1)

            region['za'][g] /= region['ntimes'][g]
            region['za'] = np.ma.masked_where( np.logical_not(g), region['za'] )

            ii = np.argwhere( region['za'][:-1] * region['za'][1:] <= 0 )
            if ii.size == 0: 
                lat = fill_value
            else: 
                i = ii[0,0]
                ti = region['za'][i] / ( region['za'][i] - region['za'][i+1] )
                lat = m.lats[i]*(1-ti) + m.lats[i+1]*(ti)
            region['lats'].append( float( lat ) )

        #  Month. 

        tm = datetime( year=tmonth.year, month=tmonth.month, day=1 )
        months.append( tm )

        #  Update month counter. 

        tmonth += timedelta( days=31 )
        tmonth = datetime( year=tmonth.year, month=tmonth.month, day=1 )

    #  Close. 

    model.close()

    for region in m.regions: 
        x = np.array( region['lats'] )
        region['lats'] = np.ma.masked_where( x == fill_value, x )

    return months, m.regions


################################################################################
#  averageWind
################################################################################

def averageWind( timerange, model ):
    """Compute the average zonal mean zonal wind profile in latitude. A two-element 
    tuple is output. The first elements is the array of latitudes. The second 
    element is a dictionary with seasonal keys ( 'DJF', 'MAM', 'JJA', 'SON' )
    that point to arrays of zonal average zonal wind averaged over the time 
    period given by timerange."""

    #  Check caller arguments. 

    m = model

    if m.model != 'monthly': 
        raise averageWindError( 'Invalid model', \
            'The model data set should be a timeseries of monthly averages.' )

    #  Initialize. 

    ubar = { 'DJF': None, 'MAM': None, 'JJA': None, 'SON': None }

    #  First month, last month. 

    t0 = datetime( year=timerange[0].year, month=timerange[0].month, day=1 )
    t1 = datetime( year=timerange[1].year, month=timerange[1].month, day=1 )

    #  Loop over season. 

    firsttime = True

    for season in ubar.keys(): 

        if season=='DJF': startmonth = 12
        if season=='MAM': startmonth = 3
        if season=='JJA': startmonth = 6
        if season=='SON': startmonth = 9

        year0 = t0.year
        t = datetime( year=t1.year, month=startmonth, day=1 ) + timedelta(days=62)
        t = datetime( year=t.year, month=t.month, day=1 )
        if t.year <= t1.year: 
            year1 = t1.year
        else: 
            year1 = t1.year - 1

        nyears = 0

        for year in np.arange(year0,year1+1): 
            t = datetime( year=year, month=startmonth, day=1 )

            nmonths = 0
            for imonth in range(3): 

                #  Is the time interval already available? 

                var = 'u'
                timeptr = None
                fileptr = None

                timeptr = m.get_data( var, t )
                if timeptr is None: break

                #  Initialization: get the longitudes, latitudes. 

                m.initialize( timeptr )
                nlons, nlats = m.lons.size, m.lats.size

                #  Initialization. 

                if firsttime: 
                    mask = np.zeros( (nlats,nlons), dtype='i' ) + 1
                    mask = np.logical_and( mask, m.oceanmask )
                    firsttime = False

                if imonth==0: 
                    annualaverage = np.zeros( nlats )

                if year == year0: 
                    average = np.zeros( nlats )

                #  Execute zonal average time average. 

                za = np.zeros( nlats )
                n = mask.sum(axis=1)
                good = ( n > 0 )
                za[good] = ( timeptr.data * mask ).sum( axis=1 )[good] / n[good]

                #  Add to annual average. 

                annualaverage = annualaverage + za
                nmonths = nmonths + 1

                #  Increment time. 

                t += timedelta( days=31 )
                t = datetime( year=t.year, month=t.month, day=1 )

            #  Annual average. 

            if nmonths==3: 
                annualaverage = annualaverage / 3
                average = average + annualaverage
                nyears = nyears + 1

        #  Average over years. 

        ubar[season] = average / nyears

    #  Return. 

    return m.lats, ubar

