#!/usr/bin/env python3

import re
import argparse
import boto3
from tropicalwidth.libtropicalwidth import default_dataroot

session = boto3.Session( region_name="us-east-1" )
batch = session.client( service_name="batch", endpoint_url="https://batch.us-east-1.amazonaws.com" )

if __name__ == "__main__": 

    #  Define command line parser. 

    parser = argparse.ArgumentParser( description="Submit a batch job to process tropical width" )

    parser.add_argument( "yearrange", type=str, help="""The (inclusive) range of 
            years to process tropical width based on surface winds of CCMP, a 
            string of format "YYYY:YYYY" """ )
    parser.add_argument( "--dataroot", default=default_dataroot, 
            help=f'The root path for the data; the default is "{default_dataroot}"' )
    parser.add_argument( "--test", "-t", dest="test", default=False, action="store_true", 
            help='Run the program in test mode, generating messages indicating what jobs would be run' )
    parser.add_argument( "--clobber", "-c", dest="clobber", default=False, action="store_true", 
            help='Clobber previously existing output files; do not clobber by default' )

    #  Parse command line. 

    args = parser.parse_args()

    #  Compose list of jobs. 

    m = re.search( r'(\d{4}):(\d{4})$', args.yearrange )
    if m: 
        year0, year1 = int( m.group(1) ), int( m.group(2) )
    else: 
        print( 'Input yearrange must have format "YYYY:YYYY".' )
        exit()

    cmds, jobNames = [], []

    for year in range(year0,year1+1): 

        cmd = [ 'process', 
                f'{year:4d}-01:{year:4d}-12', 
                f'{args.dataroot}/output/tropicalwidth.{year:4d}.nc' ]
        if args.clobber: 
            cmd.append( "--clobber" )
        jobName = f'process-{year:4d}'

        cmds.append( cmd )
        jobNames.append( jobName )

    if args.test: 
        print( 'Jobs not being actually submitted to Batch...' )

    for jobName, cmd in zip( jobNames, cmds ): 

        print( '{:}: {:}'.format( jobName, " ".join( cmd ) ) )

        if not args.test: 
            response = batch.submit_job( 
                    jobName=jobName, 
                    jobQueue='rd-tropical-width', 
                    jobDefinition='rd-tropical-width', 
                    containerOverrides={'command':cmd, 'vcpus':1, 'memory':2000 } )

