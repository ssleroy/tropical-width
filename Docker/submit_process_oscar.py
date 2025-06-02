#!/usr/bin/env python3

import argparse
import boto3

session = boto3.Session( region_name="us-east-1" )
batch = session.client( service_name="batch", endpoint_url="https://batch.us-east-1.amazonaws.com" )

def submit_process_oscar( dataroot="/fg", clobber=False ): 

    cmd = [ 'process_oscar', '--dataroot', dataroot ]
    if clobber: 
        cmd.append( '--clobber' )

    jobName = 'process_oscar'

    print( '{:}: "{:}"'.format( jobName, " ".join( cmd ) ) )

    response = batch.submit_job( 
                    jobName=jobName, 
                    jobQueue='rd-tropical-width', 
                    jobDefinition='rd-tropical-width', 
                    containerOverrides={'command':cmd, 'vcpus':1, 'memory':2000 } )

def main(): 

    default_dataroot = "/fg"

    parser = argparse.ArgumentParser( description="""Submit AWS batch jobs to generate 
            monthly climatologies of OSCAR daily surface current files.""" )
    parser.add_argument( "--dataroot", default=default_dataroot, 
            help=f'Path to project data root; "{default_dataroot}" by default' )
    parser.add_argument( "--clobber", default=False, action="store_true", 
            help='Clobber/overwrite previously existing output files; do not clobber by default' )

    args = parser.parse_args()
    ret = submit_process_oscar( args.dataroot, args.clobber )


if __name__ == "__main__": 
    main()
    pass

