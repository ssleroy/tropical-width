#!/usr/bin/env python3

import re
import argparse
import boto3
from tropicalwidth.libtropicalwidth import default_dataroot

session = boto3.Session( region_name="us-east-1" )
batch = session.client( service_name="batch", endpoint_url="https://batch.us-east-1.amazonaws.com" )

if __name__ == "__main__": 

    cmd = [ 'process_cmip6', '--dataroot', '/fg', '--clobber', '/fg/output/tropicalwidth_cmip6.nc' ]
    jobName = f'process_cmip6'

    print( '{:}: {:}'.format( jobName, " ".join( cmd ) ) )

    response = batch.submit_job( 
            jobName=jobName, 
            jobQueue='rd-tropical-width', 
            jobDefinition='rd-tropical-width', 
            containerOverrides={'command':cmd, 'vcpus':1, 'memory':2000 } )

