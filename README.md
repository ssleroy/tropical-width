# Widening of the Tropics in CCMP

This repository contains the software used to analyze the width of the Tropics according 
to latitudes of the nulls in zonal mean zonal winds in the CCMP surface air wind analysis. 
The physics is simple: the boundary of the tropical trade winds and the surface westerlies 
indicates the poleward boundary of the tropical Hadley cell. The wind reverse direction at 
this boundary because of the Coriolis force and meridional divergence of the flow at the 
edge of the Hadley cell within the planetary boundary layer. 

## Structure

The heavy computing is done in the cloud, specifically Amazon Web Services (AWS). You'll 
need to have access to an S3 bucket, FileGateway, EC2, and Batch services. What processing 
can be done on the local workstation is found in the "python" directory. The processing 
to be done in the cloud is in the "Docker" directory. 

The heart of the heavy computing is found in the local PyPI-style package 
__Docker/tropicalwidth__. It is straightforward Python code that can be installed on the 
workstation or in a Docker image. Both are done here. Be sure to always work in the 
Python environment "tropical-width". 

Trend analysis and figure production is done in a Jupyter notebook: __docker/figures.ipynb__. 

## Implementation

First, this software takes advantage of cloud computing in Amazon Web Services (AWS) and so 
requires some editing according to the user's particular AWS account. Edit the 
Docker/tropicalwidth/src/tropicalwidth/libtropicalwidth.py file and change the name of the 
"bucket" (the name of the S3 bucket where you're storing data) and the name of the AWS 
"account". The latter is necessary for your own authentication purposes. Also, edit the 
Docker/build.sh and the Docker/push.sh according to your AWS account in order to have it 
correspond to your instance of the AWS Elastic Container Repository (ECR). 

Second, build the environment for local and cloud use. For local use, 
```
conda env create -f config.yaml
```
For the AWS cloud, 
```
./build.sh
./push.sh
```
Whenever working in a shell, be sure to activate the environment: 
```
conda activate tropical-width
```

Third, download OSCAR data. This, unfortunately, must be done manually, because of the 
NASA Physical Oceanography DAAC (PODAAC). The data should find their way into the S3 "bucket" 
as follows: f's3://{bucket}/oscar/{year:4d}/{month:02d}/oscar_currents_final_{year:4d}{month:02d}{day:02d}.nc' 
in which year, month, and day are all integers. Depending on the state of OSCAR processing, it may be 
necessary to replace "final" with "nrt". You will then have to compute monthly-average climatologies 
of OSCAR data using AWS Batch: 
```
cd Docker
python submit_process_oscar.py
```

Fourth, download CMIP6 data. Again, this must be done manually. It is only necesary to download 
the zonal component of the surface air winds, monthly averages, for the AMIP-mode runs. The data should be 
loaded into the S3 "bucket" as follows: f's3://{bucket}/amip/...'. The file naming and directory 
structure is completely up to the user. 

Fifth, download ERA5 data. Be sure to have an account set up with the ECMWF Copernicus Service and 
place the authentication tokens in "~/.cdsapirc". Run the command 
```
cd python
python download_era5.py
```

Sixth, process all CCMP data for the tropical width using AWS Batch. There is a lot of computing involved; 
hence, parallel processing becomes convenient. 
```
cd Docker
python submit_process.py "1995:2024"
```

Seventh, process ERA5. This can be done on the local workstation without the advantage of AWS cloud 
computing. Execute the command, 
```
cd python
python era5.py
```

Eighth, process CMIP6. This, too, is done in AWS Batch. 
```
cd Docker
python run_process_cmip6.py
```

Finally, perform the analysis, plotting, etc., using a jupyter notebook: 
```
cd python
jupyter notebook figures.ipynb
```
