#  Instructions for Tropical Width Analysis

Author: Stephen Leroy (stephen.leroy@janusresearch.us)

Date: 17 October 2025


## Prerequisites

- You will need a large amount of scratch space for the data downloads, 
  around 900 GB. Be certain to use attached storage if you do not have 
  sufficient disk space. The root path for the project data 
  should be put into the environment variable **DATAROOT** to make the 
  rest of this work go easily.
```
export DATAROOT="/path/to/data"
```
- You will need to create a conda environment in which to work to
  perform all calculations and generate plots.
```
conda env create -f config.yaml -y
conda activate tropical-width
```
- You will need an account on the Copernicus Data Store in order to access 
  and download ERA5 data. Once done, create the file .cdsapirc in your home 
  directory and write the url and key values for the CDS API into it. You 
  can copy the contents from [here](https://cds.climate.copernicus.eu/how-to-api). 

- You will also need an account in NASA Earthdata, which is a gateway to the 
  NASA Distributed Active Archive Systems, in order to download OSCAR ocean 
  surface current data. Once done, you will have to write your NASA Earthdata 
  username and password into the file .netrc in your home directory; the 
  machine is "urs.earthdata.nasa.gov". 

##  Computations 

First, a few settings to smooth the analysis from the Linux shell to the 
jupyter notebook that does the graphical analysis.

```
export ERA5ANALYSIS="tropicalwidth_era5.nc"
export CMIP6ANALYSIS="tropicalwidth_cmip6.nc"
```

1. Download OSCAR data. This is the ocean surface current data. The 
download is done by 
```
download_oscar 1995-01:2024-12
```
It is a very time consuming operation because the NASA Physical Oceanography 
DAAC manifests the data on the server side before making it visible to the 
user.

2. Create monthly climatologies of OSCAR data. 
```
process_oscar
```

3. Download ERA5 data. The streamfunction analysis is done with this 
download. 
```
download_era5 1995:2024
```

4. Process ERA5 streamfunction data. The output analysis goes into 
$ERA5ANALYSIS. 
```
process_era5 1995:2024 $ERA5ANALYSIS
```

5. Analyze CMIP6 model data. The output analysis goes into 
$CMIP6ANALYSIS. 
```
process_cmip6 $CMIP6ANALYSIS
```

6. Analyze CCMP surface air wind data. The CCMP data are downloaded 
automatically, making this an extremely time consuming analysis. Note 
that the processing goes much more quickly if the CCMP data files are 
already present in $DATAROOT/ccmp. Otherwise, it is **highly**
recommended that this step be done in a parallel processing environment.
In meta-code, consider dividing up the job by year, producing one
CCMP analysis for each year: 
```
for year in { 1995 .. 2024 }  # Parallelize this loop
do
  process_ccmp "$(year)-01:$(year)-12" $DATAROOT/output/tropicalwidth.$(year).nc
done
```
The research at Janus/AER was done in a cloud-computing environment, 
specifically with the Amazon Web Services (AWS) Batch service. A very 
similar approach can be taken with SLURM or PBL/Torque on a 
high-performance Linux cluster. The jupyter notebook below is very 
capable of handling the multiple output files produced. 

## Trends and figures

The linear regression trend analysis and the figures are generated 
by the jupyter notebook streamfunction.ipynb and the python script 
figures-v2.py. The former notebook consists of three cells; only the first 
two need be executed. The latter  consists of several functions, 
some of which perform linear regression analyses, others of which actually 
generate diagnostic plots and the figures contained in the paper. 
```
jupyter notebook streamfunction.ipynb
python figures-v2.py
```

