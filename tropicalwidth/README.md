#  TropicalWidth

This package contains software and a few command line executables related to 
inferring the time history of the width of the Tropics using the CCMP 
surface winds reanalysis. The basis for this research is that the null in 
zonal mean zonal wind near the edge of the Tropics precisely identifies 
the latitude where the downwelling branch of the Hadley cell diverges 
equatorward and one side and poleward on the other. 

Many useful classes and parameters can be found in _tropicalwidth.libtropicalwidth_, 
especially the three classes Model, FilePtr, and TimePtr. 

Three executables currently exist: _process\_oscar_, _process\_ccmp_, and 
_process\_cmip6_. _process\_oscar_ creates monthly climatologies of OSCAR
surface current vectors downloaded from the PO-DAAC; _process\_ccmp_ 
computes timeseries of the surface air zonal wind nulls from the CCMP surface 
air winds data product; and _process\_cmip6_ does the same but for CMIP6 AMIP
model output. 

