#  TropicalWidth

This package contains software and a few command line executables related to 
inferring the time history of the width of the Tropics using the CCMP 
surface winds reanalysis. The basis for this research is that the null in 
zonal mean zonal wind near the edge of the Tropics precisely identifies 
the latitude where the downwelling branch of the Hadley cell diverges 
equatorward and one side and poleward on the other. 

Many useful classes and parameters can be found in _tropicalwidth.libtropicalwidth_, 
especially the three classes Model, FilePtr, and TimePtr. 

Two executables currently exist: _process_ and _figures_. The former processes 
(and downloads) CCMP data for the northern and southern edges of the Tropics; 
the latter generates diagnostics encapsulated postscript figures for use in 
presentation and publication. 

