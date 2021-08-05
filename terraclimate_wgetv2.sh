#!/bin/bash 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1958.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1959.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1960.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1961.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1962.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1963.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1964.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1965.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1966.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1967.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1968.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1969.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1970.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1971.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1972.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1973.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1974.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1975.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1976.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1977.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1978.nc 
wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_1979.nc 



# Combine the files

# remove the other vairalbes


cdo -b 64 -remapbil,../../../griddes/GIMMS_globalgrid TerraClimate_merged_1980to2017_ppt.nc TerraClimate_merged_1980to2017_ppt_GIMMS.nc



cdo -b 64 -ensmean TerraClimate_tmax_1980.nc TerraClimate_tmin_1980.nc TerraClimate_tmeanv2_1980.nc