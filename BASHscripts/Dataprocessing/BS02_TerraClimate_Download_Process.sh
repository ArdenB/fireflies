#!/bin/sh
for i in {1958..2019};
do
        #merge the files using cdo
        echo $i
        #  Fetch the files 
        if (( i > 1963)); then
                wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_tmax_$i.nc 
                wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_tmin_$i.nc
                wget -nc -c -nd https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_$i.nc 
        fi

        # +++++ process the temperature +++++
        cdo -b 32 -ensmean TerraClimate_tmax_${i}.nc TerraClimate_tmin_${i}.nc TerraClimate_tmean_${i}.nc
        cdo sellonlatbox,-10,180,40,70 -setname,tmean TerraClimate_tmean_${i}.nc TerraClimate_SIBERIA_tmean_${i}.nc
        cdo splitname TerraClimate_SIBERIA_tmean_${i}.nc TerraClimate_SIBERIA_tmean_${i}_

        # +++++ process the precipitation +++++
        cdo splitname -sellonlatbox,-10,180,40,70 TerraClimate_ppt_${i}.nc TerraClimate_SIBERIA_ppt_${i}_

        # Remove the excess files to free up space 
        rm TerraClimate_SIBERIA_tmean_${i}_station_influence.nc TerraClimate_tmean_${i}.nc TerraClimate_SIBERIA_ppt_${i}_station_influence.nc
done

# +++++Merge into one netcdf file +++++
#  tmean
cdo mergetime TerraClimate_SIBERIA_tmean_* TerraClimate_SIBERIA_stacked_tmean_1958to2018.nc
# ppt
cdo mergetime TerraClimate_SIBERIA_ppt_* TerraClimate_SIBERIA_stacked_ppt_1958to2018.nc

# rm TerraClimate_SIBERIA_tmean_*.nc
# rm TerraClimate_SIBERIA_ppt_*.nc


