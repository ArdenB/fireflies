

# For loop
# FILES=./*
for year in {1999..2017}; do
	#statements
	echo NDVI_monthlymax_${year}_S01.nc
	cdo -b F64 -yearmax -sellonlatbox,100,120,40,53 NDVI_monthlymax_${year}_S01.nc NDVI_yearmax_Russia_${year}_S01.nc
done



cdo mergetime NDVI_yearmax_Russia_*.nc NDVI_anmax_Russia.nc
# Get the simple regression slope
cdo regres NDVI_anmax_Russia.nc NDVI_anmax_Russia_cdoregres.nc 
