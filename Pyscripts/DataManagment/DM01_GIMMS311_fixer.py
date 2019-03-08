"""
Data Fixing script

Gimms data ships with a terrible calander.  

"""
#==============================================================================

__title__ = "Gimms Fixer"
__author__ = "Arden Burrell"
__version__ = "v1.0(25.02.2019)"
__email__ = "arden.burrell@gmail.com"

#==============================================================================
# +++++ Check the paths and set ex path to fireflies folder +++++
import os
import sys
if not os.getcwd().endswith("fireflies"):
	if "fireflies" in os.getcwd():
		p1, p2, _ =  os.getcwd().partition("fireflies")
		os.chdir(p1+p2)
	else:
		raise OSError(
			"This script was called from an unknown path. CWD can not be set"
			)
sys.path.append(os.getcwd())


#==============================================================================
# Import packages
import numpy as np
import glob
import pandas as pd
import argparse
import datetime as dt
from collections import OrderedDict
import warnings as warn
from netCDF4 import Dataset, num2date, date2num 
from scipy import stats
import subprocess as subp
import xarray as xr
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================


def main():

	filelist = NDVIfilefix()

	# outfile  = " ./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_merged.nc" 


	# subp.call(
	# 	"cdo -P 4 -b F6 mergetime %s %s" % ( 
	# 		" ./data/veg/GIMMS31g/GIMMS31v1/timecorrected/".join(filelist), outfile),
	# 	shell=True
	# 	)
	ipdb.set_trace()



#==============================================================================
def NDVIfilefix(force=False):
	"""
	Goal of the script:
		1. Fix the broken dates
		2. Create a new percentile value
		3. Pull out the quality flages
	"""
	# ========== Loop over the years ==========
	fixed = []
	for dec in range(1980, 2020, 10):
		# loop over each decade
		for year in range(dec, dec+10):
			print(year)
			# loop over each year
			fnames = glob.glob(
				"./data/veg/GIMMS31g/GIMMS31v1/geondvi3g%d*/geo8km_v1_1/ndvi3g_geo_v1_1_%d_*.nc4" %(
					dec, year))
			if fnames == []:
				print("No Valid results for %d" % year)
				continue

			for fname in fnames:
				# ========== Open the file ==========
				# fname = "./data/veg/GIMMS31g/GIMMS31v1/geondvi3g1980sv11/geo8km_v1_1/ndvi3g_geo_v1_1_1981_0712.nc4"
				ds = xr.open_dataset(fname)
				fnout = ds.attrs["FileName"][:-4]+"_timecorrected.nc"
				if os.path.isfile("./data/veg/GIMMS31g/GIMMS31v1/timecorrected/" + fnout) and not force:
					print("Valid file exists for %d" % year)
					continue 
				
				# ========== FIx the time ========== 
				dates = datefixer(ds)

				layers   = OrderedDict()
				encoding = OrderedDict()
				
				# ========== Recode the values  ==========
				layers['ndvi'], encoding['ndvi'] = NDVIscaler(ds, dates)
				layers['flag'], encoding['flag'], layers['percentile'], encoding['percentile'] = PercFlag(ds, dates)
				layers['satellites'], encoding['satellites'] = sat(ds, dates)

				# ========== Create the global attributes ==========
				global_attrs = GlobalAttributes(ds, fnout)

				ds_out = xr.Dataset(layers, attrs= global_attrs)
				print("Starting write of data")
				ds_out.to_netcdf("./data/veg/GIMMS31g/GIMMS31v1/timecorrected/" + fnout, 
					format         = 'NETCDF4', 
					encoding       = encoding,
					unlimited_dims = ["time"])
				print(".nc file created for %d" % year)
				fixed.append(fnout)
	return fixed
	
def sat(ds, dates):
	"""
	fix the dates on the sensors
	args:
		ds:xr dataset
			dataset containing values
		dates: dict
			dictionary form the time correction function
	returns:
		DA: xr dataarray
		enc: encoding dictionary

	"""
	sat = ds.satellites.values.astype(float)
	DA=xr.DataArray(sat,
		dims = ['time'], 
		coords = {'time': dates["CFTime"]},
		attrs = ({
			'_FillValue':-1, #9.96921e+36
			# 'units'     :"1",
			'standard_name':"satellites",
			'long_name':"satellites",
			# 'scale': 1,
			# 'valid_range': [bn.nanmin(Val), np.ceil(np.nanmax(Val))]
			}),
	)

	# DA.longitude.attrs['units'] = 'degrees_east'
	# DA.latitude.attrs['units']  = 'degrees_north'
	DA.time.attrs["calendar"]   = dates["calendar"]
	DA.time.attrs["units"]      = dates["units"]
	
	encoding = ({'shuffle':True, 
		# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
		'zlib':True,
		'complevel':5})
	
	return DA, encoding


def PercFlag(ds, dates):
	"""
	Takes the NDVI values and rescales them 
	args:
		ds:xr dataset
			dataset containing values
		dates: dict
			dictionary form the time correction function
	returns:
		DA: xr dataarray
		enc: encoding dictionary
	"""

	# ========== Set up the layer params ==========
	kyp        = 'percentile'
	long_namep = "percentile"
	
	ky        = 'flag'
	long_name = "flag"

	fill_val        = -1.0

	# ========== pull out the flag ==========
	# percentile = 2000*flag + perc
	flag = np.floor(ds.percentile.values / 2000)
	# actual percentile = perc / 10
	perc = (ds.percentile.values - flag*2000)/10.0

	# ========== Build the netcdf ==========
	DA, enc = DAbuilder(flag, ds, dates, ky, long_name, fill_val)
	DAp, encp = DAbuilder(perc, ds, dates, kyp, long_namep, fill_val)

	return DA, enc, DAp, encp

def NDVIscaler(ds, dates):
	"""
	Takes the NDVI values and rescales them 
	args:
		ds:xr dataset
			dataset containing values
		dates: dict
			dictionary form the time correction function
	returns:
		DA: xr dataarray
		enc: encoding dictionary
	"""

	# ========== Set up the layer params ==========
	ky        = 'ndvi'
	long_name = "normalized_difference_vegetation_index"
	fill_val        = -1.0

	# ========== Modify the ndvi values ==========
	ndvi            = ds.ndvi.values.astype(float)
	ndvi            /= 10000
	ndvi[ndvi<-0.3] = np.NAN


	# ========== Start making the netcdf ==========
	DA, enc = DAbuilder(ndvi, ds, dates, ky, long_name, fill_val)

	return DA, enc

def DAbuilder(Val, ds, dates, ky, long_name, fill_val):
	"""
	Args: 
		Val: array
			np array for shape time, lat, lon with values
		ds: xarray ds
			ds being processsed
		dates: dict
			dictionary of the dataset infomation
		ky: str
			short name
		long_name: str
			the CF longname
		fill_value:
			fill value of the array

	"""
	
	# ========== get the lat and lon ==========
	lat       = ds.lat.values
	lon       = ds.lon.values

	# ========== Create the xr DA ==========
	try:
		# build xarray dataset
		DA=xr.DataArray(Val,
			dims = ['time', 'latitude', 'longitude'], 
			coords = {'time': dates["CFTime"],'latitude': lat, 'longitude': lon},
			attrs = ({
				'_FillValue':fill_val, #9.96921e+36
				'units'     :"1",
				'standard_name':ky,
				'long_name':long_name,
				# 'scale': 1,
				'valid_range': [bn.nanmin(Val), np.ceil(np.nanmax(Val))]
				}),
		)

		DA.longitude.attrs['units'] = 'degrees_east'
		DA.latitude.attrs['units']  = 'degrees_north'
		DA.time.attrs["calendar"]   = dates["calendar"]
		DA.time.attrs["units"]      = dates["units"]
		
		encoding = ({'shuffle':True, 
			# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
			'zlib':True,
			'complevel':5})
	
		return DA, encoding
	except Exception as e:
		warn.warn("Code failed with: \n %s \n Going Interactive" % e)
		ipdb.set_trace()
		raise e

def datefixer(ds):
	"""
	Opens a netcdf file and fixes the data, then save a new file and returns
	the save file name
	args:
		ds: xarray dataset
			dataset of the xarray values
	return
		time: array
			array of new datetime objects
	"""


	# ========== create the new dates ==========
	year = ds.Year

	# +++++ set up the list of dates +++++
	dates = OrderedDict()
	tm = [dt.datetime(int(year) , int(np.floor(tm)), int(tm%1*30+1)) for tm in ds.time]
	dates["time"] = pd.to_datetime(tm)

	dates["calendar"] = 'standard'
	dates["units"]    = 'days since 1900-01-01 00:00'
	
	dates["CFTime"]   = date2num(
		tm, calendar=dates["calendar"], units=dates["units"])

	return dates

def GlobalAttributes(ds, fnout):
	"""
	Creates the global attributes for the netcdf file that is being written
	these attributes come from :
	https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/metadata/DataDiscoveryAttConvention.html
	args
		ds: xarray ds
			Dataset containing the infomation im intepereting
		fnout: str
			filename out 
	returns:
		attributes 	Ordered Dictionary cantaining the attribute infomation
	"""
	# ========== Create the ordered dictionary ==========
	attr = ds.attrs

	# fetch the references for my publications
	# pubs = puplications()
	
	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["FileName"]           = fnout
	attr["title"]               = "NDVI"
	attr["summary"]             = "Reprocessed GIMMS31v1.1 NDVI" 
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)
	attr["history"]            += ds.history

	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["Institution"]         = "University of Leicester"
	attr["date_created"]        = str(pd.Timestamp.now())
	
	# ++++++++++ Netcdf Summary infomation ++++++++++ 
	# attr["time_coverage_start"] = str(dt.datetime(ds['time.year'].min(), 1, 1))
	# attr["time_coverage_end"]   = str(dt.datetime(ds['time.year'].max() , 12, 31))
	return attr

if __name__ == '__main__':
	main()