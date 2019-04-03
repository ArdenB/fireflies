"""
Data Fixing script

upen multiple files 

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
	"""for files in list"""

	# flist = glob.glob("/media/ubuntu/Seagate Backup Plus Drive/Data51/NDVI/4.CGLS/M0039243/processed/Monthly/NDVI_monthlymax_*_S01.nc")
	dele = []
	path = "/media/ubuntu/Seagate Backup Plus Drive/Data51/NDVI/4.CGLS/M0039243/processed/Monthly/"
	for year in range(1999, 2019):

		fname = path+"NDVI_monthlymax_%d_S01.nc" % year
		fout  = path+"NDVI_AnnualMax_%d_S01.nc" % year

		subp.call(
			"cdo -P 2 yearmax %s %s" % 
			(fname, fout),
			shell=True
			)
		dele.append(fout)

	ipdb.set_trace()
	# ========== Open the dataset as a multifile dataset ==========
	# ds = xr.open_mfdataset(flist)
	# yearmax = ds.NDVI.groupby("time.year").max("time")



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