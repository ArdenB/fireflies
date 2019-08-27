
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "Hansen Product Merger"
__author__ = "Arden Burrell"
__version__ = "v1.0(21.08.2019)"
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
import pandas as pd
import argparse
import datetime as dt
from collections import OrderedDict
import warnings as warn
from netCDF4 import Dataset, num2date, date2num 
from scipy import stats
import rasterio
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats
import glob
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import regionmask as rm
import itertools
# Import debugging packages 
import ipdb
# from rasterio.warp import transform
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio import features
from affine import Affine
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf


def main():
	# ========== Create the mask dates ==========
	dates  = datefixer(2018, 12, 31)
	force  = True
	region = "SIBERIA"

	# DAin, global_attrs = dsloader(data, dsn, dates)

	# ========== Setup the loop ==========
	ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/HANSEN"
	pptex = ({"treecover2000":"FC2000", "lossyear":"lossyear", "datamask":"mask"})

	# ========== open the forest cover file ==========
	for ft in pptex:
		# ========== Create the outfile name ==========
		fpath = "%s/%s/" % (ppath, pptex[ft])
		fnout = "%sHansen_GFC-2018-v1.6_%s_%s.nc" % (fpath, ft, region)
		if os.path.isfile(fnout) and not force:
			print("dataset for %s %03d %02d already exist. going to next chunk" % (ft, LatM))
			continue
		
		# ========== Stripe the lat bands ==========
		DA_LC = [_lat_concat(LatM, fpath, dates, ft) for LatM in range(70, 50, -10)]
		DA    = xr.concat(DA_LC, dim="latitude")

		# ========== Fix the metadata ==========
		DA           = attrs_fixer(DA, ft, dates)
		global_attrs = GlobalAttributes(None, ft)

		# ========== Start making the netcdf ==========
		layers     = OrderedDict()
		layers[ft] = DA
		
		encoding   = OrderedDict()
		enc = ({'shuffle':True,
			# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
			'zlib':True,
			'complevel':5})
		encoding[ft] = enc
		
		# ========== Create the dataset ==========
		ds = xr.Dataset(layers, attrs= global_attrs)
		ds.attrs["FileName"] = fnout

		# ========== Save it as a netcdf file ==========
		delayed_obj = ds.to_netcdf(fnout, 
			format         = 'NETCDF4', 
			encoding       = encoding,
			unlimited_dims = ["time"],
			compute=False)

		print("Starting write of data at", pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()

		ipdb.set_trace()	

	ipdb.set_trace()

#==============================================================================
#============================ Internal function ===============================
#==============================================================================

def _lat_concat(LatM, fpath, dates, ft):
	"""Function to open all the individual lat rows """
	# ========== lOAD THE Hansen Forest GFC ==========
	gpath   = "%sHansen_GFC-2018-v1.6_%s_%02dN_*E.tif" %(fpath, ft, LatM)
	fn_in   = glob.glob(gpath)
	da_list = [_daload(fn, dates) for fn in fn_in]
	# ========== Concatinate the results into a single dataarray ==========
	return xr.concat(da_list, dim="longitude")

def _daload(fn, dates):
	"""Short function to open the geotifs and fix metadata """
	# Load the file
	da = xr.open_rasterio(fn)
	# rename the bands
	da = da.rename({"band":"time", "x":"longitude", "y":"latitude"}) 
	# chunk the file
	da = da.chunk({'latitude': 1000})   #, "longitude":1000
	# convert the time
	da["time"] = dates["CFTime"]
	
	return da

#==============================================================================

def attrs_fixer(da, sname, dates):
	"""
	Function to fix the metadata of the dataarrays
	args:
		origda:	XR datarray, 
			the original meta data
		da: xa dataarray
			the dataarray to be fixed
		vname: str
			name of the variable
	retunrs
		da: xr dataarray with fixed metadata
	"""
	# ========== Make the DA ==========
	da.attrs['_FillValue']	=-1, #9.96921e+36
	da.attrs['units'     ]	="1",
	da.attrs['standard_name']	= sname 
	da.attrs['long_name'	]	= "BorealForest_%s" % sname
	
	if sname == "treecover2000":
		da.attrs['valid_range']	= [-1, 100.0]	
	else:
		da.attrs['valid_range']	= [0.0, 1.0]
	
	da.longitude.attrs['units'] = 'degrees_east'
	da.latitude.attrs['units']  = 'degrees_north'
	
	da.time.attrs["calendar"]   = dates["calendar"]
	da.time.attrs["units"]      = dates["units"]
	return da

def datefixer(year, month, day):
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
	# +++++ set up the list of dates +++++
	dates = OrderedDict()
	tm = [dt.datetime(int(year) , int(month), int(day))]
	dates["time"] = pd.to_datetime(tm)

	dates["calendar"] = 'standard'
	dates["units"]    = 'days since 1900-01-01 00:00'
	
	dates["CFTime"]   = date2num(
		tm, calendar=dates["calendar"], units=dates["units"])

	return dates

def GlobalAttributes(ds, dsn, fname=""):
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
	if ds is None:
		attr = OrderedDict()
	else:
		attr = ds.attrs

	# fetch the references for my publications
	# pubs = puplications()
	
	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["FileName"]            = fname
	attr["title"]               = "BorealForest2000forestcover"
	attr["summary"]             = ("Hansen Forest loss %s tile merge" % (dsn))
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = '''%s: Netcdf file created using %s (%s):%s by %s. Original data aquired from:
	https://storage.googleapis.com/earthenginepartners-hansen/GFC-2018-v1.6/Hansen_GFC-2018-v1.6_datamask_*N_*E.tif''' % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)
	
	if not ds is None:
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

#==============================================================================
if __name__ == '__main__':
	main()