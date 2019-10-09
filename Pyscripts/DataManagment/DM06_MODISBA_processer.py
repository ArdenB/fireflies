"""
Script goal, 

Process MODIS burned areas for short paper 

"""
#==============================================================================

__title__ = "data fetcher"
__author__ = "Arden Burrell"
__version__ = "v1.0(11.09.2019)"
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

import ipdb
# import wget
# import glob
# import tarfile  
import xarray as xr
import numpy as np
import pandas as pd
import warnings as warn
import datetime as dt
from dask.diagnostics import ProgressBar
from collections import OrderedDict
from netCDF4 import Dataset, num2date, date2num 
# import gzip

# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 

#==============================================================================

def main():
	# ========== Setup the path ==========
	path = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/"
	drop = ["crs", "Burn_Date_Uncertainty", "First_Day", "Last_Day", "QA"]

	# ========== open the file ==========
	ds = xr.open_dataset(path+"MCD64A1.006_500m_aid0001.nc").drop(drop).rename(
		{"lon":"longitude", "lat":"latitude"}).sel(
		dict(time=slice("2001-01-01", "2018-12-31"))).chunk({"time":12})

	# ========== Process the file ==========
	ds_bool = ds>0 
	ds_BA   = ds_bool.groupby("time.year").any(dim="time").rename({"year":"time", "Burn_Date":"BA"})
	ds_BA["time"] =  [pd.Timestamp("%d-12-31" % yr) for yr in ds_BA.time.values]

	# ========== create the filname ==========
	fnout = path + "MODIS_MCD64A1.006_500m_aid0001_reprocessedBA.nc"

	# ========== Fix the metadata ==========
	GlobalAttributes(ds_BA, fnout)	

	# ========== Create the encoding ==========
	encoding = OrderedDict()
	encoding["BA"] = ({
		'shuffle':True, 
		# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
		'zlib':True,
		'complevel':5})
	
	# ========== Write the file out ==========
	delayed_obj = ds_BA.to_netcdf(fnout, 
		format         = 'NETCDF4', 
		encoding       = encoding,
		unlimited_dims = ["time"],
		compute=False)

	print("Starting write of BA data at",  pd.Timestamp.now())
	with ProgressBar():
		results = delayed_obj.compute()

	ipdb.set_trace()


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
	attr["FileName"]            = fnout
	attr["title"]               = "BurnedArea"
	attr["summary"]             = "Reprocessed MODIS Burned Area" 
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
#==============================================================================
if __name__ == '__main__':
	main()