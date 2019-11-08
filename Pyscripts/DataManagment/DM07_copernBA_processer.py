"""
Script goal, 

Build evaluation maps of GEE data

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
import glob
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
	# ========== setup the path ========== 
	path   = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/COPERN_BA/"
	ppath  = path+"processed/"

	indata = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/M0044633/"
	cf.pymkdir(path)
	cf.pymkdir(path+"tmp/")
	cf.pymkdir(ppath)
	force  = True# False
	dsn    = "COPERN_BA"
	
	latmax = 70.0
	latmin = 40.0
	lonmin = -14.0
	lonmax =  180.0
	shapes = []

	for yr in range(2015, 2019):
		# ========== loop over layers ==========		
		print("Starting %d at:" % (yr), pd.Timestamp.now())

		fnames = glob.glob(indata+"c_gls_BA300_%d*.nc" % yr)

		# ========== Check if a file already exists ==========
		ANfn    = ppath + "%s_gls_%d_burntarea.nc" % (dsn, yr)
		if os.path.isfile(ANfn) and not force:
			ds = xr.open_dataset(ANfn)
		else:
			ds = annualfile(yr, path, ppath, force, ANfn, latmax, latmin, lonmin, lonmax, dsn, fnames)
			print("%d BA calculation completed at:" % (yr), pd.Timestamp.now(), " Starting Cleanup")
		shapes.append(ds["BA"].shape)
		print(shapes)
		# ipdb.set_trace()
	ipdb.set_trace()

#==============================================================================

def annualfile(yr, path, ppath, force, ANfn, latmax, latmin, lonmin, lonmax, dsn, fnames):
	
	# ========== rapidly stack results ==========
	files = None
	nx    = 0 
	t0    = pd.Timestamp.now()
	mask  = None

	for fn in fnames:
		ds_yin = xr.open_dataset(fn).rename({"lon":"longitude", "lat":"latitude"}).sel(dict(
			latitude =slice(latmax, latmin), 
			longitude=slice(lonmin, lonmax))).drop(["CP_DEKAD", "crs", "FDOB_DEKAD", "BA_DEKAD"]).rename({"FDOB_SEASON":"BA"})#.chunk({'latitude':1000, 'longitude': 1000})

		ds_out = (ds_yin >((yr - 1980) * 1e3))

		if files is None:
			files=ds_out
		else:
			ipdb.set_trace()
			sys.exit()
			try:
				files = xr.concat([files, ds_out], dim="time").any(dim="time")
			except Exception as e:
				files = xr.concat([files, ds_out.isel(time=0, drop=True)], dim="time").any(dim="time")
		
		# ========== Implement the mask here ==========
		if yr == 2018 and mask is None:
			mask = ds_yin.where(ds_yin == 254, 1).where(ds_yin != 254).where(~np.isnan(ds_yin)).rename({"BA":"mask"})
		
		# ========== print the output here ==========
		nx +=1
		t1  = pd.Timestamp.now()
		ave = (t1-t0)/nx
		string = "\r %d of %d complete, average time %s, eta %s" % (
			nx, len(fnames), str(ave), str(ave*(len(fnames)-nx)))
		sys.stdout.write(string)
		sys.stdout.flush()

	# ds_yin = xr.open_mfdataset(fnames, concat_dim="time").rename({"lon":"longitude", "lat":"latitude"}).sel(dict(
	# 	latitude =slice(latmax, latmin), 
	# 	longitude=slice(lonmin, lonmax))).drop(["CP_DEKAD", "crs", "FDOB_DEKAD", "BA_DEKAD"]).rename({"FDOB_SEASON":"BA"}).chunk({'latitude':1000, 'longitude': 1000})


	# ========== convert to boolean array ==========
	ds_out = files.astype("float32").expand_dims({"time":[pd.Timestamp("%d-12-31" % yr)]})
	ds_out.attrs = ds_yin.attrs
	GlobalAttributes(ds_out, ANfn)


	ds_out = tempNCmaker(ds_out, ANfn, "BA", chunks={'latitude':1000, 'longitude': 1000}, skip=False, pro = "Annual BA data")

	# ========== Implement the mask here ==========
	if yr == 2018:
		# ds_yin = ds_yin.where(ds_yin != 254)
		maskfn = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/%s/FRI/%s_landseamask.nc" %(dsn, dsn)
		# print("I need to add a mask here")
		mask = mask.expand_dims({"time":[pd.Timestamp("%d-12-31" % yr)]})
		# mask.attrs = ds_yin.attrs
		GlobalAttributes(mask, maskfn)

		mask = tempNCmaker(mask, maskfn, "mask", chunks={'latitude':1000, 'longitude': 1000}, skip=False, pro = "Mask")
		ipdb.set_trace()

	return ds_out

#==============================================================================
def tempNCmaker(ds, fntmp, vname, chunks={'longitude': 1000}, skip=False, pro = "tmp"):

	""" Function to save out a tempary netcdf """
	# cf.pymkdir(tmppath)
	
	encoding =  ({vname:{'shuffle':True,'zlib':True,'complevel':5}})
	if not all([skip, os.path.isfile(fntmp)]):
		delayed_obj = ds.to_netcdf(fntmp, 
			format         = 'NETCDF4', 
			encoding       = encoding,
			unlimited_dims = ["time"],
			compute=False)

		print("Starting write of %s data at" % pro, pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()
	dsout = xr.open_dataset(fntmp, chunks=chunks) 
	return dsout
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
	attr["summary"]             = "Reprocessed copern Burned Area" 
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