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
	# drop = ["crs", "Burn_Date_Uncertainty", "First_Day", "Last_Day", "QA"]
	# fin  = path+"MCD64A1.006_500m_aid0001.nc"

	force = False



	drop = ["crs", "QA"]
	fin  = path+"MCD64A1.006_500m_aid0001v2.nc"
	if not os.path.isfile(fin) or force:
		# ========== open the file ==========
		ds = xr.open_dataset(fin).drop(drop).rename(
			{"lon":"longitude", "lat":"latitude"}).sel(
			dict(time=slice("2001-01-01", "2018-12-31"))).chunk({"time":12})

		# ========== Process the file ==========
		ds_bool = ds>0 
		ds_BA   = ds_bool.groupby("time.year").any(dim="time").rename({"year":"time", "Burn_Date":"BA"})
		ds_BA["time"] =  [pd.Timestamp("%d-12-31" % yr) for yr in ds_BA.time.values]

		# ========== create the filname ==========
		# fnout = path + "MODIS_MCD64A1.006_500m_aid0001_reprocessedBA.nc"
		fnout = path + "MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc"

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
	else:
		ds_BA = xr.open_dataset(fin)

	# ========== Build the mask ==========
	mask = maskmaker(force)
	ipdb.set_trace()

#==============================================================================

def maskmaker(force, dsn="MODIS"):

	print("Building a new mask for %s at:" % dsn, pd.Timestamp.now())
	# ========== Set up the paths ==========
	ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/%s/FRI/" %  dsn
	cf.pymkdir(ppath)

	masknm = ppath + "MODIS_landseamask.nc"

	if not os.path.isfile(masknm) or force:
		
		# ========== create a date ==========
		dates    = datefixer(2018, 12, 31)

		# ========== load the modis mask ==========
		maskfn = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"
		raw_mask = xr.open_dataset(maskfn)

		# ========== Start on the mask ==========
		raw_mask = raw_mask.drop(
			["QC", "crs"]).chunk({"time":1}).isel(time=-1).rename(
			{"LW":"mask","lon":"longitude", "lat":"latitude"}).expand_dims({"time":dates["CFTime"]})

		# ===== Create a boolean mask =====
		raw_mask = raw_mask - 1.0
		raw_mask = raw_mask.where(raw_mask==1)

		# ===== fix the time =====
		raw_mask["time"] = dates["CFTime"]
		raw_mask.time.attrs["calendar"]   = dates["calendar"]
		raw_mask.time.attrs["units"]      = dates["units"]

		# ===== add to the creation history =====
		raw_mask.attrs["history"] = "%s: Netcdf file created using %s (%s):%s by %s. Grid matches %s data. " % (
			str(pd.Timestamp.now()), __title__, __file__, __version__, __author__, dsn) + raw_mask.attrs["history"]
		

		# ===== save the file out =====
		raw_mask = tempNCmaker(raw_mask, masknm, "mask")

	else:
		raw_mask = xr.open_dataset(masknm)
	return raw_mask

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