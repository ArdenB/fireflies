
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "FRI calculator for the other datasets"
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

#==============================================================================

def main():
	# ========== Setup the paths ==========
	data = datasets()
	
	# ========== select and analysis scale ==========
	mwbox     = [1, 2, 5]#, 1, 10] #in decimal degrees
	# BPT       = 0.4
	force     = False

	for dsn in data:
		# ========== Set up the filename and global attributes =========
		ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/%s/FRI/" %  dsn
		cf.pymkdir(ppath)
		
		# ========== Get the dataset =========
		ds, mask = dsloader(data, dsn, ppath, force)

		# ========== Calculate the annual burn frewuency =========
		ds_ann = ANNcalculator(data, dsn, ds, mask,force, ppath)


		# ========== work out the FRI ==========
		FRIcal(ds_ann, mask, dsn, force, ppath, mwbox, data)

		force = True



		ipdb.set_trace()


#==============================================================================
def FRIcal(ds_ann, mask, dsn, force, ppath, mwbox, data):
	""""""
	""" Function to caluclate the FRI at different resolutions """
	
	# ========== Setup a working path ==========
	tpath = ppath+"tmp/"
	
	# ========== work out the ration ==========
	pix    =  abs(np.unique(np.diff(ds_ann.latitude.values))[0]) 
	ds_ann = ds_ann.chunk({"latitude":1000, "longitude":-1})

	# ========== Build a cleanup list ==========
	cleanup = []
	for mwb in mwbox:
		print("Starting %s %d degree moving window at:" %(dsn, mwb), pd.Timestamp.now())
		fname = "%s_annual_burns_MW_%ddegreeBox.nc" % (dsn, mwb)
		if os.path.isfile(ppath+fname) and not force:
			cleanup.append(ppath+tname)
			continue
		# ===== get the ratio =====
		SF = np.round(mwb /pix).astype(int)

		tname = "%s_annual_burns_lonMW_%ddegreeBox.nc" % (dsn, mwb)

		# ===== Create a masksum =====
		# This is so i can count the number of values that are valid in each location
		mask_sum = mask.rolling({"longitude":SF}, center = True, min_periods=1).sum()
		mask_sum = mask_sum.rolling({"latitude":SF}, center = True, min_periods=1).sum()

		# ===== Calculate the Moving window =====
		dsan_lons = ds_ann.rolling({"longitude":SF}, center = True, min_periods=1).mean() 
		# warn.warn("Implement some form of masking here")
		dsan_lons = dsan_lons.where(mask["mask"].values == 1)

		# ========== Mask out bad pixels ==========
		# ds_con = ds_con.where(mask.mask.values == 1)
		dsan_lons = tempNCmaker(dsan_lons, tpath, tname, "AnBF", {'latitude': 1000}, readchunks={'longitude': 1000}, skip=False)

		# ===== Calculate the Moving window in the other dim =====
		ds_out = dsan_lons.rolling({"latitude":SF}, center = True, min_periods=1).mean() 
		ds_out = ds_out.where(mask["mask"].values == 1) #Mask out water
		ds_out = ds_out.where(mask_sum["mask"].values > ((SF/2)**2)) #Mask out points that lack data
		ds_out = ds_out.where(ds_out > 0)
		
		# ===== Calculate a FRI =====
		ds_out["FRI"] = 1.0/ds_out["AnBF"]

		# ===== Save the file out =====
		ds_out = tempNCmaker(ds_out, ppath, fname, "AnBF", {'longitude': 1000}, readchunks=data[dsn]["chunks"], skip=False)

		cleanup.append(ppath+tname)

		if mwb ==  1:
			ipdb.set_trace()

	warn.warn("I need to implement a cleanup here")
	ipdb.set_trace()
	for file in  cleanup:
		if os.path.isfile(file):
			os.remove(file)
	ipdb.set_trace()

def ANNcalculator(data, dsn, ds, mask,force, ppath):
	""" Function to calculate the FRI 
	args
		data: 	Ordered dict
		dsn:	str of the dataset name
		ds:		XR dataset
	"""

	# ========== Setup a working path ==========
	tpath = ppath+"tmp/"
	cf.pymkdir(tpath)

	# ======================================================
	# ========== Build the annual mean burnt area ==========
	# ======================================================
	
	# ========== setup the temp filnames ==========
	tname = "%s_annual_burns.nc" % dsn


	if not os.path.isfile(tpath+tname) or force:
		# ========== calculate the sum ==========
		dates   = datefixer(2018, 12, 31)
		ds_flat = ds.mean(dim="time", keep_attrs=True).expand_dims({"time":dates["CFTime"]}).rename({data[dsn]["var"]:"AnBF"})
		ds_flat.time.attrs["calendar"]   = dates["calendar"]
		ds_flat.time.attrs["units"]      = dates["units"]

		# ========== Write out the file ==========
		attrs = GlobalAttributes(ds_flat, dsn, fnameout=tpath+tname)

		# ========== add some form of mask here ==========
		ds_flat = ds_flat.where(mask["mask"].values == 1)

		ds_flat = tempNCmaker(
			ds_flat, tpath, tname, "AnBF", 
			data[dsn]["chunks"], skip=False)
	
	else:
		print("Opening existing temp file")
		ds_flat = xr.open_dataset(tpath+tname, chunks=data[dsn]["chunks"])

	return ds_flat

#==============================================================================
#==============================================================================
def tempNCmaker(ds, tmppath, tmpname, vname, writechunks, readchunks={'longitude': 1000}, skip=False):

	""" Function to save out a tempary netcdf """
	cf.pymkdir(tmppath)
	
	fntmp    = tmppath + tmpname
	encoding =  ({vname:{'shuffle':True,'zlib':True,'complevel':5}})
	if not all([skip, os.path.isfile(fntmp)]):
		delayed_obj = ds.to_netcdf(fntmp, 
			format         = 'NETCDF4', 
			encoding       = encoding,
			unlimited_dims = ["time"],
			compute=False)

		print("Starting write of temp data at", pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()
	dsout = xr.open_dataset(fntmp, chunks=readchunks) 
	return dsout

def dsloader(data, dsn, ppath, force):
	"""Takes in infomation about datasets and loads a file
	args
		data: Ordered dict
		dsn:	str of the dataset name
	returns:
		ds
	"""
	# ========== check if the name needs to be globbed  ==========
	if "*" in data[dsn]["fname"]:
		# ========== get all the file names ==========
		fnames = glob.glob(data[dsn]["fname"])
		lat = []
		for fn in fnames:
			dsin = xr.open_dataset(fn, chunks=data[dsn]["chunks"])
			print(dsin["BA"].shape[1] )
			lat.append(dsin.latitude.values)

		# ========== open the dataset ==========
		ds = xr.open_mfdataset(fnames, concat_dim="time", chunks=data[dsn]["chunks"])
		# ipdb.set_trace()
	else:
		ds = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
	
	mask = landseamaks(data, dsn, ppath, ds, force )

	return ds, mask

def landseamaks(data, dsn, ppath, ds, force, chunks=None ):

	# ========== create the mask fielname ==========
	masknm = "%s_landseamask.nc" % dsn

	if dsn == "esacci":
		chunks = data[dsn]["chunks"]
		raw_mask = xr.open_dataset(ppath+masknm, chunks=chunks)
		ipdb.set_trace()
		return raw_mask

	# ========== check if the mask already exists ==========
	if os.path.isfile(ppath+masknm) and not force:
		raw_mask = xr.open_dataset(ppath+masknm, chunks=chunks)
	else:
		print("Building a new mask for %s at:" % dsn, pd.Timestamp.now())
		
		dates    = datefixer(2018, 12, 31)
		# ========== load the modis mask ==========
		raw_mask = xr.open_dataset(data[dsn]["maskfn"])
		if dsn == "MODIS":
			raw_mask = raw_mask.drop(
				["QC", "crs"]).chunk({"time":1}).isel(time=-1).rename(
				{"LW":"mask","lon":"longitude", "lat":"latitude"}).expand_dims({"time":dates["CFTime"]})
		if not chunks is None:
			raw_mask = raw_mask.chunk(chunks)
		
		# ===== regrid to match the new dataset =====
		if not dsn == "MODIS":
			ipdb.set_trace()
			attrs = raw_mask.attrs.copy()
			raw_mask = raw_mask.reindex({"latitude":ds.latitude, "longitude":ds.longitude}, method="nearest")

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
		raw_mask = tempNCmaker(
			raw_mask, ppath, masknm, "mask", 
			data[dsn]["chunks"], readchunks=chunks, skip=False)


	return raw_mask

def datasets():
	# ========== set the filnames ==========
	data= OrderedDict()
	# data["GIMMS"] = ({
	# 	"fname":"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_annualmax.nc",
	# 	'var':"ndvi", "gridres":"8km", "region":"global", "timestep":"Annual", 
	# 	"start":1982, "end":2017, "rasterio":False, "chunks":{'time': 36},
	# 	"rename":None
	# 	})
	# data["COPERN"] = ({
	# 	'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
	# 	'var':"NDVI", "gridres":"1km", "region":"Global", "timestep":"AnnualMax",
	# 	"start":1999, "end":2018,"rasterio":False, "chunks":{'time':1}, 
	# 	"rename":{"lon":"longitude", "lat":"latitude"}
	# 	})
	data["COPERN_BA"] = ({
		'fname':"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/COPERN_BA/processed/COPERN_BA_gls_*.nc",
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":None, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})
	data["MODIS"] = ({
		"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBA.nc",
		'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1,'longitude': 1000, 'latitude': 10000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001.nc"
		})
	data["esacci"] = ({
		"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_FireCCI_*_burntarea.nc",
		'var':"BA", "gridres":"250m", "region":"Asia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1,'longitude': 1000, 'latitude': 10000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_landseamask.nc"
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	return data

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

#==============================================================================

def GlobalAttributes(ds, dsn, fnameout="",):
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
	attr["FileName"]            = ""
	attr["title"]               = "FRI"
	attr["summary"]             = "BorealForestFRI_%sData" % (dsn)
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. FRI caluculated using %s data" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__, dsn)
	
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