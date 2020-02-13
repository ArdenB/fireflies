
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "Hansen Active fire version 2"
__author__ = "Arden Burrell"
__version__ = "v1.0(20.11.2019)"
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
#==============================================================================
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
from dask.distributed import Client
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats

import shutil
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import itertools

# Import debugging packages 
import ipdb

# ========== Import specific packages  ==========
# from rasterio.warp import transform
# from shapely.geometry import Polygon
# import geopandas as gpd
# from rasterio import features
# from affine import Affine
# # import fiona as fi
# import regionmask as rm
# import matplotlib.path as mplPath
from rasterio import features
from affine import Affine
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================

def main():
	# ==========
	force = False
	
	# ========== Get the path ==========
	dpath, chunksize = syspath()

	# client = Client(n_workers=4, threads_per_worker=2, memory_limit='15GB')
	client = None

	# ========== Make the files ==========
	# fnames = ActiveFireMask(dpath, force, client,)
	fnames = None
	
	# ========== Build some netcdf versions ==========
	fns_NC = MODIS_shptoNC(dpath, fnames, force, client, ymin=2001, ymax=2019, dsn = "esacci")

	# ========== Resample the Hansen ==========
	fly_nm = Hansen_resizer(dpath, force, fns_NC, client, ymin=2001, ymax=2019, dsn = "esacci")
	ipdb.set_trace()
	# ========== Loop over the datasets ==========
	# HansenMasker(fnames, force, dpath, ymin=2001, ymax=2019, )


#==============================================================================

def Hansen_resizer(dpath, force, fns_NC, client, ymin=2001, ymax=2019, dsn = "esacci"):
	"""
	take the Hansen data and resize it to match the datagrid
	"""
	forestlossnm = []
	temp_filesnm = []
	cf.pymkdir(dpath + "/BurntArea/HANSEN/lossyear/tmp/")
	# ========== Loop over the datasets ==========	
	force = True
	warn.warn("I've got a manual force in place, i will need to turn this of asap \n\n")
	for yr, fnx in zip(range(ymin, ymax), fns_NC):
		if yr < 2011
		# ========== load in the hansen file ==========
		fname = dpath + "/BurntArea/HANSEN/lossyear/Hansen_GFC-2018-v1.6_lossyear_SIBERIA.nc"
		fnout = dpath + "/BurntArea/HANSEN/lossyear/Hansen_GFC-2018-v1.6_%d_totalloss_SIBERIAat%s.nc" % (yr, dsn)
		fntmp = dpath + "/BurntArea/HANSEN/lossyear/tmp/tmp_Hansen_GFC-2018-v1.6_%d_totalloss_SIBERIAat%s.nc" % (yr, dsn)
		
		if os.path.isfile(fnout) and not force:
			print("Valid existing value for %d:" % yr, pd.Timestamp.now())
			forestlossnm.append(fnout)
			continue
		
		ds_in = xr.open_dataset(fname)#, chunks={'latitude': 999, 'longitude':999})
		ds_in = ds_in.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))
		ds_in = ds_in.chunk({'latitude': 10000, 'longitude':20000})

		# ========== Open the resolution dataset ===========
		ds_res = xr.open_dataset(fnx, chunks={'latitude': 100, 'longitude':100})

		# ========== Find the places that were lost in each year ===========
		ds_BOOL = (ds_in == yr-2000).astype("float32")
		ds_BOOL["time"] = ds_res.time # time fix
		ds_BOOL = tempNCmaker(
			ds_BOOL, fntmp, "lossyear", client, chunks={'latitude': 9999, 'longitude':999}, 
			skip=True, name="%d Forest Bool " % yr)
		temp_filesnm.append(fntmp)

		# ========== Calculate the bounding box ===========
		def _Scalingfactors(ds_in, ds_res):
			# +++++ Find the resolution +++++
			res_lat = abs(np.unique(np.round(np.diff(ds_res.latitude.values), decimals=8)))[0]
			din_lat = abs(np.unique(np.round(np.diff(ds_in.latitude.values), decimals=6))[0])

			res_lon = abs(np.unique(np.round(np.diff(ds_res.longitude.values), decimals=8)))[0]
			din_lon = abs(np.unique(np.round(np.diff(ds_in.longitude.values), decimals=6))[0])
			
			# +++++ Find the scale factors +++++
			SF_lat = int(np.round(res_lat/din_lat))
			SF_lon = int(np.round(res_lon/din_lon))

			return SF_lat, SF_lon
		
		SF_lat, SF_lon = _Scalingfactors(ds_in, ds_res)
		print("Start coarsening and reindex for %d at:" % yr, pd.Timestamp.now())
		with ProgressBar():
			ds_out = ds_BOOL.coarsen(
				dim={"latitude":SF_lat, "longitude":SF_lon}, boundary="pad").mean(
				).reindex_like(ds_res, method="nearest")#.compute()

		# ========== Fix the attributes ==========
		ds_out.attrs = ds_in.attrs.copy()
		ds_out.attrs["title"]   = "Regridded Annual Forest Loss"
		ds_out.attrs["summary"] = "Forest loss for each year downsampled to workable resolution using xr.coarsen" 
		
		# ++++++++++ Data Provinance ++++++++++ 
		ds_out.attrs["history"] = "%s: Netcdf file created using %s (%s):%s by %s" % (
			str(pd.Timestamp.now()), __title__, __file__, __version__, __author__) + ds_out.attrs["history"]

		# ++++++++++ Write the data out ++++++++++ 
		fnout = dpath + "/BurntArea/HANSEN/lossyear/Hansen_GFC-2018-v1.6_%d_totalloss_SIBERIAat%s.nc" % (yr, dsn)
		ds_out = tempNCmaker(
			ds_out, fnout, "lossyear", client, chunks={'latitude': 1000, 'longitude':1000}, 
			skip=False, name="%d Forest loss " % yr)
		forestlossnm.append(fnout)

	ipdb.set_trace()
	return forestlossnm, temp_filesnm
	

# def function():
# 	pass

def MODIS_shptoNC(dpath, fnames, force, client, ymin=2001, ymax=2019, dsn = "esacci"):
	""" 
	Function to open the shape file and get the values into a raster at a workable 
	resolu
	"""
	# ========== Open the Raster dataset ==========
	fname = dpath + "/masks/landwater/%s_landwater.nc" % dsn
	ds_in = xr.open_dataset(fname, chunks={'latitude': 10000, 'longitude':1000})
	ds_in = ds_in.sortby("latitude", ascending=False)
	ds_in = ds_in.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))

	fnames_nc = []
	
	# ========== Open the vector dataset ==========	
	for yr in range(ymin, ymax):

		print("Starting the boolean netcdf creation for %d at:" %  yr, pd.Timestamp.now())
		# ========== Setup the year ==========
		date  = datefixer(yr, 12, 31)

		fnout = dpath + "/BurntArea/MODIS_ActiveFire/AnnualActiveFire_%d_%s_res.nc" % (yr, dsn)

		if force or not os.path.isfile(fnout):

			# creat the file name of the results
			fn = dpath + "/BurntArea/MODIS_ActiveFire/AnnualActiveFire%d.shp" % yr

			# load the shape
			afr    = gpd.read_file(fn)
			shapes = [(shape, n+1) for n, shape in enumerate(afr.geometry)]

			transform = transform_from_latlon(ds_in['latitude'], ds_in['longitude'])
			out_shape = (len(ds_in['latitude']), len(ds_in['longitude']))
			raster    = features.rasterize(shapes, out_shape=out_shape,
			                            fill=0, transform=transform, dtype="int16")
			raster = raster.astype(bool)

			ds_out = ds_in.copy(data={"landwater":raster}).rename({"landwater":"ActiveFire"})
			ds_out.attrs["title"]       = "ActiveFireData"
			ds_out.attrs["summary"]     = "Activefire" 
			ds_out.attrs["Conventions"] = "CF-1.7"
			ds_out["time"]              = date["time"]
			
			# ++++++++++ Data Provinance ++++++++++ 
			ds_out.attrs["history"]             = "%s: Netcdf file created using %s (%s):%s by %s" % (
				str(pd.Timestamp.now()), __title__, __file__, __version__, __author__) + ds_out.attrs["history"]

			# ++++++++++ Write the data out ++++++++++ 
			ds_out = tempNCmaker(ds_out, fnout, "ActiveFire", client, chunks={'latitude': 10000, 'longitude':1000}, skip=False, name="%d Active FIre " % yr)

		fnames_nc.append(fnout)
	return fnames_nc

def ActiveFireMask(dpath, force, client, ymin=2001, ymax=2019):
	def _annualfire(actfire, yr):
		# ========== Convert to an equal area projection ==========
		print("starting equal area reprojection at:	", pd.Timestamp.now())
		actfire = actfire.to_crs({'init': 'epsg:3174'})

		# ========== Add a 4km buffer ==========
		print("starting buffer at:				", pd.Timestamp.now())
		actfire["geometry"] = actfire.geometry.buffer(4000)

		# ========== Convert back to projection ==========
		print("starting latlon reprojection at:   	", pd.Timestamp.now())
		actfire = actfire.to_crs({'init': 'epsg:4326'})

		# ========== Disolve by year ==========
		print("starting dissolve at:			", pd.Timestamp.now())
		actfire = actfire.dissolve(by='year')

		# ========== Save the results ==========
		print("starting data write at:			", pd.Timestamp.now())
		actfire.to_file(outfn)
	
	# ========== Build the results ==========
	path  = dpath + "/BurntArea/MODIS_ActiveFire/DL_FIRE_M6_85602/fire_archive_M6_85602.shp"
	


	fnames = [dpath + "/BurntArea/MODIS_ActiveFire/AnnualActiveFire%d.shp" % yr for yr in range(ymin, ymax)]

	if not all([os.path.isfile(fn) for fn in fnames]) or force:
		ipdb.set_trace()


		# ========== Load the data ==========
		print("starting active fire data read at:	", pd.Timestamp.now())
		afr = gpd.read_file(path)

		# ========== Add a new column ==========
		years = pd.Series([pd.Timestamp(date).year for date in afr.ACQ_DATE.values])
		afr["year"] = years
	
		# ========== Subset the data ==========
		afr = afr[afr.CONFIDENCE >= 30]
		afr = afr[['year', 'geometry']]

		# ========== Export the reActive fire masks ==========
		for yr, outfn in zip(range(ymin, ymax), fnames):
			if not os.path.isfile(outfn) or force:	
				print("Building new active fire shapefile for ", yr)
				# ========== Pull out the relevant year ==========
				actfire = afr[afr.year == yr].copy()

				# ========== Make the new shapefile ==========
				_annualfire(actfire, yr)

				# ========== Free the memory ==========
				actfire = None


			else:
				print("Shapefile already exists for ", yr)

	# ========== Load the data ==========
	return fnames

#==============================================================================
def tempNCmaker(ds, fnout, vname, client, chunks={'latitude': 10000, 'longitude':1000}, skip=False, name="tmp"):

	""" Function to Quickly save netcdfs"""
	
	if type(vname) == list:
		encoding = OrderedDict()
		for vn in vname:
			encoding[vn] = {'shuffle':True,'zlib':True,'complevel':5}
	else:
		encoding =  ({vname:{'shuffle':True,'zlib':True,'complevel':5}})
	if not all([skip, os.path.isfile(fnout)]):
		delayed_obj = ds.to_netcdf(fnout, 
			format         = 'NETCDF4', 
			encoding       = encoding,
			unlimited_dims = ["time"],
			compute=False)

		print("Starting write of %s data at" % name, pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()
	dsout = xr.open_dataset(fnout, chunks=chunks) 
	return dsout


def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def datasets(dpath, chunksize):
	# ========== set the filnames ==========
	data= OrderedDict()
	data["COPERN_BA"] = ({
		'fname':dpath+"/masks/landwater/",
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":{'time':1, 'longitude': chunksize, 'latitude': chunksize}, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})

	data["MODIS"] = ({
		"fname":dpath+"/BurntArea/MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc",
		'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1,'longitude': chunksize, 'latitude': chunksize},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"
		})
	data["esacci"] = ({
		"fname":dpath+"/BurntArea/esacci/processed/esacci_FireCCI_*_burntarea.nc",
		'var':"BA", "gridres":"250m", "region":"Asia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': chunksize, 'latitude': chunksize},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_landseamask.nc"
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})


def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-UA7CT9Q':
		# dpath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h/Data51"
		# clpath = "/mnt/d/Data51/climate/TerraClimate"
		dpath = "/mnt/d/Data51"
		chunksize = 50
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51"
		dpath = "/media/ubuntu/Harbinger/Data51"
		chunksize = 50
		
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
	elif 'ccrc.unsw.edu.au' in sysname:
		dpath  = "/srv/ccrc/data51/z3466821"
		# clpath = "/srv/ccrc/data51/z3466821/Input_data/TerraClimate"
		chunksize = 20
	else:
		ipdb.set_trace()
	return dpath, chunksize	

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

if __name__ == '__main__':
	main()