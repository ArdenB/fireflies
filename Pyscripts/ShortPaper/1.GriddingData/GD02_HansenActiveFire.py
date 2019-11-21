
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "Hansen Active fire"
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
from shapely.geometry import Polygon
import geopandas as gpd
from rasterio import features
from affine import Affine
# import fiona as fi
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
	spath = pathfinder()

	# ========== Make the files ==========
	fnames = ActiveFireMask(spath, force)


	# ========== Loop over the datasets ==========
	HansenMasker(fnames, ymin=2002, ymax=2010,)
	ipdb.set_trace()

#==============================================================================

def HansenMasker(fnames, ymin=2001, ymax=2019, **kwargs):
	"""Takes a list of file names and masks the hansen data"""
	# ========== load the hansen ==========\
	region = "SIBERIA"
	# ========== Setup the loop ==========
	ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/HANSEN"
	ft    = "lossyear"

	tpath = "/home/ubuntu/Documents/fireflies/data/tmp/"
	cf.pymkdir(tpath)

	# ========== Create the outfile name ==========
	fpath = "%s/%s/" % (ppath, ft)
	fnout = "%sHansen_GFC-2018-v1.6_%s_%s.nc" % (fpath, ft, region)
	# ========== Open the dataset ==========
	ds = xr.open_dataset(fnout, chunks={'latitude': 10000, 'longitude':10000})
	lmax = int(np.round(ds.latitude.values.max()))
	lmin = int(np.round(ds.latitude.values.min()))
	window = -1

	for yr, fn in zip(range(ymin, ymax), fnames):
		Fyout = fntmp  =  tpath+"HansenActiveFire_%d.nc" % (yr)

		if not os.path.isfile(Fyout) or force:
			date  = datefixer(yr, 12, 31)
			# ========== Load the results ==========
			afr    = gpd.read_file(fn)
			# maskre = rm.Regions_cls("AFY",[0],["activefire"], ["af"],  afr.geometry)
			shapes = [(shape, n+1) for n, shape in enumerate(afr.geometry)]

			# ========== empty container for the files ==========
			filenames = []


			# ========== Chunk specific sections ==========
			for lm in range(lmax, lmin, window):
				print(yr, lm, pd.Timestamp.now())
				fntmp  =  tpath+"HansenActiveFire_%d_%d.nc" % (yr, lm)
				if not os.path.isfile(fntmp): 
					def _dsSlice(fnout, yr, lm, window, shapes, afr):
						# ========== open the file ==========
						ds_in = xr.open_dataset(fnout, chunks={'latitude': 1000}).sel(
							dict(latitude =slice(int(lm), int(lm)+window)))#.compute()


						# ========== build a mask ==========
						# mask = maskre.mask(dsbool.longitude.values, dsbool.latitude.values)
						# ipdb.set_trace()
						transform = transform_from_latlon(ds_in['latitude'], ds_in['longitude'])
						out_shape = (len(ds_in['latitude']), len(ds_in['longitude']))
						raster    = features.rasterize(shapes, out_shape=out_shape,
						                            fill=0, transform=transform,
						                            dtype="int16", **kwargs)

						# ========== build a boolean array ==========
						raster = raster.astype(bool)
						with ProgressBar():
							dsbool = (ds_in == (yr-2000) or ds_in == (yr-2000+1)).compute()

						dsbool *= raster

						# ========== Save the file out ==========
						encoding = ({"lossyear":{'shuffle':True,'zlib':True,'complevel':5}})

						dsbool.to_netcdf(fntmp, format = 'NETCDF4',encoding=encoding, unlimited_dims = ["time"])


					_dsSlice(fnout, yr, lm, window, shapes, afr)
				filenames.append(fntmp)

			# ========== open multiple files at once ==========
			dsout = xr.open_mfdataset(filenames, concat_dim="latitude")
			
			# ========== Set the date ==========
			dsout["time"] = date["time"]

			# ========== rename the variable to somehting sensible ==========
			dsout = dsout.rename({"lossyear":"fireloss"})			

			# Check its the same size as the sliced up ds
			# ========== Save the file out ==========
			encoding = ({"fireloss":{'shuffle':True,'zlib':True,'complevel':5}})
			print ("Starting write of combined data for %d at:" % yr, pd.Timestamp.now())

			with ProgressBar():
				dsout.to_netcdf(Fyout, format = 'NETCDF4',encoding=encoding, unlimited_dims = ["time"])
			# cleanup the excess files

			for fnr in filenames:
				if os.path.isfile(fnr):
					os.remove(fnr)
		

		# with ProgressBar():
		# 	dsbool = dsbool.compute()
		
		# ipdb.set_trace()


	ipdb.set_trace()

#==============================================================================
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
def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def ActiveFireMask(spath, force, ymin=2001, ymax=2019):
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
	path  = spath + "BurntArea/MODIS_ActiveFire/DL_FIRE_M6_85602/fire_archive_M6_85602.shp"
	


	fnames = [spath + "BurntArea/MODIS_ActiveFire/AnnualActiveFire%d.shp" % yr for yr in range(ymin, ymax)]

	if not all([os.path.isfile(fn) for fn in fnames]) or force:
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


def pathfinder():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-CSHARFM':
		# LAPTOP
		spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/"

	elif sysname == "owner":
		spath = "/mnt/d/Data51/"
	elif sysname == "ubuntu":
		# Work PC
		spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/"

	return spath

	

#==============================================================================

if __name__ == '__main__':
	main()