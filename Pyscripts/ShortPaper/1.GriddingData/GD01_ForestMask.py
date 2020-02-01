
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "Hansen GFC FRI calculator"
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
# import rasterio
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
from scipy import stats
import subprocess as subp

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
# import regionmask as rm
import itertools

from cdo import Cdo
# Import debugging packages 
import ipdb
# from rasterio.warp import transform
# from shapely.geometry import Polygon
# import geopandas as gpd
# from rasterio import features
# from affine import Affine
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf


def main():
	"""
	The goal of this script is to produce three masks.
	1. The landscape scale mask based on similar rules
	2. The pixel perfect mask based on rules
	3. A simple landsea mask based on rules 
	"""
	# ========== Setup the paths ==========
	ft     = "treecover2000"
	region = "SIBERIA"
	force  =  False
	# force  =  True
	

	# ========== Create the mask dates ==========
	dates  = datefixer(2000, 1, 1)
	nfval  = 0.0  # TC Value considered not forest
	minTC  = 10.0 # Minimum Tree cover
	maxNF  = 0.30 # Max fraction of non forest

	# ========== load in the datasets ==========
	dpath = syspath()
	ppath = dpath + "/Data51/BurntArea/HANSEN"
	data  = datasets(dpath)
	
	# ========== Create the broad mask ==========
	broadMask(dpath, region, minTC, nfval, force, data, dates)

	# ========== Create the landsea mask ==========
	landwatermask(dpath, region, maxNF, nfval, force, data)
	
	# ========== Create the original masks ==========
	warn.warn("Stuff beyond this point i have not vetted as well")
	ds    = HansenCoarsenLoad(ppath, region, maxNF, nfval, force, data)

	# ds    = ds.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-15, 180.0)))
	ipdb.set_trace()

	# ========== Loop over the datasets ==========
	cleanup = []
	for dsn in data:
		print(dsn)
		# ========== Set up the filename and global attributes =========
		
		fpath     = "./data/other/ForestExtent/%s/" % dsn
		storepath = dpath + "/Data51/ForestExtent/%s/" % dsn
		for dirr in [fpath, storepath]:
			cf.pymkdir(dirr)
			cf.pymkdir(dirr+"tmp/")
		
		# ========== Load the grids =========
		DAin, global_attrs = dsloader(data, dsn, dates)

		# ========== subset the dataset in to match the forest cover ==========
		ds = ds.sel(dict(
			latitude=slice(DAin.latitude.max().values, DAin.latitude.min().values), 
			longitude=slice(DAin.longitude.min().values, DAin.longitude.max().values)))
		
		# ========== generate the new datasets ==========
		out, tmp = _dsroller(fpath, storepath, ds, DAin, dsn, data, maxNF, force, region, global_attrs, dates)
		cleanup.append(tmp)

		# ========== Implement moving of file ==========
		warn.warn("File move needs to occour here")
		ipdb.set_trace()
		if not storepath in out:
			shutil.move(out, out.replace(fpath, storepath))
	ipdb.set_trace()

	print("Starting excess file cleanup at:", pd.Timestamp.now())
	for fn in cleanup:
		if os.path.isfile(tmp):
			os.remove(tmp)

#==============================================================================
# The middle grids
#==============================================================================
def landwatermask(dpath, region, maxNF, nfval, force, data):
	"""
	Logic build a basic land sea datamask at every resolution
		if a dataset has its own mask great, if not just use the MODIS one
	"""
	print("starting Landwater Masking at:", pd.Timestamp.now())
	ppath = dpath + "/Data51/masks/landwater"
	cf.pymkdir(ppath)
	cf.pymkdir(ppath+"/tmp")


	# =========== MODIS gets processed first ===========
	def _modislandwater(ppath, force):
		# THIS FUNCTION IS JUST FOR MODIS
		if not os.path.isfile(ppath+"/MODIS_landwater.nc") or force:
			rawmaskfn = dpath +"/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"

			# +++++ load the data +++++
			ds_Rmask = xr.open_dataset(rawmaskfn, chunks={"time":1}).rename({"lat":"latitude", "lon":"longitude"})

			# +++++ get the values +++++
			with ProgressBar():
				ds_mask = (
					ds_Rmask.isel(time=-1).drop(
						["QC", "crs"]).rename({"LW":"landwater"}) == 2.0).astype("float32").compute()
			# ========== fix the metadata ==========
			ds_mask.attrs = GlobalAttributes(ds_Rmask, "MODIS", fname="", title="LandWaterMask")
			ds_mask.attrs["summary"] = "Land water mask for BA data"
			dts = datefixer(2018, 12, 31)
			ds_mask["time"] = dts["time"]

			ds_tmp = tempNCmaker(ds_mask.coarsen(latitude=3, longitude=3).median(), ppath+"/tmp/MODIS_landwater.nc", "landwater", 
				pchunk=None, chunks=None, skip=False, 
				pro = "MODIS land water mask")
			
			ds_mask = ds_mask.where(ds_mask==1)

			ds = tempNCmaker(ds_mask, ppath+"/MODIS_landwater.nc", "landwater", 
				pchunk=None, chunks=None, skip=False, 
				pro = "MODIS land water mask")
		else:
			ds = xr.open_dataset(ppath+"/MODIS_landwater.nc", chunks= {'latitude': 1000, 'longitude':1000})
		return ds
	# ===== Fetch the modis mask =====
	ds_mod = _modislandwater(ppath, force)

	# ========== Loop over each dataset ==========
	for dsn in data:
		if dsn == 'MODIS':
			# The broad mask is alread at this resolution
			continue
		elif dsn in ["TerraClimate", "COPERN", "GIMMS"]:
			print("Upscaling %s started at:" % dsn, pd.Timestamp.now())

			def _upscaler(ppath, dsn, data, ds_mod, force):
				# Functiuon for scaling up datasets
				# ds_ref = xr.open_dataset(data[dsn]["fname"])

				fntmp = ppath+"/tmp/%s_landwater.nc" % dsn
				fnout = ppath+"/%s_landwater.nc" % dsn
				if not os.path.isfile(fnout) or force:
					if not os.path.isfile(fntmp) or force:
						cdo = Cdo()
						
						griddes = (ppath+"/tmp/griddes_%s" % dsn)
						# create a grid 
						subp.call("cdo griddes %s > %s" % (data[dsn]["fname"], griddes), shell=True)
						# remap the data
						print("remapcon2 of %s started at:" % dsn, pd.Timestamp.now())
						cdo.remapcon2(griddes, input=ppath+"/tmp/MODIS_landwater.nc",  
							output=fntmp, options = "-P 8")

					# ==========c open the file out ==========
					ds_tmp = xr.open_dataset(fntmp)
					try:
						ds_tmp = ds_tmp.rename({"lat":"latitude", "lon":"longitude"})
					except:pass

					# ==========c make the file boolean ==========
					ds_tmp = (ds_tmp>.5).astype("float32").where(ds_tmp == 1)
					ds_tmp["time"] = ds_mod.time
					ds_tmp.attrs = ds_mod.attrs
					ds_ref = tempNCmaker(ds_tmp, fnout, "landwater", 
						pchunk=None, chunks=None, skip=False, pro = "%s land water mask" % dsn)
				else:
					print("A land water mask alread exists")
			ds_out = _upscaler(ppath, dsn, data, ds_mod, force)
		
		elif dsn in ["COPERN_BA", "esacci"]:
			print("Upscaling %s started at:" % dsn, pd.Timestamp.now())
			# ========== reindex the mask ==========
			def _downscaler(ppath, dsn, data, ds_mod, force):
				if not os.path.isfile(ppath+"/%s_landwater.nc" %dsn) or force:

					ds_ref = xr.open_dataset(data[dsn]["fname"])

					ds_mask = ds_mod.reindex({
						"longitude":ds_ref.longitude, 
						"latitude":ds_ref.latitude}, method="nearest")
					ds_mask = ds_mask.transpose('time', 'latitude', 'longitude')

					ds_ref = tempNCmaker(ds_mask, ppath+"/%s_landwater.nc" %dsn, "landwater", 
						pchunk=None, chunks= {'latitude': 1000, 'longitude':1000}, skip=False, 
						pro = "%s land water mask" % dsn)
				else:
					print("A land water mask alread exists")

			_downscaler(ppath, dsn, data, ds_mod, force)

		else:
			ipdb.set_trace()
	# ========== Cleanup at the end ==========
	subp.call("rm -rf %s " % ((ppath+"/tmp/")), shell=True) 

	ipdb.set_trace()
	sys.exit()

def _broadmaskGridder(ppath, region, ds, box, data, dsn, dates, force):
	"""
	Function to regrid and build masks for all the datasets
	"""
	# ========== make the new filename ==========
	datafn = "%s/Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (ppath, region, dsn)
	if os.path.isfile (datafn) and not force:
		print("a broad resolution mask file already exists for %s" % (dsn))
		return datafn

	pchunk=1000

	# ========== load the data ==========
	DAin, global_attrs = dsloader(data, dsn, dates)
	# +++++ fixe a problem with the esa data +++++
	if dsn == "esacci":
		DAin = DAin.sortby(DAin.latitude, ascending = False)
	elif dsn == "GIMMS":
		pchunk=100
	DAin_sub = DAin.sel(dict(latitude=slice(box[3], box[2]), longitude=slice(box[0], box[1])))


	# ========== reindex the mask ==========
	ds_mask = ds.reindex({
		"longitude":DAin_sub.longitude, 
		"latitude":DAin_sub.latitude}, method="nearest")

	# ========== Loop over the variables ==========
	for var in ["treecover2000", "datamask"]:
		ds_mask[var].attrs["res"] = ([abs(np.unique(np.round(np.diff(ds_mask.latitude.values), decimals=4))[0]),abs(np.unique(np.round(np.diff(ds_mask.longitude.values), decimals=4))[0])])
	
	ds_mask = tempNCmaker(ds_mask, datafn, ["treecover2000", "datamask"], 
		pchunk=pchunk, chunks={'longitude': pchunk, "latitude":pchunk}, skip=False, pro = "%s scale Treecover" % dsn)
	return datafn

def broadMask(dpath, region, minTC, nfval, force, data, dates):
	"""
	Logical method
		For broad resolution
			Coursen until it get to a resolution that i can actually work in with CDO
			CDO remapcon2 to get it to the same grid as the climate data
			......
			return an array that has both the forest cover and some form of boolean mask 
			Maybe apply some form of smoothin
	"""
	# ===============================
	# This is the low resolution mask
	# ===============================
	ppath = dpath + "/Data51/masks/broad"
	cf.pymkdir(ppath)
	ds    = HansenCoarsenLoad(ppath, region, minTC, nfval, force, data, mode="4km")

	# ========== Loop over each dataset ==========
	for dsn in data:
		if dsn == 'TerraClimate':
			# The broad mask is alread at this resolution
			continue
		else:
			print(dsn)

		# ========== load the resampling function ==========
		if region == "SIBERIA":
			box = [-10.0, 180, 40, 70]
		
		_broadmaskGridder(ppath, region, ds, box, data, dsn, dates, force)


#==============================================================================
#==============================================================================
def _dsroller(fpath, storepath, ds, DAin_sub, dsn, data, maxNF, force, region, global_attrs, dates):
	"""
	Takes the datasets and rolls them to get the mean forest fraction
	args:
		ds:		xr ds
			the hansen is forest bool
		DAin_sub: xr da
			the dataarray with the matched grid
	"""
	# ========== Setup the file name and check overwrite ==========
	Tpath = storepath + "Hansen_GFC-2018-v1.6_regrid_%s_%s_BorealMaskV2.nc" % (dsn, region)
	fnout = fpath + "Hansen_GFC-2018-v1.6_regrid_%s_%s_BorealMaskV2.nc" % (dsn, region)
	ftmp  = fpath + "tmp/Hansen_GFC-2018-v1.6_regrid_%s_lonMW.nc" % (dsn)

	if os.path.isfile (Tpath) and not force:
		print("a file already exists for %s" % (dsn))
		return Tpath, ftmp
	elif os.path.isfile (fnout) and not force:
		print("a file already exists for %s" % (dsn))
		return fnout, ftmp

	# ========== calculate the scale factor ==========
	rat = np.round(np.array(DAin_sub.attrs["res"]) / np.array(ds.datamask.attrs["res"]))
	# the scale factor between datasets
	if np.unique(rat).shape[0] == 1:
		SF = int(rat[0])
		RollF = 1 #int(SF/4 - 0.5)

	else:
		warn.warn("Lat and lon have different scale factors")
		ipdb.set_trace()
		sys.exit()

	# +++++ Moving window Smoothing +++++

	def _test(ds, SF, RollF, DAin_sub, fnout):
		# ========== Testing alternative approach ==========
		dsMW = ds.copy().chunk({"latitude":10000, "longitude":10000})
		dsMW = dsMW.rolling({"longitude":SF}, center = True, min_periods=RollF).construct('WDlon')
		dsMW = dsMW.rolling({"latitude" :SF}, center = True, min_periods=RollF).construct('WDlat')

		# ========== stack the two windows, get the mean and then subset ==========
		dsMW = dsMW.stack(mw=("WDlon", "WDlat"))
		ipdb.set_trace()
		dsMW = dsMW.mean(dim="mw")
		dsMW = dsMW.reindex({"longitude":DAin_sub.longitude,"latitude":DAin_sub.latitude}, method="nearest")

		# +++++ Fix the metadata +++++
		dsMW.attrs = ds.attrs
		dsMW.attrs["history"]  = "%s: Converted to a boolean forest mask, then resampled to match %s grid resolution using %s" % (
			(str(pd.Timestamp.now())), dsn, __file__) +ds.attrs["history"]
		dsMW.attrs["FileName"] = fnout
		dsMW.datamask.attrs    = ds.datamask.attrs	
		dsMW.latitude.attrs    = ds.latitude.attrs		
		dsMW.longitude.attrs   = ds.longitude.attrs


		# ========== Create the new layers ==========
		dsMW["mask"] = (dsMW.datamask >= maxNF).astype(int)		
		dsMW         =  dsMW.rename({"datamask":"ForestFraction"})

		dsMW = tempNCmaker(dsMW, fnout, ["ForestFraction", "mask"], pro = "complete mask")
		return dsMW

	# MW_FC = _test(ds, SF, RollF, DAin_sub, fnout)
	# ipdb.set_trace()

	MW_lons = ds.rolling({"longitude":SF}, center = True, min_periods=RollF).mean().astype("float32")
	MW_lons = MW_lons.reindex({"longitude":DAin_sub.longitude}, method="nearest")
	MW_lons = tempNCmaker(MW_lons, ftmp, "datamask", chunks={'longitude': 3000})

	# MW_lonsRI = MW_lonsRI.chunk({"latitude":-1, "longitude":1000})

	# +++++ Apply the second smooth +++++
	MW_FC    = MW_lons.rolling({"latitude":SF}, center = True, min_periods=RollF).mean()
	MW_FC_RI = MW_FC.reindex({"latitude":DAin_sub.latitude}, method="nearest")


	# +++++ Fix the metadata +++++
	MW_FC_RI.attrs = ds.attrs
	MW_FC_RI.attrs["history"]  = "%s: Converted to a boolean forest mask, then resampled to match %s grid resolution using %s" % ((str(pd.Timestamp.now())), dsn, __file__) +MW_FC_RI.attrs["history"]
	MW_FC_RI.attrs["FileName"] = fnout
	MW_FC_RI.datamask.attrs    = ds.datamask.attrs	
	MW_FC_RI.latitude.attrs    = ds.latitude.attrs		
	MW_FC_RI.longitude.attrs   = ds.longitude.attrs


	# ========== Create the new layers ==========
	MW_FC_RI["mask"] = (MW_FC_RI.datamask >= maxNF).astype("int16")		
	MW_FC_RI         = MW_FC_RI.rename({"datamask":"ForestFraction"})
	# MW_FC_RI         = MW_FC_RI.transpose("time", "latitude", "longitude")

	if dsn in ["COPERN_BA", "esacci"]:
		MW_FC_RI.chunk({"time":1, "latitude":1000, "longitude":1000})
		
		enc = ({'shuffle':True,
			'chunksizes':[1, 1000, 1000],
			'zlib':True,
			'complevel':5})
	else:
		enc = ({'shuffle':True,
			'zlib':True,
			'complevel':5})

	encoding = OrderedDict()
	for ky in ["ForestFraction", "mask"]:
		encoding[ky] = 	 enc
	# ipdb.set_trace()
	delayed_obj = MW_FC_RI.to_netcdf(fnout, 
		format         = 'NETCDF4', 
		encoding       = encoding,
		unlimited_dims = ["time"],
		compute=False)

	print("Starting write of %s gridded data at:" % dsn, pd.Timestamp.now())
	with ProgressBar():
		results = delayed_obj.compute()
	# ========== Test the file to see if it opens ==========
	MW_FC_RI = xr.open_dataset(fnout, chunks={'longitude': 3000})
	return fnout, ftmp

def HansenCoarsenLoad(ppath, region, maxNF, nfval, force, data,  mode="50m"):
	"""
	Function to open the hansen data products then mask them with key values
	args:
		ppath:		str
			path to the processed nc files
		region:		str
			name of the study region
		naxNF:		float
			the fraction of tree cover needed for a forest
		nfval:		float
			the fraction of pixels needed to consider a pixel in
		force;		bool
		data:		dict
		mode:		str

	"""

	# ========== Set up the filename and global attributes =========
	if mode == "50m":
		datafn = "%s/Hansen_GFC-2018-v1.6_%s_ProcessedTo%s.nc" % (ppath, region, mode)
	else:
		datafn = "%s/Hansen_GFC-2018-v1.6_%s_ProcessedToTerraClimate.nc" % (ppath, region)

	# ========== Check if the file already exists =========
	if os.path.isfile(datafn) and not force:
		ds_IF = xr.open_dataset(datafn, chunks={'latitude': 100})
	else:
		pptex = ({"treecover2000":"FC2000", "lossyear":"lossyear", "datamask":"mask"})
		fpath        = "%s/FRI/" %  ppath
		cf.pymkdir(fpath)

		# ========== Setup the paths ==========
		def _Hansenfile(ppath, pptex, ft, region):
			dpath  = "%s/%s/" % (ppath, pptex[ft])
			datafn = "%sHansen_GFC-2018-v1.6_%s_%s.nc" % (dpath, ft, region)
			# fnout  = "%sHansen_GFC-2018-v1.6_forestmask_%s.nc" % (dpath, region)
			return xr.open_dataset(datafn, chunks={'latitude': 10000, 'longitude':10000})

		# ========== get the datatsets ==========
		if mode == "4km":
			# LOGIC:
			# 1. Coursen until it get to a resolution that i can actually work in with CDO
			# 2. CDO remapcon2 to get it to the same grid as the climate data

			# ========== Setup the temp filenames ==========
			tmpfn1   = "%s/Hansen_GFC-2018-v1.6_%s_tmpfn1_%s.nc" % (ppath, region, mode)
			gridfile = "./data/cli/1.TERRACLIMATE/griddes"
			tmpfn2   = "%s/Hansen_GFC-2018-v1.6_%s_tmpfn2_%s.nc" % (ppath, region, mode)
			
			# ============ check if the file already exists ===========
			if (not os.path.isfile(tmpfn1) or force):
				# ========== Load the tree cover and mask negative values ==========
				ds_tc = _Hansenfile(ppath, pptex, "treecover2000", region)
				ds_tc = ds_tc.where(ds_tc >= 0)

				# ========== Save the attrs ==========
				global_attrs =  GlobalAttributes(ds_tc, "TerraClimate", fname="")
				attrs = ds_tc.attrs
				vattr = ds_tc.treecover2000.attrs

				# ========== Coarsen out to 2.5km ==========
				ds_tmp = ds_tc.coarsen(latitude=100, longitude=100).mean()
				ds_tmp.attrs = attrs
				ds_tmp.treecover2000.attrs = vattr
				
				# ========== save the file out ==========
				ds_tmp = tempNCmaker(ds_tmp, tmpfn1, "treecover2000", 
					pchunk=100, chunks={'longitude': 100, 'latitude': 100}, 
					skip=False, pro = "Hansen 2.5km Downscaled")
				ds_tmp.close()
				ds_tmp = None

			# ========== Make the next step ==========
			if (not os.path.isfile(tmpfn2)) or force:
				# ========== Use cdo to regrid the fiel =========
				cdo = Cdo()
				cdo.remapcon2(gridfile, input=tmpfn1,  output=tmpfn2, options = "-P 8")

			# ========== Add the boolean mask ==========
			ds_cov = xr.open_dataset(tmpfn2).rename({"lat":"latitude", "lon":"longitude"})
			# +++++ Create a new dataarray +++++
			ds_cov["datamask"] = (ds_cov["treecover2000"] > maxNF).astype("float32")
			# +++++ Fix the history and global attrs +++++
			hist = "%s: Netcdf file created using %s (%s):%s by %s. Mask built using data from %s" % (str(pd.Timestamp.now()), __title__, __file__, __version__, __author__, "https://glad.umd.edu/projects/gfm/boreal/data.html")
			ds_cov.attrs["history"] = hist + ds_cov.attrs["history"]

			# +++++ fix the local attrs +++++
			ds_cov.datamask.attrs = ds_cov.treecover2000.attrs
			ds_cov.datamask.attrs["res"] = ([abs(np.unique(np.round(np.diff(ds_cov.latitude.values), decimals=4))[0]),abs(np.unique(np.round(np.diff(ds_cov.longitude.values), decimals=4))[0])])
			ds_cov.datamask.attrs["standard_name"] ="FC mask"
			ds_cov.datamask.attrs['long_name']='BorealForestTreecoverMask'

			# ========== save the files out ==========
			ds_IF = tempNCmaker(ds_cov, datafn, ["treecover2000", "datamask"], pchunk=100, chunks={'longitude': 100, "latitude":100}, skip=False, pro = "Large scale Treecover")

			# ========== Cleanup the results ==========
			for fn in [tmpfn1, tmpfn2]:
				if os.path.isfile(fn):
					os.remove(fn)
			print("Cleanup of temp files complete")



		elif mode == "50m":
			ds_tc = _Hansenfile(ppath, pptex, "treecover2000", region)
			ds_tc = ds_tc.coarsen(latitude=2, longitude=2).mean()

			# ds_ly = _Hansenfile(ppath, pptex, "lossyear", region)
			ds_dm = _Hansenfile(ppath, pptex, "datamask", region)
			attrs = ds_dm.attrs
			vattr = ds_dm.datamask.attrs
			ds_dm = ds_dm.coarsen(latitude=2, longitude=2).max()
			# ========== Check if its a forest ==========
			ds_IF = (ds_tc > nfval).astype("float32").rename({"treecover2000":"datamask"})
			ds_IF = ds_IF.where(ds_dm == 1, 0)
			# ========== Add back the attrs i need ==========
			ds_IF.attrs = attrs
			ds_IF.datamask.attrs = vattr
			ds_IF.datamask.attrs["res"] = ([
				abs(np.unique(np.round(np.diff(ds_IF.latitude.values), decimals=4))[0]),
				abs(np.unique(np.round(np.diff(ds_IF.longitude.values), decimals=4))[0])])

			ds_IF = tempNCmaker(ds_IF, datafn, "datamask", 
				pchunk=10000, chunks={'latitude': 100}, 
				skip=False, pro = "Hansen Downscaled")
	return ds_IF

#==============================================================================
# FUnctions i'm not usre if i'm using 
#==============================================================================
def tempNCmaker(ds, fntmp, vname, pchunk=1000, chunks={'longitude': 1000, "latitude":1000}, skip=False, pro = "tmp"):

	""" Function to save out a tempary netcdf """
	# cf.pymkdir(tmppath)
	if pchunk is None:
		enc = {'shuffle':True,'zlib':True,'complevel':5}
	else:
		enc = {'shuffle':True,'zlib':True,'complevel':5, "chunksizes":[1, pchunk, pchunk]}

	if type(vname) == list:
		encoding = OrderedDict()
		for vn in vname:
			encoding[vn] = enc
	else:
		encoding =  ({vname:enc})

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

def dsbuilder(DAin_sub, dates, BFbool, MeanFC, NFfrac, global_attrs, fnout):

	# ========== Start making the netcdf ==========
	layers   = OrderedDict()
	encoding = OrderedDict()
	layers["ForestMask"]        = attrs_fixer(
		DAin_sub, BFbool,"ForestMask", "mask", dates)
	layers["TreeCover"]         = attrs_fixer(
		DAin_sub, MeanFC, "TreeCover", "TC", dates)
	layers["NonForestFraction"] = attrs_fixer(
		DAin_sub, NFfrac,"NonForestFraction", "NF", dates)

	enc = ({'shuffle':True,
		# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
		'zlib':True,
		'complevel':5})
	for ly in layers:
		encoding[ly] = enc
	# ========== make the dataset ==========
	ds = xr.Dataset(layers, attrs= global_attrs)
	ds.attrs["FileName"] = fnout
	return ds, encoding

def attrs_fixer(origda, da, vname, sname, dates):
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
	da.attrs = origda.attrs.copy()
	da.attrs['_FillValue']	=-1, #9.96921e+36
	da.attrs['units'     ]	="1",
	da.attrs['standard_name']	="BF%s" % sname, 
	da.attrs['long_name'	]	="BorealForest%s" %vname,
	if sname == "TC":
		da.attrs['valid_range']	= [-1, 100.0]	
	else:
		da.attrs['valid_range']	= [0.0, 1.0]
	da.longitude.attrs['units'] = 'degrees_east'
	da.latitude.attrs['units']  = 'degrees_north'
	da.time.attrs["calendar"]   = dates["calendar"]
	da.time.attrs["units"]      = dates["units"]

	return da

#==============================================================================
#==============================================================================
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

def dsloader(data, dsn, dates):
	"""Function to load and process data"""


	# ========== Load the dataset =========
	if data[dsn]["chunks"] is None:
		DS           = xr.open_dataset(data[dsn]["fname"])
		DAin         = DS[data[dsn]["var"]]
	else:
		try:
			DS           = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
			
		except Exception as e:
			print(str(e))
			ipdb.set_trace()
		# Pull out the chunks 
		if 'time' in data[dsn]["chunks"].keys():
			DAin         = DS[data[dsn]["var"]].isel({'time':0})
		else:
			warn.warn("This has not been implemented yet")
			ipdb.set_trace()
			sys.exit()

	if not data[dsn]["rename"] is None:
		DAin    = DAin.rename(data[dsn]["rename"])
	try:
		DAin["time"] = dates["CFTime"]
	except ValueError:
		DAin["time"] = np.squeeze(dates["CFTime"])
	global_attrs = GlobalAttributes(DS, dsn)
	
	try:
		len(DAin.attrs["res"])
	except KeyError:
		DAin.attrs["res"] = ([
			abs(np.unique(np.diff(DAin.longitude.values))[0]),
			abs(np.unique(np.diff(DAin.latitude.values))[0]) ])
	except Exception as e:
		print("Unknown error", str(e))
		ipdb.set_trace()
		raise e
	return DAin, global_attrs

def GlobalAttributes(ds, dsn, fname="", title="BorealForest2000forestcover"):
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
	attr["title"]               = title
	attr["summary"]             = ('''
		Boreal Forest Mask For %s Data. 
		''' % (dsn))
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. Mask built using data from %s" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__, "https://glad.umd.edu/projects/gfm/boreal/data.html")
	
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

def datasets(dpath):
	# ========== set the filnames ==========
	data= OrderedDict()
	data["TerraClimate"] = ({
		"fname":os.getcwd()+"/data/cli/1.TERRACLIMATE/TerraClimate_SIBERIA_tmean_1959_tmean.nc",
		'var':"tmean", "gridres":"4km", "region":"SIBERIA", "timestep":"Monthly", 
		"start":1959, "end":1959, "rasterio":False, "chunks":{'time': 12},
		"rename":None
		})
	data["MODIS"] = ({
		"fname":dpath + "/Data51/BurntArea/MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc",
		'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None, "maskfn":dpath + "/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"
		})
	data["COPERN_BA"] = ({
		'fname':dpath + "/Data51/BurntArea/COPERN_BA/processed/COPERN_BA_gls_2014_burntarea_SensorGapFix.nc",
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None#{"lon":"longitude", "lat":"latitude"}
		})
	data["esacci"] = ({
		"fname":dpath + "/Data51/BurntArea/esacci/processed/esacci_FireCCI_2001_burntarea.nc",
		'var':"BA", "gridres":"250m", "region":"Asia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None, "maskfn":dpath + "/Data51/BurntArea/esacci/processed/esacci_landseamask.nc"
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	data["GIMMS"] = ({
		"fname":"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_annualmax.nc",
		'var':"ndvi", "gridres":"8km", "region":"global", "timestep":"Annual", 
		"start":1982, "end":2017, "rasterio":False, "chunks":{'time': 36},
		"rename":None
		})
	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
		'var':"NDVI", "gridres":"1km", "region":"Global", "timestep":"AnnualMax",
		"start":1999, "end":2018,"rasterio":False, "chunks":{'time':1}, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})
	
	# ipdb.set_trace()

	# data["MODISaqua"] = ({
	# 	"fname":"./data/veg/MODIS/aqua/processed/MYD13Q1_A*_final.nc",
	# 	'var':"ndvi", "gridres":"250m", "region":"SIBERIA", "timestep":"16day", 
	# 	"start":2002, "end":2019
	# 	})
	return data

# ======= DO NOT DELETE THESE YET< THE INDEXING APPROACH MAK BE GREAT FOR LANDSAT MOVIE ++++++++++++
def Isforest(tp, FC, maxNF = 0.40, minCC = 0.3):

	
	# ===== Mean canapy cover ==========
	fcM = FC.mean().compute()
	ipdb.set_trace()
	sys.exit()
	# ===== Fraction of non forest ==========
	fcB =  (FC == 0).sum().compute()/np.prod(FC.shape).astype(float)
	if (fcM >= minCC) and (fcB <= maxNF):
		return 1
	else:
		return 0 	

def lonforest(BAlon, fcLon, lonres, latres):
	"""
	Function to loop over each row 
	"""
	array = np.zeros(BAlon.latitude.shape[0])
	for ilat in range(0, BAlon.latitude.shape[0]):
		tp = BAlon.isel(latitude=ilat)
		FC = 	fcLon.sel(dict(	latitude=slice(tp.latitude.values+latres, tp.latitude.values-latres)))
		array[ilat] = Isforest(tp, FC, maxNF = 0.40, minCC = 0.3)
	return array

def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/mnt/h"
	elif sysname == "ubuntu":
		# Work PC
		dpath = "/media/ubuntu/Seagate Backup Plus Drive"
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
	else:
		ipdb.set_trace()
	return dpath	
#==============================================================================
if __name__ == '__main__':
	main()