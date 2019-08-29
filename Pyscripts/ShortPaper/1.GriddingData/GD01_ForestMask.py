
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
import rasterio
import xarray as xr
from dask.diagnostics import ProgressBar
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
	# ========== Setup the paths ==========
	ft     = "treecover2000"
	region = "SIBERIA"
	force  = False
	

	# ========== Create the mask dates ==========
	dates  = datefixer(2000, 1, 1)
	nfval  = 0.0  # TC Value considered not forest
	minTC  = 0.30 # Minimum Tree cover
	maxNF  = 0.30 # Max fraction of non forest
	data   = datasets()

	# ========== load in the datasets ==========
	ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/HANSEN"
	ds    = HansenNCload(ppath, region, maxNF, nfval)

	# ========== Loop over the datasets ==========
	for dsn in data:
		# ========== Set up the filename and global attributes =========
		fpath        = "./data/other/ForestExtent/%s/" % dsn
		cf.pymkdir(fpath)
		
		# ========== Load the grids =========
		DAin, global_attrs = dsloader(data, dsn, dates)

		# ========== subset the dataset in to match the forest cover ==========
		DAin_sub = DAin.sel(dict(
			latitude=slice(ds.latitude.max().values, ds.latitude.min().values), 
			longitude=slice(ds.longitude.min().values, ds.longitude.max().values)))
		
		# ========== generate the new datasets ==========
		out = _dsroller(fpath, ds, DAin_sub, dsn, data, maxNF, force, region, global_attrs, dates)

	ipdb.set_trace()

#==============================================================================
def _dsroller(fpath, ds, DAin_sub, dsn, data, maxNF, force, region, global_attrs, dates):
	"""
	Takes the datasets and rolls them to get the mean forest fraction
	args:
		ds:		xr ds
			the hansen is forest bool
		DAin_sub: xr da
			the dataarray with the matched grid
	"""
	# ========== Setup the file name and check overwrite ==========
	fnout = fpath + "Hansen_GFC-2018-v1.6_regrid_%s_%s_BorealMask.nc" % (dsn, region)

	if os.path.isfile (fnout) and not force:
		print("a file already exists for %s" % (dsn))
		return fnout

	# ========== calculate the scale factor ==========
	rat = np.round(np.array(DAin_sub.attrs["res"]) / np.array(ds.datamask.attrs["res"]))
	# the scale factor between datasets
	if np.unique(rat).shape[0] == 1:
		SF = int(rat[0])
		RollF = int(SF/4 - 0.5)

	else:
		warn.warn("Lat and lon have different scale factors")
		ipdb.set_trace()
		sys.exit()

	# +++++ Moving window Smoothing +++++
	MW_lons   = ds.rolling({"longitude":SF}, center = True, min_periods=RollF).mean() 
	MW_lonsRI = MW_lons.reindex({"longitude":DAin_sub.longitude}, method="nearest")
	MW_lonsRI = MW_lonsRI.chunk({"latitude":-1, "longitude":1000})
	# MW_lonsRI = tempNCmaker(MW_lonsRI, fptmp, fntmp, "datamask", chunks={'longitude': 1000}, skip=False)

	# +++++ Apply the second smooth +++++
	MW_FC    = MW_lonsRI.rolling({"latitude":SF}, center = True, min_periods=RollF).mean()
	MW_FC_RI = MW_FC.reindex({"latitude":DAin_sub.latitude}, method="nearest")


	# +++++ Fix the metadata +++++
	MW_FC_RI.attrs = ds.attrs
	MW_FC_RI.attrs["history"]  = "%s: Converted to a boolean forest mask, then resampled to match %s grid resolution using %s" % ((str(pd.Timestamp.now())), dsn, __file__) +MW_FC_RI.attrs["history"]
	MW_FC_RI.attrs["FileName"] = fnout
	MW_FC_RI.datamask.attrs    = ds.datamask.attrs	
	MW_FC_RI.latitude.attrs    = ds.latitude.attrs		
	MW_FC_RI.longitude.attrs   = ds.longitude.attrs


	# ========== Create the new layers ==========
	MW_FC_RI["mask"] = (MW_FC_RI.datamask >= maxNF).astype(int)		
	MW_FC_RI         = MW_FC_RI.rename({"datamask":"ForestFraction"})


	encoding = OrderedDict()
	for ky in ["ForestFraction", "mask"]:
		encoding[ky] = 	 ({'shuffle':True,
			# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
			'zlib':True,
			'complevel':5})

	delayed_obj = MW_FC_RI.to_netcdf(fnout, 
		format         = 'NETCDF4', 
		encoding       = encoding,
		unlimited_dims = ["time"],
		compute=False)

	print("Starting write of %s gridded data at:" % dsn, pd.Timestamp.now())
	with ProgressBar():
		results = delayed_obj.compute()

	return fnout

def HansenNCload(ppath, region, maxNF, nfval):
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
	"""

	# ========== Set up the filename and global attributes =========
	pptex = ({"treecover2000":"FC2000", "lossyear":"lossyear", "datamask":"mask"})
	fpath        = "%s/FRI/" %  ppath
	cf.pymkdir(fpath)

	# ========== Setup the paths ==========
	def _Hansenfile(ppath, pptex, ft, region):
		dpath  = "%s/%s/" % (ppath, pptex[ft])
		datafn = "%sHansen_GFC-2018-v1.6_%s_%s.nc" % (dpath, ft, region)
		# fnout  = "%sHansen_GFC-2018-v1.6_forestmask_%s.nc" % (dpath, region)
		return xr.open_dataset(datafn, chunks={'latitude': 100})

	# ========== get the datatsets ==========
	ds_tc = _Hansenfile(ppath, pptex, "treecover2000", region)
	# ds_ly = _Hansenfile(ppath, pptex, "lossyear", region)
	ds_dm = _Hansenfile(ppath, pptex, "datamask", region)
	# ========== Check if its a forest ==========
	ds_IF = (ds_tc > nfval).astype(float).rename({"treecover2000":"datamask"})
	ds_IF = ds_IF.where(ds_dm == 1, 0)
	# ========== Add back the attrs i need ==========
	ds_IF.datamask.attrs = ds_dm.datamask.attrs

	return ds_IF

#==============================================================================
# FUnctions i'm not usre if i'm using 
#==============================================================================
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
	# if sname == "TC":
	# 	da.attrs['valid_range']	= [-1, 100.0]	
	# else:
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
	if data[dsn]["rasterio"]:
		DAin    = xr.open_rasterio(data[dsn]["fname"])
		if not data[dsn]["rename"] is None:
			DAin    = DAin.rename(data[dsn]["rename"])
		if not data[dsn]["chunks"] is None:
			DAin = DAin.chunk(data[dsn]["chunks"])    
		
		# ========== calculate the bounding box of the pixel ==========
		lonres = DAin.attrs["res"][0]/2
		latres = DAin.attrs["res"][1]/2
		DAin["time"] = dates["CFTime"]
		global_attrs = GlobalAttributes(None, dsn)	


	else:
		# ========== Load the dataset =========
		if data[dsn]["chunks"] is None:
			DS           = xr.open_dataset(data[dsn]["fname"])
			DAin         = DS[data[dsn]["var"]]
		else:
			DS           = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
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
	attr["summary"]             = ('''
		Boreal Forest Mask For %s Data. 
		''' % (dsn))
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s. Maks built using data from %s" % (
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

def datasets():
	# ========== set the filnames ==========
	data= OrderedDict()
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
	data["COPERN_BA"] = ({
		'fname':"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/M0044633/c_gls_BA300_201812200000_GLOBE_PROBAV_V1.1.1.nc",
		'var':"BA_DEKAD", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":None, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})
	
	data["esacci"] = ({
		"fname":"./data/BurntArea/20010101-ESACCI-L3S_FIRE-BA-MODIS-AREA_4-fv5.1-JD.tif",
		'var':"BurntArea", "gridres":"250m", "region":"Asia", "timestep":"Monthly", 
		"start":2001, "end":2018, "rasterio":True, "chunks":{'latitude': 1000},
		"rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	# data["MODISaqua"] = ({
	# 	"fname":"./data/veg/MODIS/aqua/processed/MYD13Q1_A*_final.nc",
	# 	'var':"ndvi", "gridres":"250m", "region":"SIBERIA", "timestep":"16day", 
	# 	"start":2002, "end":2019
	# 	})
	# data["MODIS_CMG"] = ({
	# 	"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/NDVI/5.MODIS/terra/processed/MODIS_terra_MOD13C1_5kmCMG_anmax.nc",
	# 	'var':"ndvi", "gridres":"5km", "region":"Global", "timestep":"AnnualMax", 
	# 	"start":2000, "end":2018
		# })
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
	
#==============================================================================
if __name__ == '__main__':
	main()