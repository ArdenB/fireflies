
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
# import rasterio
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
# import regionmask as rm
# import itertools
# Import debugging packages 
import ipdb
# from rasterio.warp import transform
from shapely.geometry import Polygon
# import geopandas as gpd
# from rasterio import features
# from affine import Affine
# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================

def main():
	# ========== Setup the paths ==========
	TCF = 10
	# TCF = 50
	dpath, chunksize = syspath()
	data  = datasets(dpath, chunksize, TCF=TCF)
	
	# ========== select and analysis scale ==========
	mwbox = [1]#, 2, 5]#, 10] #in decimal degrees
	maskds = "esacci"
	maskforce = False # Added to allow skiping of the mask
	for dsn in data:
		print(dsn)
		# ========== Set up the filename and global attributes =========
		if dsn.startswith("HANSEN"):
			ppath = dpath + "/BurntArea/HANSEN/FRI/"
			force = True
		else:
			ppath = dpath + "/BurntArea/%s/FRI/" %  dsn
			force = False
		cf.pymkdir(ppath)
		
		# ========== Get the dataset =========
		mask = landseamaks(data, dsn, dpath, maskforce )

		# ========== Calculate the annual burn frewuency =========
		ds_ann = ANNcalculator(data, dsn, mask, force, ppath, dpath, chunksize, TCF)
		# force = True
		# breakpoint()
		# ========== work out the FRI ==========
		FRIcal(ds_ann, mask, dsn, force, ppath, dpath, mwbox, data, chunksize, TCF)
		# force = False
		print(dsn, " Complete at:", pd.Timestamp.now())
		
	ipdb.set_trace()






#==============================================================================
def FRIcal(ds_ann, mask, dsn, force, ppath, dpath, mwbox, data, chunksize, TCF):
	""""""
	""" Function to caluclate the FRI at different resolutions """

	# ========== Add a loading string for forest cover in Hansen Datasets ==========
	if dsn.startswith("HANSEN") and (TCF > 0.):
		tcfs = "_%dperTC" % np.round(TCF)
	else:
		tcfs = ""

	# ========== Setup a working path ==========
	tpath = ppath+"tmp/"
	
	# ========== work out the ration ==========
	pix    =  abs(np.unique(np.diff(ds_ann.latitude.values))[0]) 
	# ds_ann = ds_ann.chunk({"latitude":chunksize, "longitude":-1})
	print(f"Loading annual data into ram at: {pd.Timestamp.now()}")
	ds_ann.persist()

	# ========== Build a cleanup list ==========
	cleanup = []
	for mwb in mwbox:
		print("Starting %s %d degree moving window at:" %(dsn, mwb), pd.Timestamp.now())
		fname  = "%s%s_annual_burns_MW_%ddegreeBox.nc" % (dsn, tcfs, mwb)
		tname  = "%s%s_annual_burns_lonMW_%ddegreeBox.nc" % (dsn, tcfs, mwb)
		tname2 = "%s%s_annual_burns_latMW_%ddegreeBox.nc" % (dsn, tcfs, mwb)
		tMnme  = "%s%s_annual_burns_lonMW_tmpmask_%ddegreeBox.nc" % (dsn, tcfs, mwb)

		# ========== Check if a valid file already exists ==========
		if os.path.isfile(ppath+fname) and not force:
			cleanup.append(ppath+tname)
			cleanup.append(tpath+tMnme)
			continue
		# ===== get the ratio =====
		SF = np.round(mwb /pix).astype(int)

		# # ===== Create a masksum =====
		# warn.warn("I need to reimplement the mask here:")
		def _maskmaker(SF, mask, tpath, tMnme, dsn):
			mask_sum = mask.fillna(0).rolling({"longitude":SF}, center = True, min_periods=1).sum(skipna=False)
			print("Mask Role 1:", pd.Timestamp.now())
			mask_sum = mask_sum.rolling({"latitude":SF}, center = True, min_periods=1).sum(skipna=False)
			mask_sum = (mask_sum > ((SF/2)**2)).astype("int16")
			print("Mask Role 2:", pd.Timestamp.now())
			if dsn.startswith("HANSEN"):
				mask_sum = mask_sum.sortby("latitude", ascending=False)
				mask_sum = mask_sum.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))
			mask_sum = tempNCmaker(mask_sum, tpath, tMnme, "landwater", 
				None, readchunks={'longitude': 500}, skip=False)
			mask_sum.close()
			mask_sum = None
			

		if not os.path.isfile(tpath + tMnme):
			_maskmaker(SF, mask, tpath, tMnme, dsn)
			# sys.exit()
		
		print("Mask reload:", pd.Timestamp.now())
		mask_sum = xr.open_dataset(tpath+tMnme)
		# continue
		# This is so i can count the number of values that are valid in each location
		# breakpoint()

		# ===== Calculate the Moving window on dim 1 =====
		if dsn.startswith("HANSEN") and (TCF > 0.):
			# Hansen use TCF weights as it cannot measure loss in non forest
			# ========== Read in the weights ==========
			fn_wei = dpath + f"/BurntArea/HANSEN/lossyear/Hansen_GFC-2018-v1.6_weights_SIBERIAatesacci{tcfs}.nc"
			weights = xr.open_dataset(fn_wei)#.rename({"lossyear":"AnBF"})
			# ========== Do the MV on the weights ==========
			print(f"Starting the MW for the weight calculation at {pd.Timestamp.now()}")
			ds_ww  = weights.rolling({"longitude":SF},center = True, min_periods=1).sum(skipna=False)
			ds_ww  = ds_ww.rolling({"latitude":SF},center = True, min_periods=1).sum(skipna=False).compute()
			
			# ========== Multiply the raw weights by the ds_ann then roll the ds_ann ==========
			print(f"Starting the MW for {dsn} at {pd.Timestamp.now()}")
			dsan_lons = (ds_ann.fillna(0)*weights.lossyear.values).rolling(
				{"longitude":SF}, center = True, min_periods=1).sum(skipna=False)
			ds_out = dsan_lons.rolling({"latitude":SF}, center = True, min_periods=1).sum(skipna=False)
			
			# ========== Divide the rolled MW rolled weights ==========
			ds_out["AnBF"] /= ds_ww.lossyear.values
			# breakpoint()

		else:
			if dsn == 'GFED':
				print(f"Loading {dsn} data into ram at", pd.Timestamp.now())
				with ProgressBar():
					dsan_lons = ds_ann.rolling({"longitude":SF}, center = True, min_periods=1).mean().compute() 
			else:
				dsan_lons = ds_ann.rolling({"longitude":SF}, center = True, min_periods=1).mean() 
			
			dsan_lons = tempNCmaker(
				dsan_lons, tpath, tname, "AnBF", 
				{'latitude': chunksize}, readchunks={'longitude': chunksize}, skip=False)
			
			print(f"Loading temp rolled dataset data into ram at: {pd.Timestamp.now()}")
			if dsn == 'GFED':
				with ProgressBar():
					dsan_lons = dsan_lons.compute()
			else:
				dsan_lons.persist()
			
			# ===== Calculate the Moving window in the other dim =====
			ds_out = dsan_lons.rolling({"latitude":SF}, center = True, min_periods=1).mean() 
			
		# ========== Mask out bad pixels ==========
		# ===== Deal with the locations with no fire history =====
		ds_out = ds_out.where(ds_out > 0, 0.000001)
		ds_out = ds_out.where(mask["landwater"].values == 1) #Mask out water
		ds_out = ds_out.where(mask_sum["landwater"].values == 1) #Mask out points that lack data
		
		# ===== Calculate a FRI =====
		ds_out["FRI"] = 1.0/ds_out["AnBF"]

		# ===== add some attrs =====
		ds_out.attrs   = ds_ann.attrs
		GlobalAttributes(ds_out, dsn, fnameout=ppath+fname)

		# ===== Save the file out =====
		ds_out = tempNCmaker(
			ds_out, ppath, fname, ["AnBF", "FRI"], {'longitude': chunksize}, 
			readchunks=data[dsn]["chunks"], skip=False, name="%s %d degree MW" % (dsn, mwb))


		# ipdb.set_trace()
		cleanup.append(tpath+tname)
		cleanup.append(tpath+tMnme)

	print("Starting excess file cleanup at:", pd.Timestamp.now())
	for file in  cleanup:
		if os.path.isfile(file):
			os.remove(file)

def ANNcalculator(data, dsn, mask, force, ppath, dpath, chunksize, TCF):
	""" Function to calculate the FRI 
	args
		data: 	Ordered dict
		dsn:	str of the dataset name
		ds:		XR dataset
	"""
	# ========== Add a loading string for forest cover in Hansen Datasets ==========
	if dsn.startswith("HANSEN") and (TCF > 0.):
		tcfs = "_%dperTC" % np.round(TCF)
	else:
		tcfs = ""

	# ========== Setup a working path ==========
	tpath = ppath+"tmp/"
	cf.pymkdir(tpath)

	# ======================================================
	# ========== Build the annual mean burnt area ==========
	# ======================================================
	
	# ========== setup the temp filnames ==========
	tname = "%s%s_annual_burns.nc" % (dsn, tcfs)

	if not os.path.isfile(tpath+tname) or force:
		# breakpoint()
		# ========== load the data ==========
		ds = dsloader(data, dsn, ppath, dpath, force)
		if dsn == 'GFED':
			print(f"Loading {dsn} data into ram at", pd.Timestamp.now())
			with ProgressBar():
				ds = ds.compute()

		# ========== calculate the sum ==========
		dates   = datefixer(data[dsn]["end"], 12, 31)
		ds_flat = ds.mean(dim="time", keep_attrs=True).expand_dims({"time":dates["CFTime"]}).rename({data[dsn]["var"]:"AnBF"})
		ds_flat.time.attrs["calendar"]   = dates["calendar"]
		ds_flat.time.attrs["units"]      = dates["units"]

		# ========== Write out the file ==========
		attrs = GlobalAttributes(ds_flat, dsn, fnameout=ppath+tname)

		# ========== add some form of mask here ==========
		try:
			ds_flat = ds_flat.where(mask["landwater"].values == 1).astype("float32")
		except Exception as e:
			# ========== Fix the Hansen mask ==========
			print("starting mask reprocessing at:", pd.Timestamp.now())
			mask = mask.sortby("latitude", ascending=False)
			mask = mask.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))
			ds_flat = ds_flat.where(mask["landwater"].values == 1).astype("float32")

		# ========== create a date ==========
		dates    = datefixer(data[dsn]["end"], 12, 31)

		# ===== fix the time =====
		ds_flat["time"] = dates["CFTime"]
		ds_flat.time.attrs["calendar"]   = dates["calendar"]
		ds_flat.time.attrs["units"]      = dates["units"]

		# ===== fix the attrs =====
		ds_flat.attrs = ds.attrs
		GlobalAttributes(ds_flat, dsn, fnameout=ppath+tname)


		ds_flat = tempNCmaker(
			ds_flat, tpath, tname, "AnBF", 
			data[dsn]["chunks"], skip=False, name="%s annual BA" % dsn)
		# breakpoint()
	
	else:
		print("Opening existing Annual Burn Fraction file")
		ds_flat = xr.open_dataset(tpath+tname)#, chunks=data[dsn]["chunks"])

	return ds_flat

#==============================================================================
#==============================================================================
def tempNCmaker(ds, tmppath, tmpname, vname, writechunks, readchunks={'longitude': 1000}, skip=False, name="tmp"):

	""" Function to save out a tempary netcdf """
	cf.pymkdir(tmppath)
	
	fntmp    = tmppath + tmpname
	if type(vname) == list:
		encoding = OrderedDict()
		for vn in vname:
			encoding[vn] = {'shuffle':True,'zlib':True,'complevel':5}
	else:
		encoding =  ({vname:{'shuffle':True,'zlib':True,'complevel':5}})
	if not all([skip, os.path.isfile(fntmp)]):
		delayed_obj = ds.to_netcdf(fntmp, 
			format         = 'NETCDF4', 
			encoding       = encoding,
			unlimited_dims = ["time"],
			compute=False)

		print("Starting write of %s data at" % name, pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()
	dsout = xr.open_dataset(fntmp, chunks=readchunks) 
	return dsout

def dsloader(data, dsn, ppath, dpath, force):
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
		lat = []	#a test to make sure the sizes are correct
		for fn in fnames:
			dsin = xr.open_dataset(fn, chunks=data[dsn]["chunks"])
			lat.append(dsin[data[dsn]["var"]].shape[1] )

		# ========== open the dataset ==========
		ds = xr.open_mfdataset(fnames, combine='nested', concat_dim="time",chunks=(data[dsn]["chunks"]))

		
		# ========== Add a simple dataset check ==========
		if not np.unique(lat).shape[0] == 1:
			warn.warn("the datasets have missmatched size, going interactive")
			ipdb.set_trace()
			sys.exit()
	else:
		ds = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
	return ds

def landseamaks(data, dsn, dpath, force, chunks=None, maskds = "esacci"):
	# ========== create the mask fielname ==========
	# masknm = ppath + "%s_landseamask.nc" % dsn
	if dsn.startswith("HANSEN"):
		print("starting mask reprocessing at:", pd.Timestamp.now())
		masknm = dpath+"/masks/landwater/%s_landwater.nc" % maskds
		raw_mask = xr.open_dataset(masknm, chunks=chunks)
		raw_mask = raw_mask.sortby("latitude", ascending=False)
		raw_mask = raw_mask.sel(dict(latitude=slice(70.0, 40.0), longitude=slice(-10.0, 180.0)))
	else:
		masknm = dpath+"/masks/landwater/%s_landwater.nc" % dsn
		raw_mask = xr.open_dataset(masknm, chunks=chunks)
	# if dsn == "esacci":
	# 	chunks = data[dsn]["chunks"]

	# raw_mask = xr.open_dataset(masknm, chunks=chunks)
	return raw_mask

def datasets(dpath, chunksize, TCF = 0):
	"""
	args:
		TCF: int
			the fraction of Hansen forest cover included in the analysis, TCF = 10 gives 10 percent,
	"""
	if TCF == 0.:
		tcfs = ""
	else:
		tcfs = "_%dperTC" % np.round(TCF)
	# ========== set the filnames ==========
	data= OrderedDict()
	data["GFED"] = ({
		'fname':"./data/BurntArea/GFED/processed/GFED_annual_burendfraction.nc",
		'var':"BA", "gridres":"5km", "region":"SIBERIA", "timestep":"Annual",
		"start":1997, "end":2016,"rasterio":False, "chunks":{'time':1,'longitude': chunksize, 'latitude': chunksize}, 
		"rename":None
		})
	data["COPERN_BA"] = ({
		'fname':dpath+"/BurntArea/COPERN_BA/processed/COPERN_BA_gls_*_SensorGapFix.nc",
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
		})

	data["HANSEN"] = ({
		"fname":dpath+"/BurntArea/HANSEN/lossyear/Hansen_GFC-2018-v1.6_*_totalloss_SIBERIAatesacci%s.nc" % tcfs,
		'var':"lossyear", "gridres":"250m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': chunksize, 'latitude': chunksize},
		"rename":None, 
		})
	data["HANSEN_AFmask"] = ({
		"fname":dpath+"/BurntArea/HANSEN/lossyear/Hansen_GFC-2018-v1.6_*_totalloss_SIBERIAatesacci%s_MODISAFmasked.nc" % tcfs,
		'var':"lossyear", "gridres":"250m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': chunksize, 'latitude': chunksize},
		"rename":None, 
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

def GlobalAttributes(ds, dsn, fnameout=""):
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


def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h/Data51"
		dpath = "/mnt/d/Data51"
		chunksize = 20
		# chunksize = 5000
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51"
		dpath = "/media/ubuntu/Harbinger/Data51"
		chunksize = 50
		
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
	elif 'ccrc.unsw.edu.au' in sysname:
		dpath = "/srv/ccrc/data51/z3466821"
		chunksize = 20
		# chunksize = 5000
	elif sysname in ['burrell-pre5820', 'LAPTOP-8C4IGM68', 'DESKTOP-KMJEPJ8']:
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51"
		dpath = "./data"
		chunksize = 300
	elif sysname == 'arden-Precision-5820-Tower-X-Series':
		# WHRC linux distro
		# dpath= "/media/arden/Harbingerq/Data51"
		dpath = "./data"
		chunksize = 300
	else:
		ipdb.set_trace()
	return dpath, chunksize	
#==============================================================================

if __name__ == '__main__':
	main()