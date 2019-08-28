
"""
This script creates a boolean mask based on rules
1. is it boreal forest zone
2. In 2000, was there sufficent forest
"""
#==============================================================================

__title__ = "Boreal Forest Mask"
__author__ = "Arden Burrell"
__version__ = "v1.0(19.08.2019)"
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
	# sys.exit()

	force = False
	# ========== Create the dates ==========
	dates  = datefixer(2018, 12, 31)
	data   = datasets()
	dsn    = "HansenGFC"
	tcf    = 0.0  
	region = "SIBERIA"

	# ========== select and analysis scale ==========
	mwbox     = [10, 2, 5, 1] #in decimal degrees
	BPT       = 0.4

	# ========== Set up the filename and global attributes =========
	ppath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/HANSEN"
	pptex = ({"treecover2000":"FC2000", "lossyear":"lossyear", "datamask":"mask"})
	fpath        = "%s/FRI/" %  ppath
	cf.pymkdir(fpath)

	# ========== Setup the paths ==========
	def _Hansenfile(ppath, pptex, ft):
		dpath  = "%s/%s/" % (ppath, pptex[ft])
		datafn = "%sHansen_GFC-2018-v1.6_%s_%s.nc" % (dpath, ft, region)
		# fnout  = "%sHansen_GFC-2018-v1.6_forestmask_%s.nc" % (dpath, region)
		return xr.open_dataset(datafn, chunks={'latitude': 100})

	# ========== get the datatsets ==========
	ds_tc = _Hansenfile(ppath, pptex, "treecover2000")
	ds_ly = _Hansenfile(ppath, pptex, "lossyear")
	ds_dm = _Hansenfile(ppath, pptex, "datamask")

	# ========== Load in a test dataset and fix the time ==========
	ds_test = xr.open_dataset("./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_annualmax.nc").isel(time=-1)
	ds_test["time"] = ds_tc["time"] 
	
	# ========== subset the dataset in to match the forest cover ==========
	ds_testSUB = ds_test.sel(dict(
		latitude =slice(ds_tc.latitude.max().values,  ds_tc.latitude.min().values), 
		longitude=slice(ds_tc.longitude.min().values, ds_tc.longitude.max().values)))

	# ========== Test of the incoming values ==========
	# with ProgressBar():
	# 	test_tc = ds_tc.reindex_like(ds_testSUB, method="nearest").compute()
	# with ProgressBar():
	# 	test_dm = ds_dm.reindex_like(ds_testSUB, method="nearest").compute()    
	# with ProgressBar():
	# 	test_ly = ds_ly.reindex_like(ds_testSUB, method="nearest").compute()

	for mwb in  mwbox:
		# ds = TotalForestLoss(ds_tc, ds_ly, ds_dm, fpath, mwb, region, dates, tcf, ds_testSUB=ds_testSUB)
		ValueTester(mwb, dates, ds_SUB=ds_testSUB)

		# THIS IS THE BIT WHERE I LOOP OVER THE DIFFERENT OUTPUT GRID SIZES

	ipdb.set_trace()


		# ========== Save the results out ==========

	warn.warn('''
		This script did not work as planned, trying a new approach. Until its implelmted 
		this script is obsolete ''')
	sys.exit()

	# ========== Note, this is to be transformed into a loop ==========
	fparts = []
	for LonM, LatM in list(itertools.product(range(100, 120, 10), range(60, 70, 10))):
		
		
		# ========== Open the Hansen Forest Loss ==========
		BA, BFC, dsmask, global_attrs = ForestDataloader(LonM, LatM, dates, maskpath, dsnRES, dsn)


		def _annualtester(BA, dsmask, dates, data, dsn, BPT):
			""" 
			Test function to calculate the fraction that burns each year
			args:

			returns:

			"""
			outp  = OrderedDict()
			for yr in range(1, 19):
				
				BE      = (BA == yr) # Burn Event1
				BFpix   = (dsmask["ForestMask"].sum().values/1120**2) * 40000**2  
				
				BFyr    = BE.sum().values
				AnnFrac = BFyr/((40000**2)-BFpix)
				AnFRI   = (1/AnnFrac)
				outp["%d" %  (2000+yr)] = ({"FunrFrac":AnnFrac, "AnFRI":AnFRI})
				# outp["FunrFrac"] = AnnFrac
				# outp["FunrFrac"] = AnFRI
				print(yr, AnnFrac, AnFRI)

			ipdb.set_trace()
		_annualtester(BA, dsmask, dates, data, dsn, BPT)
		ipdb.set_trace()


		# ========== Determine Annual Forest loss fraction ==========
		AnnualForestLoss(BA, BFC, dsmask, global_attrs, FRIwin, dsn)
		# AFN = ForesLossCal(BA, dsmask, dates, data, dsn, BPT)

		def _FRIcal(AFN, FRIwin, dsmask):
			if AFN.chunks is None:
				MW_AFN = (AFN.rolling(
				 	{"longitude":FRIwin}, center = True, min_periods=25).mean().rolling(
				 	{"latitude" :FRIwin}, center = True, min_periods=25).mean())
				MW_AFN = MW_AFN.where(dsmask["ForestMask"]==1)
				FRI    = (1/MW_AFN)
				return FRI
		ipdb.set_trace()
		sys.exit()

		# ========== calculate the mean forest cover ==========
		# +++++ Moving window Smoothing +++++

		# # +++++ Moving window Smoothing +++++
		# MeanFC = MW_FC.reindex_like(DAin_sub, method="nearest")    

	

		# # ========== make the dataset ==========
		# ds, encoding = dsbuilder(DAin_sub, dates, BFbool, MeanFC, NFfrac, global_attrs, fnout)
		# delayed_obj = ds.to_netcdf(fnout, 
		# 	format         = 'NETCDF4', 
		# 	encoding       = encoding,
		# 	unlimited_dims = ["time"],
		# 	compute=False)

		# print("Starting write of data at", pd.Timestamp.now())
		# with ProgressBar():
		# 	results = delayed_obj.compute()
		# # ipdb.set_trace()
		# fparts.append(fnout)

		ipdb.set_trace()


#==============================================================================

def TotalForestLoss(ds_tc, ds_ly, ds_dm, fpath, mwb, region, dates, tcf, ds_testSUB=None):
	"""
	function us applied to the hansen forest loss products 
	"""
	

	# ========== Create the outfile name ==========
	fnout        = "%sHansen_GFC-2018-v1.6_FRI_%ddegMW_%s.nc" % (fpath, mwb, region)
	if os.path.isfile(fnout) and not force:
		print("dataset for %d deg already exist. going to next window" % (mwb))
		return xr.open_dataset(fnout, chunks={'longitude': 1000}) 

	# ========== Calculate scale factors ==========
	rat = np.round(mwb / np.array(ds_tc["treecover2000"].attrs["res"]) )
	if np.unique(rat).shape[0] == 1:
		# the scale factor between datasets
		SF    = int(rat[0])
		RollF = int(SF/4 - 0.5) # the minus 0.5 is correct for rolling windows
	else:
		warn.warn("Lat and lon have different scale factors")
		ipdb.set_trace()
		sys.exit()
	
	# ========== Calculate the amount of forest that was lost ==========
	ba_ly = ds_ly > 0

	# ========== implement the masks ==========
	ba_ly = ba_ly.where((ds_tc >  tcf).rename({"treecover2000":"lossyear"}))  # mask out the non forest
	ba_ly = ba_ly.where((ds_dm == 1.0).rename({"datamask":"lossyear"})) # mask out the non data pixels
	warn.warn("\n\n I still need to implement some form of boolean MODIS Active fire mask in order to get Fire only \n\n")

	# ========== Perform the moving windows ==========
	MW_lons  = ba_ly.rolling({"longitude":SF}, center = True, min_periods=RollF).mean()
	# Save out aa file so i can rechunk it  
	MW_lons  =  tempNCmaker(
		MW_lons, fpath+"tmp/", 
		"Hansen_GFC-2018-v1.6_boolloss_MWtmp_lon_%s_%ddegMW.nc" % (region, mwb), 
		"lossyear", chunks={'longitude': 1000}, skip=True)
	
	with ProgressBar():
		test_mw1 = MW_lons.reindex_like(ds_testSUB, method="nearest").compute()

	MW_loss  = MW_lons.rolling({"latitude":SF}, center = True, min_periods=RollF).mean()
	MW_loss  = MW_loss.where(~(MW_loss<0), 0)
	# MW_loss  = MW_loss.where((ds_tc >  tcf).rename({"treecover2000":"lossyear"}))  # mask out the non forest
	# MW_loss  = MW_loss.where((ds_dm == 1.0).rename({"datamask":"lossyear"})) # mask out the non data pixels
	
	# ========== Fix all the metadata ==========
	MW_loss.attrs                       = ds_ly.attrs
	MW_loss["lossyear"].attrs           = ds_ly["lossyear"].attrs
	MW_loss["lossyear"].latitude.attrs  = ds_ly["lossyear"].latitude.attrs
	MW_loss["lossyear"].longitude.attrs = ds_ly["lossyear"].longitude.attrs
	MW_loss["lossyear"].time.attrs      = ds_ly["lossyear"].time.attrs

	# ========== Save it out ==========
	encoding =  ({"lossyear":{'shuffle':True,'zlib':True,'complevel':5}})
	delayed_obj = MW_loss.to_netcdf(fnout, 
		format         = 'NETCDF4', 
		encoding       = encoding,
		unlimited_dims = ["time"],
		compute=False)

	print("Starting write of data at", pd.Timestamp.now());
	with ProgressBar():	results = delayed_obj.compute()

	MW_loss = xr.open_dataset(fnout, chunks={'longitude': 1000}) 

	# ========== implemented for testing ==========
	with ProgressBar():
		test_mw2 = MW_loss.reindex_like(ds_testSUB, method="nearest").compute()

	ipdb.set_trace()
	return MW_loss

def tempNCmaker(ds, tmppath, tmpname, vname, chunks={'longitude': 1000}, skip=False):

	""" Function to save out a tempary netcdf """
	cf.pymkdir(tmppath)
	
	fntmp    = tmppath + tmpname
	encoding =  ({vname:{'shuffle':True,'zlib':True,'complevel':5}})

	if skip and os.path.isfile(fntmp):
		delayed_obj = ds.to_netcdf(fntmp, 
			format         = 'NETCDF4', 
			encoding       = encoding,
			unlimited_dims = ["time"],
			compute=False)

		print("Starting write of temp data at", pd.Timestamp.now())
		with ProgressBar():
			results = delayed_obj.compute()
	dsout = xr.open_dataset(fntmp, chunks=chunks) 
	return dsout

#==============================================================================

def ValueTester(mwb, dates, LatM=60, LonM=100, ds_SUB=None, var="ndvi" ):
	"""
	This is a test function, it opens the original geotifs and the generated 
	netcdf and estimates the value at a given point
	args:
		mwb: 	int
			size in decimal degrees of the test box
		LatMin, LonMin
			the lat and lon values used to open the file names
	"""
	print("Entering Value Checker at", pd.Timestamp.now())
	
	if LonM >= 0:
		lon = "%03dE" % LonM
	else:
		lon = "%03dW" % abs(LonM)
	if LatM >= 0:
		lat = "%02dN" % LatM
	else:
		lon = "%02dS" % abs(LatM)
	
	# ========== Create the path and load data ==========
	path = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/HANSEN"
	# pptex = ({"treecover2000":"FC2000", "lossyear":"lossyear", "datamask":"mask"})
	def _rasterio_open(fname, dates):
		da = xr.open_rasterio(fname)
		da = da.rename({"band":"time", "x":"longitude", "y":"latitude"}) 
		da = da.chunk({'latitude': 1000, "longitude":1000})
		da["time"] = dates["CFTime"]
		return da
	
	pptex = ({"treecover2000":"FC2000", "lossyear":"lossyear", "datamask":"mask"})
	da_ly = _rasterio_open("%s/lossyear/Hansen_GFC-2018-v1.6_lossyear_%s_%s.tif" %(path, lat, lon), dates)
	da_tc = _rasterio_open("%s/FC2000/Hansen_GFC-2018-v1.6_treecover2000_%s_%s.tif" %(path, lat, lon), dates)
	da_dm = _rasterio_open("%s/mask/Hansen_GFC-2018-v1.6_datamask_%s_%s.tif" %(path, lat, lon), dates)

	# ========== Mask the data ==========
	da_lb = (da_ly > 0).astype(float)
	da_lb = da_lb.where(da_tc >0)
	da_lb = da_lb.where(da_dm==1)

	# ========== Make a box ==========
	LaMe = da_lb.latitude.mean().values 
	LoMe = da_lb.longitude.mean().values 
	da_box = da_lb.sel(dict(
		latitude =slice(LaMe + (mwb/2.0),  LaMe - (mwb/2.0)), 
		longitude=slice(LoMe - (mwb/2.0),  LoMe + (mwb/2.0))))

	# ========== subset the test dataset ==========
	if not ds_SUB is None:
		ds_t = ds_SUB.sel(dict(
			latitude =slice(LaMe + (mwb/2.0),  LaMe - (mwb/2.0)), 
			longitude=slice(LoMe - (mwb/2.0),  LoMe + (mwb/2.0)))).rename({var:"band"})
		ds_t["time"] = da_lb.time
		with ProgressBar():
			test_mw = da_box.reindex_like(ds_t, method="nearest").compute()
		test_mw.plot()

	 # ========== get the test values ==========
	tot_bnt = da_box.mean().values
	ann_bnt = (da_box.mean().values / 18)
	est_fri = 1/ann_bnt

	print(tot_bnt, ann_bnt, est_fri)

	ipdb.set_trace()


#==============================================================================
def AnnualForestLoss(BA, BFC, dsmask, global_attrs, FRIwin, dsn):
	""" 
	Function to work out how much or the forest burnt in a given yeat
	"""
	# ========== Calculate scale factors ==========
	rat = np.round(FRIwin / np.array(BFC.attrs["res"]) )
	if np.unique(rat).shape[0] == 1:
		# the scale factor between datasets
		SF    = int(rat[0])
		RollF = int(SF/2 - 0.5) # the minus 0.5 is correct for rolling windows
	else:
		warn.warn("Lat and lon have different scale factors")
		ipdb.set_trace()
		sys.exit()

	# ========== Workout if its a forest in 2000 ==========
	FstThres = 0.30
	IsForest = (BFC >= FstThres)

	# ========== Pull the Non Forest fraction from the mask ========== 
	NF       = dsmask["NonForestFraction"]

	# ========== Create an active fire detection layer ==========
	warn.warn("\n\n I still need to implement some form of boolean MODIS Active fire mask in order to get Fire only \n\n")
	
	# ========== Loop over each year ==========
	for yr in range(1, 19):
		
		# ========== Convert the BA into a boolean value ==========
		BA_yr = (BA==yr).astype(float)
		BA_yr =  BA_yr.where(IsForest) # Nan mask non forests

		# ########################## SKIP FOR TESTING ONLY ##############
		skip = False

		# ========== Start the rolling ==========
		MW_lons  = BA_yr.rolling({"longitude":SF}, center = True, min_periods=RollF).mean()
		# Save out aa file so i can rechunk it  
		print("Startung 20%02d Moving window tempfile at:" % yr, pd.Timestamp.now())
		MW_lons  = tempNCmaker(dsn, MW_lons, "MW_lon_%d" % yr, chunks={'longitude': 1000}, skip=skip)
		MW_loss  = MW_lons.rolling({"latitude":SF}, center = True, min_periods=RollF).mean()
		MW_loss  = MW_loss.where(~(MW_loss<0), 0).chunk({'latitude':1000})

		# ========== Reindex to the mask gird ===========
		FL       = MW_loss.reindex_like(NF, method="nearest").compute()
		ipdb.set_trace()
		sys.exit()

def ForesLossCal(BA, dsmask, dates, data, dsn, BPT):
	""" 
	Function to calculate the forest loss at the grid scale of the mask
	args:
		BA: xr DA
			Hansen Loss year for the region

		BPT: float
			the percent of the small scale pixels that need to burn for an area to count as having burnt

	returns:

	"""

	# ========== calculate the grid scale factor ==========
	rat = np.round(np.array(dsmask["NonForestFraction"].attrs["res"]) / np.array(BA.attrs["res"]))
	if np.unique(rat).shape[0] == 1:
		# the scale factor between datasets
		SF    = int(rat[0])
		RollF = int(SF/4 - 0.5) # the minus 0.5 is correct for rolling windows
	else:
		warn.warn("Lat and lon have different scale factors")
		ipdb.set_trace()
		sys.exit()

	warn.warn("\n\n I still need to implement some form of boolean MODIS Active fire mask in order to get Fire only \n\n")
	
	# ========== Pull the Non Forest fraction from the mask ========== 
	NF       = dsmask["NonForestFraction"]
	
	# ========== Convert forest loss to bool ==========
	TC_loss  = (BA>0).astype(float)
	# +++++ Get the number of non forested pixels +++++

	MW_lons  = TC_loss.rolling({"longitude":SF}, center = True, min_periods=RollF).mean()
	# Save out aa file so i can rechunk it  
	MW_lons  = tempNCmaker(dsn, MW_lons, "MW_lon", chunks={'longitude': 1000}, skip=False)
	MW_loss  = MW_lons.rolling({"latitude":SF}, center = True, min_periods=RollF).mean()
	MW_loss  = MW_loss.where(~(MW_loss<0), 0).chunk({'latitude':1000})

	# ========== Reindex to the mask gird ===========
	FL       = (MW_loss.reindex_like(NF, method="nearest")).values
	FL[FL>1] = 1
	OF       = (1.0 - NF).where((1.0 - NF)>0)   # Original Forest (circa 200), where to rm div by zero
	# ========== calculate the fraction of original forest that was lost ==========
	BF       = (FL / OF).rename("BurntFraction").persist()              # Burnt Fraction
	BF       = BF.where(BF<=1, 1)
	ipdb.set_trace()

	# ========== Convert to an Anuual Burnt area  ==========
	yrs = 18.0 #2000-2018
	ABF =  (BF/yrs).rename("AnnualBurntFraction") #Annual Burnt Fraction
	return ABF
	
def ForestDataloader(LonM, LatM, dates, maskpath, dsnRES, dsn):
	"""
	Function takes a lon and a lat and them opens the appropiate 2000 forest 
	cover data
	args:
		LonM: int
			must be divisible by 10
		LatM: int
			must be divisible by 10
	returns:
		ds: xarray dataset
			processed xr dataset 
	"""
	if LonM >= 0:
		lon = "%03dE" % LonM
	else:
		lon = "%03dW" % abs(LonM)
	if LatM >= 0:
		lat = "%02dN" % LatM
	else:
		lon = "%02dS" % abs(LatM)
	# ========== Create the path ==========
	path = "./data/Forestloss/Lossyear/Hansen_GFC-2018-v1.6_lossyear_%s_%s.tif" % (lat, lon)
	da = xr.open_rasterio(path)
	da = da.rename({"band":"time", "x":"longitude", "y":"latitude"}) 
	# da = da.chunk({'latitude': 4444})   
	da = da.chunk({'latitude': 1000})   #, "longitude":4000
	# da = da.chunk({'latitude': 1000, "longitude":4000})   #

	da["time"] = dates["CFTime"]

	# ========== Create the path ==========
	tpath = "./data/Forestloss/2000Forestcover/Hansen_GFC-2018-v1.6_treecover2000_%s_%s.tif" % (lat, lon)
	daFC = xr.open_rasterio(tpath)
	daFC = daFC.rename({"band":"time", "x":"longitude", "y":"latitude"}) 
	daFC = daFC.chunk({'latitude': 1000})   #, "longitude":1000
	daFC["time"] = dates["CFTime"]

	# ========== Open the forest Mask file for the output grid ==========
	maskfn = maskpath + "BorealForest_2000forestcover_%s_%03d_%02d.nc" % (dsnRES, LonM, LatM)
	dsmask = xr.open_dataset(maskfn) 
	dsmask["time"] = dates["CFTime"]
	# for var in ["ForestMask","TreeCover","NonForestFraction"]:
	# 	dsmask[var]["time"] = dates["CFTime"]
	global_attrs = GlobalAttributes(dsmask, dsn)
	
	return da, daFC, dsmask, global_attrs

#==============================================================================
#==============================================================================
#==============================================================================
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
		if data[dsn]["chunks"] is None:
			DS           = xr.open_dataset(data[dsn]["fname"])
			
			DAin         = DS[data[dsn]["var"]]
			DAin["time"] = dates["CFTime"]
			if not data[dsn]["rename"] is None:
				DAin    = DAin.rename(data[dsn]["rename"])
			
			global_attrs = GlobalAttributes(DS, dsn)
			try:
				len(DAin.attrs["res"])
			except KeyError:
				DAin.attrs["res"] = ([
					abs(np.unique(np.diff(DAin.longitude.values))[0]),
					abs(np.unique(np.diff(DAin.latitude.values))[0]) ])
			except:
				print("Unknown error")
				ipdb.set_trace()
				sys.exit()
		else:
			warn.warn("This has not been implemented yet")
			ipdb.set_trace()
			sys.exit()
	return DAin, global_attrs

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

def GlobalAttributes(ds, dsn):
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
	attr["summary"]             = "BorealForestLossandFRI_%sData" % (dsn)
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
# @jit
#==============================================================================

def datasets():
	# ========== set the filnames ==========
	data= OrderedDict()
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
	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
		'var':"NDVI", "gridres":"1km", "region":"Global", "timestep":"AnnualMax",
		"start":1999, "end":2018,"rasterio":False, "chunks":{'time':1}, 
		"rename":{"lon":"longitude", "lat":"latitude"}
		})
	data["HansenGFC"] = ({
		'fname':"./data/Forestloss/Lossyear/Hansen_GFC-2018-v1.6_lossyear_*_*.tif",
		'var':"lossyear", "gridres":"25m", "region":"Global", "timestep":"Annual",
		"start":2000, "end":2018,"rasterio":False, "chunks":{'latitude': 1000},
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

# 	pass

# _annualtester(BA, dsmask, dates, data, dsn, BPT)
# ipdb.set_trace()
# sys.exit()

	
#==============================================================================
if __name__ == '__main__':
	main()