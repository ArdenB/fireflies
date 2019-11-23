
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

	force = True
	# ========== Create the dates ==========
	dates  = datefixer(2018, 12, 31)
	data   = datasets()
	dsn    = "HansenGFC"
	tcf    = 0.2  
	region = "SIBERIA"

	# ========== select and analysis scale ==========
	mwbox     = [2, 5, 10] #in decimal degrees
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
		return xr.open_dataset(datafn, chunks={'latitude': 5000, "longitude":5000})



	# ========== Loop over the datasets ==========
	finalclean = []
	for dsn in data:
		fpath = ppath+"/FRI/"
		cf.pymkdir(fpath)
		# ========== Load the grids =========
		dsgrid, global_attrs = dsloader(data, dsn, dates)

		la = ([70.0,40.0, -14.0, 180.0])
		# ========== get the datatsets ==========
		ds_tc = (_Hansenfile(ppath, pptex, "treecover2000")).sel(dict(latitude=slice(la[0], la[1]),longitude=slice(la[2], la[3])))
		ds_ly = (_Hansenfile(ppath, pptex, "lossyear")).sel(dict(latitude=slice(la[0], la[1]),longitude=slice(la[2], la[3])))
		ds_dm = (_Hansenfile(ppath, pptex, "datamask")).sel(dict(latitude=slice(la[0], la[1]),longitude=slice(la[2], la[3])))
		# Add in the loss year
		DS_af =  xr.open_dataset(
			'/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/HANSEN/HansenMODIS_activefiremask.nc', 
			chunks={'latitude': 5000, "longitude": 5000}).sel(dict(latitude=slice(la[0], la[1]),longitude=slice(la[2], la[3])))
		for mwb in  mwbox:

			CLEAN = []

			for ds_af in [DS_af, None]:
				fto, cleanup, dsnmaskfn, HFCMclean = dsFRIcal(
					dsn, data, ds_tc, ds_ly, ds_dm, ds_af, 
					fpath, mwb, region, dates, tcf, force, dsgrid, global_attrs)

				for fncc in cleanup:
					CLEAN.append(fncc)
				ipdb.set_trace()

			print("perform the cleanup")
			ipdb.set_trace()
			for fnc in CLEAN:
				if os.path.isfile(fnc):
					os.remove(fnc)

			finalclean.append(dsnmaskfn)
			finalclean.append(HFCMclean)
	print("perform the cleanup")
	ipdb.set_trace()

	for fncc in finalclean:
		if os.path.isfile(fncc):
			os.remove(fncc)

	ipdb.set_trace()
	sys.exit()

#==============================================================================

#==============================================================================
def dsFRIcal(dsn, data, ds_tc, ds_ly, ds_dm, ds_af,  fpath, mwb, region, dates, tcf, force, dsgrid, global_attrs):
	"""
	This function will try a different approach to reindexing
	"""
	# ========== Create the outfile name ==========
	tnMSK = "%sHansen_GFC-2018-v1.6_regrided_%s_FRI_MSK_%stmp.nc" % (fpath, dsn, region)
	
	if ds_af is None:
		print("starting %s %d without MODIS Active Fire masking at:" % (dsn, mwb), pd.Timestamp.now())
		# +++++ Coursened and masked HGFC which is dsn and mwb independant +++++
		tnam0 = "%sHansen_GFC-2018-v1.6_regrided_FRI_firstmask_%stmp.nc" % (fpath, region)
		# +++++ the lon smoothed MWB tmp file +++
		tname = "%sHansen_GFC-2018-v1.6_regrided_%s_FRI_lon%ddegMW_%stmp.nc" % (fpath, dsn, mwb, region)
		# +++++ Final file name +++++
		fnout = "%sHansen_GFC-2018-v1.6_regrided_%s_FRI_%ddegMW_%s.nc" % (fpath, dsn, mwb, region)
	else:
		print("starting %s %d with MODIS Active Fire masking at:" % (dsn, mwb), pd.Timestamp.now())
		# +++++ Coursened and masked HGFC which is dsn and mwb independant +++++
		tnam0 = "%sHansen_GFC-2018-v1.6_regrided_FRI_firstmask_%stmpMAF.nc" % (fpath, region)
		# +++++ the lon smoothed MWB tmp file +++
		tname = "%sHansen_GFC-2018-v1.6_regrided_%s_FRI_lon%ddegMW_%stmpMAF.nc" % (fpath, dsn, mwb, region)
		# +++++ Final file name +++++
		fnout = "%sHansen_GFC-2018-v1.6_regrided_%s_FRI_%ddegMW_%sMAF.nc" % (fpath, dsn, mwb, region)
		# tnMSK = "%sHansen_GFC-2018-v1.6_regrided_%s_FRI_LONMSK%ddegMW_%stmp.nc" % (fpath, dsn, mwb, region)
	
	cleanup = []
	# ========== Check if a valid file already exists ==========
	if not os.path.isfile(fnout) or force:
		# ========== Calculate scale factors ==========
		rat = np.round(mwb / np.array(ds_tc["treecover2000"].attrs["res"]) )
		if np.unique(rat).shape[0] == 1:
			# the scale factor between datasets
			SF    = int(rat[0])
			GSF   = np.round(dsgrid.attrs["res"][0]/ds_tc["treecover2000"].attrs["res"][0]).astype(int)

			ASF   = np.round(mwb /(np.array(ds_tc["treecover2000"].attrs["res"])[0]*GSF))

			PF    = np.round(((ASF*GSF)/2)**2)
			# RollF = int(SF/4 - 0.5) # the minus 0.5 is correct for rolling windows
			RollF = 1
		else:
			warn.warn("Lat and lon have different scale factors")
			ipdb.set_trace()
			sys.exit()

		# ===== Create a masksum =====
		# This is so i can count the number of values that are valid in each location
		# This is inperfect but it ended up being the only way i could deal with the memory usage
		warn.warn("Uncomment the mask once testing is complete")
		mask_sum = ds_dm.chunk({'latitude': 1000, "longitude":1000}).astype("int32").coarsen(dim={"latitude":GSF, "longitude":GSF}, boundary="pad").sum()
		mask_sum = mask_sum.reindex({"longitude":dsgrid.longitude, "latitude":dsgrid.latitude}, method="nearest")
		mask_sum = tempNCmaker(
			mask_sum, tnMSK, "datamask", 
			chunks={'time':1, 'longitude': 1000, 'latitude': 1000}, skip=True)

		mask_sum = mask_sum.chunk({'time':1, 'longitude': 5000, 'latitude': 5000})
		mask_sum = mask_sum.rolling({"longitude":ASF}, center = True, min_periods=1).sum()
		mask_sum = mask_sum.rolling({"latitude":ASF}, center = True, min_periods=1).sum()
		# with ProgressBar():print(mask_sum.mean().compute())
		mask_sum = (mask_sum > PF)

		# ========== Calculate the amount of forest that was lost due to fire ==========
		if not ds_af is None:
			ds_ly = ds_ly.where(ds_af.rename({"fireloss":"lossyear"}).lossyear, 0)


		ba_ly = (ds_ly > 0).astype("float32")#.chunk({'latitude': 500, "longitude":500})

		# ========== implement the masks ==========
		ba_ly = ba_ly.where((ds_tc >  tcf).rename({"treecover2000":"lossyear"}))  # mask out the non forest
		ba_ly = ba_ly.where((ds_dm == 1.0).rename({"datamask":"lossyear"})) # mask out the non data pixels
		
		# +++++ Coursen the resolution in order to make analysis more memory efficent +++++
		ba_ly = ba_ly.coarsen(dim={"latitude":2, "longitude":2}, boundary="pad").mean()
		ba_ly.attrs = global_attrs
		ba_ly.attrs["res"] = ([
			abs(np.round(np.unique(np.diff(ba_ly.longitude.values))[0], decimals=4)),
			abs(np.round(np.unique(np.diff(ba_ly.latitude.values))[0], decimals=4))])
		ba_ly = tempNCmaker(ba_ly, tnam0, "lossyear", chunks={'latitude': 1}, skip=True)

		# +++++ make a new working scale factor +++++
		NSF = np.round(mwb /ba_ly.attrs["res"][0]).astype(int)
		
		# +++++ Moving window Smoothing +++++
		MW_lons = ba_ly.rolling({"longitude":NSF}, center = True, min_periods=RollF).mean()#.astype("float32")
		MW_lons = MW_lons.reindex({"longitude":dsgrid.longitude}, method="nearest")
		MW_lons = tempNCmaker(MW_lons, tname, "lossyear", chunks={'longitude': 1000})

		ipdb.set_trace()
		sys.exit()

		# +++++ Apply the second smooth +++++
		MW_FC    = MW_lons.rolling({"latitude":NSF}, center = True, min_periods=RollF).mean()
		MW_FC_RI = MW_FC.reindex({"latitude":DAin_sub.latitude}, method="nearest")


		# +++++ Fix the metadata +++++
		MW_FC_RI       = MW_FC_RI.rename({"lossyear":"lossfrac"})
		MW_FC_RI.attrs = global_attrs
		MW_FC_RI.attrs["history"]  = "%s: Fraction of burnt forest after a %d degree spatial smoothing, then resampled to match %s grid resolution using %s" % ((str(pd.Timestamp.now())), mwb, dsn, __file__) +MW_FC_RI.attrs["history"]
		MW_FC_RI.attrs["FileName"] = fnout
		MW_FC_RI.lossfrac.attrs    = ds_ly.lossyear.attrs	
		MW_FC_RI.latitude.attrs    = ds_ly.latitude.attrs		
		MW_FC_RI.longitude.attrs   = ds_ly.longitude.attrs


		# ========== Create the new layers ==========
		ipdb.set_trace()
		MW_FC_RI = MW_FC_RI.where(mask_sum["mask"].values > ((SF/2)**2))
		MW_FC_RI["FRI"] = (1/MW_FC_RI["lossfrac"].where(MW_FC_RI["lossfrac"]> 0)) * 18


		ipdb.set_trace()


		encoding = OrderedDict()
		for ky in ["lossfrac", "FRI"]:
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

	
		# cleanup.append(tnam0)
		cleanup.append(tname)	
		# cleanup.append(tnMSK)
	else:
		print("dataset for %d deg already exist. going to next window" % (mwb))
		# cleanup.append(tnam0)
		cleanup.append(tname)
		# cleanup.append(tnMSK)

	dsret = xr.open_dataset(fnout, chunks={'longitude': 1000}) 
	

	return dsret, cleanup, tnMSK, tnam0

#==============================================================================

def tempNCmaker(ds, fntmp, vname, pchunk=1000, chunks={'longitude': 1000, "latitude":1000}, skip=False, pro = "tmp"):

	""" Function to save out a tempary netcdf """
	# cf.pymkdir(tmppath)
	enc = {'shuffle':True,'zlib':True,'complevel':5}# "chunksizes":[1, 100, -]}
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

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
def dsloader(data, dsn, dates):
	"""Function to load and process data"""

	if data[dsn]["chunks"] is None:
		DS           = xr.open_dataset(data[dsn]["fname"])
	else:
		DS           = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])
	if 'time' in data[dsn]["chunks"].keys():
		DS         = DS.isel({'time':0})
	else:
		warn.warn("This has not been implemented yet")
		ipdb.set_trace()
		sys.exit()

	global_attrs = GlobalAttributes(DS, dsn)
		
	try:
		len(DS.attrs["res"])
	except KeyError:
		DS.attrs["res"] = ([
			abs(np.unique(np.diff(DS.longitude.values))[0]),
			abs(np.unique(np.diff(DS.latitude.values))[0]) ])
	except Exception as e:
		print("Unknown error", str(e))
		ipdb.set_trace()
		raise e
	return DS, global_attrs



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
	data["esacci"] = ({
		"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_FireCCI_2001_burntarea.nc",
		'var':"BA", "gridres":"250m", "region":"Asia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_landseamask.nc"
		})
	data["MODIS"] = ({
		"fname":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc",
		'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc"
		})
	data["COPERN_BA"] = ({
		'fname':"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/COPERN_BA/processed/COPERN_BA_gls_2014_burntarea_SensorGapFix.nc",
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":{'time':1, 'latitude': 1000},
		"rename":None#{"lon":"longitude", "lat":"latitude"}
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
	return data
	
#==============================================================================
if __name__ == '__main__':
	main()