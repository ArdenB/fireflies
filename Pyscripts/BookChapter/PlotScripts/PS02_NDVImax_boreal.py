"""
Script goal, to produce plots of multiple NDVI datasets

"""
#==============================================================================

__title__ = "Site Vegetation data"
__author__ = "Arden Burrell"
__version__ = "v1.0(04.04.2019)"
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
from scipy import stats
import xarray as xr
from numba import jit
import bottleneck as bn
import scipy as sp
import glob
import myfunctions.PlotFunctions as pf
import myfunctions.corefunctions as cf

# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================
def main():
	# +++++ set up the path +++++
	fpath = "./plots/bookchapter/firstdraft/NDVImax/"
	cf.pymkdir(fpath)

	# ========== get the datasets ==========
	data = datasets()
	for region in ["global"]:
		fname = fpath+ "%s_boreal_NDVImax.csv" % region
		NDVIts = NDVIpuller(fname, data, region)

	

	ipdb.set_trace()


#==============================================================================
# ========================== The NDVI value function ==========================
#==============================================================================
def NDVIpuller(fname, data, region):
	"""
	takes the NDVI data and makes a saveable CSV file 
	args:
		fame: 	str
			the name of the file to be saved
		data:	OD
			dictionary with dataset infomation
	returns:
		dataframe with relevant data
	"""

	NDVI = OrderedDict()

	for dsn in data:
		print(dsn)
		if not data[dsn]["region"] == region:
			if region == "global":
				continue
			else:
				ipdb.set_trace()

		var = data[dsn]['var']

		# ========== open and modify the dataset ==========
		ds = xr.open_dataset(data[dsn]["fname"])
		if dsn == 'COPERN':
			ds = ds.drop(["crs", "time_bnds"]).rename({"lat":"latitude", "lon":"longitude"})
		elif dsn == "GIMMS31v10":
			ds = ds.drop(["percentile", "time_bnds"]).rename({"lat":"latitude", "lon":"longitude"})
			ds[var] = (ds[var].where(ds[var] >= 0)/10000.0)
		# ========== open the mask dataset ==========
		mask = xr.open_dataset(
		"./data/other/ForestExtent/BorealForestMask_%s.nc"%(data[dsn]["gridres"]))

		def _timefixer(time):
			""" 
			function to fix the dates of the netcdf files 
			it changes the months so they are the same """
			pdtime = pd.to_datetime(time)
			year   = pdtime.year

			# +++++ set up the list of dates +++++
			dates = OrderedDict()
			tm    = [dt.datetime(int(yr) , int(6), int(30)) for yr in year]
			dates = pd.to_datetime(tm)
			return dates

		dates = _timefixer(ds[var].time.values)
		try: 
			ds["time"] = dates
		except:
			warn.warn("Time setting did not work")
			ipdb.set_trace()

		# ========== multiple by the mask ==========
		if (ds.nbytes * 1e-9) <  8:
			# ========== mask the values ==========
			ds[var] *= mask.BorealForest.values
			# ========== Get the data ==========
			NDVI[dsn] = ds[data[dsn]['var']].mean(dim=["latitude", "longitude"]).to_pandas()

		else:
			# ========== Build an empty array  ==========
			tmeans    = np.zeros(ds.time.shape[0])
			tmeans[:] = np.NAN
			t0 = pd.Timestamp.now()
			# ========== chunk and mask the NDVI values ==========
			ds = ds.chunk({"time":1})
			DA = ds[var].where(mask.BorealForest.values == 1)
			# NDVI[dsn] = DA.mean(dim=["latitude", "longitude"]).compute()
			# ipdb.set_trace()
			for num in range(0, ds.time.shape[0]):
			# 	# print the line
				_lnflick(num, ds.time.shape[0], t0, lineflick=1)

			# 	# Get the Dataarray
				# DA = ds[var].isel(time=num).values
			# 	ds[var] *= mask.BorealForest.values
				tmeans[int(num)] = bn.nanmean(DA.isel(time=num).values)
			# 	DA = None
			NDVI[dsn] = xr.DataArray(tmeans,dims = ['time'],coords = {'time': ds.time}).to_pandas()
	# ========== Make metadata infomation ========== 
	df = pd.DataFrame(NDVI)
	if not (fname is None):
		df.to_csv(fname)
		maininfo = "Data Saved using %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, fname, gitinfo]
		cf.writemetadata(fname, infomation)
	
	return df
#==============================================================================
# ========================== Other usefull functions ==========================
#==============================================================================
def _lnflick(line, line_max, t0, lineflick=100000):
	if (line % lineflick == 0):
		string = ("\rLine: %d of %d" % 
					(line, line_max))
		if line > 0:
			# TIME PER LINEFLICK
			lfx = (pd.Timestamp.now()-t0)/line
			lft = str((lfx*lineflick))
			trm = str(((line_max-line)*(lfx)))
			string += (" t/%d lines: %s. ETA: %s" % (
				lineflick,lft, trm) )
			
		sys.stdout.write(string)
		sys.stdout.flush()
	else:
		pass

def datasets():
	"""
	Create the summary of the datasets to be analyised
	"""


	data= OrderedDict()

	data["GIMMS31v11"] = ({
		'fname':"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_annualmax.nc",
		'var':"ndvi", "gridres":"GIMMS", "region":"global", "Periods":["AnnualMax"]
		})
	data["MODISaqua"] = ({
		'fname': sorted(glob.glob("./data/veg/MODIS/aqua/processed/MYD13Q1_A*_final.nc"))[1:],
		'var':"ndvi", "gridres":"MODIS", "region":"Siberia", "Periods":["All"]
		})
	data["GIMMS31v10"] = ({
		'fname':"./data/veg/GIMMS31g/3.GLOBAL.GIMMS31.1982_2015_AnnualMax.nc",
		'var':"ndvi", "gridres":"GIMMS", "region":"global", "Periods":["AnnualMax"]
		})
	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
		'var':"NDVI", "gridres":"COPERN", "region":"global", "Periods":["AnnualMax"]
		})
	return data


if __name__ == '__main__':
	main()