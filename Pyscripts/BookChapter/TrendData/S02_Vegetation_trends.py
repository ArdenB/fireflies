"""
Script goal, to produce trends in netcdf files
This script can also be used in P03 if required

"""
#==============================================================================

__title__ = "Global Vegetation Trends"
__author__ = "Arden Burrell"
__version__ = "v1.0(28.03.2019)"
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
import xarray as xr
import dask.array as da
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
import glob
from scipy import stats
import statsmodels.stats.multitest as smsM
import myfunctions.PlotFunctions as pf

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cpf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import socket
# Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)
#==============================================================================
def main():
	# ========== Get the relevant datasets ==========
	data = datasets()
	# ========== loop over the datasets ==========
	for dsn in data:
		# ========== Create the components ==========
		var          = data[dsn]["var"]
		method       = "TheilSen"
		region       = data[dsn]["region"]
		grid         = data[dsn]["gridres"]
		# ========== open the dataset ==========
		ds           = xr.open_dataset(data[dsn]["fname"])#, chunks={"latitude":480})
		global_attrs = GlobalAttributes(ds, var)

		# ========== Make the filename ==========
		fout = './results/netcdf/%s_%s_%s_%s_to%d_%s%s.nc' % (
			dsn, var, method, 
			pd.to_datetime(ds.time.values).year.min(),
			pd.to_datetime(ds.time.values).year.max(), region, grid)
		if not (os.path.isfile(fout)):

			if dsn == 'COPERN':
				ds = ds.drop(["crs", "time_bnds"])
			# ========== check if the file uses the correct names ==========
			try:
				nl = ds.latitude.values.shape[0]
			except AttributeError:
				# rename the lats and lons
				ds    = ds.rename({"lat":"latitude", "lon":"longitude"})
				nl = ds.latitude.values.shape[0]
			

			# if ".ccrc.unsw.edu.au" in socket.gethostbyaddr(socket.gethostname())[0]:
			chunk = data[dsn]["chunk"]
			if chunk:
				Lcnks = 10
				nlats = ds.latitude.values.shape[0]
				nlons = ds.longitude.values.shape[0]
				dsc = ds.chunk({
					"latitude":int(nlats/Lcnks), 
					"longitude":int(nlons/Lcnks)})
				with ProgressBar():	
					dsout = nonparmetric_correlation(dsc, 'time').compute()
			else:
				dsc = ds
				print("starting trend calculation at:", pd.Timestamp.now())
				dsout = nonparmetric_correlation(dsc, 'time').compute()
			# else:
			# 	# ========== subset for smaller ram ==========
			# 	Lcnks = 40
			# 	mapdet = pf.mapclass("boreal")
			# 	dsc = ds.loc[dict(
			# 		longitude=slice(mapdet.bounds[0], mapdet.bounds[1]),
			# 		latitude=slice(mapdet.bounds[2], mapdet.bounds[3]))]
			# 	# ========== Set the number of chunks ==========
			# 	ipdb.set_trace()
			# 	nlats = dsc.latitude.values.shape[0]
			# 	nlons = dsc.longitude.values.shape[0]
			# 	dsc = dsc.chunk({
			# 		"latitude":int(nlats/Lcnks), 
			# 		"longitude":int(nlons/Lcnks)})

			# ========== Calculates the amount of area ==========

			try:
				# ========== Pull out the individual arrays and start sorting ==========
				trends = []
				kys   = ["slope", "intercept", "rho", "pvalue"]
				for num in range(0, 4):	
					trends.append(dsout[data[dsn]["var"]].isel(slope=num).values)#rename(kys[num])

				trends, kys = MultipleComparisons(trends, kys, aplha = 0.10)

				# ========== Build a new dataarray ==========
				layers, encoding = dsmaker(ds, data[dsn]["var"], trends, kys, method)
				ds_trend         = xr.Dataset(layers, attrs= global_attrs)

				print("Starting write of data")
				ds_trend.to_netcdf(fout, 
					format         = 'NETCDF4', 
					encoding       = encoding,
					unlimited_dims = ["time"])
			except Exception as e:
				print(e)
				warn.warn(" \n something went wrong with the save, going interactive")
				ipdb.set_trace()

		ipdb.set_trace()

#==============================================================================
# ============================ xarray nonparmetric ============================
#==============================================================================
# @jit(nogil=True)
def scipyTheilSen(array):
	"""
	Function for rapid TheilSen slop estimation with time. 
	the regression is done with  an independent variable 
	rangeing from 0 to array.shape to make the intercept 
	the start which simplifies calculation
	
	args:
		array 		np : numpy array of annual max VI over time 
	return
		result 		np : slope, intercept
	"""
	try:
		if bn.allnan(array):
			return np.array([np.NAN, np.NAN, np.NAN, np.NAN])

		slope, intercept, _, _ = stats.mstats.theilslopes(
			array, np.arange(array.shape[0]))
		rho, pval = stats.spearmanr(
			array, np.arange(array.shape[0]))
		# change = (slope*array.shape[0])
		return np.array([slope, intercept, rho, pval])

	except:
		# print(e)
		# warn.warn("unhandeled Error has occured")
		return np.array([np.NAN, np.NAN, np.NAN, np.NAN])
		# return np.NAN


def nonparmetric_correlation(array, dim='time' ):
    return xr.apply_ufunc(
        scipyTheilSen, array, 
        input_core_dims=[[dim]],
        vectorize=True,
        # dask="allowed",
        dask='parallelized',
        # output_dtypes=[float, float, float, float],
        output_dtypes=[float],
        output_core_dims=[['slope']], #, ['intercept'], ['rho'], ['pvalue']
        output_sizes=({"slope":4})
        )

#==============================================================================
# ========================== Other usefull functions ==========================
#==============================================================================

def GlobalAttributes(ds, var):
	"""
	Creates the global attributes for the netcdf file that is being written
	these attributes come from :
	https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/metadata/DataDiscoveryAttConvention.html
	args
		ds: xarray ds
			Dataset containing the infomation im intepereting
		var: str
			name of the variable
	returns:
		attributes 	Ordered Dictionary cantaining the attribute infomation
	"""
	# ========== Create the ordered dictionary ==========
	attr = OrderedDict()

	# fetch the references for my publications
	# pubs = puplications()
	
	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["title"]               = "Trend in Climate (%s)" % (var)
	attr["summary"]             = "Annual and season trends in %s" % var
	attr["Conventions"]         = "CF-1.7"
	
	# ++++++++++ Data Provinance ++++++++++ 
	attr["history"]             = "%s: Netcdf file created using %s (%s):%s by %s" % (
		str(pd.Timestamp.now()), __title__, __file__, __version__, __author__)
	attr["history"]            += ds.history

	attr["creator_name"]        = __author__
	attr["creator_url"]         = "ardenburrell.com"
	attr["creator_email"]       = __email__
	attr["institution"]         = "University of Leicester"
	attr["date_created"]        = str(pd.Timestamp.now())
	
	# ++++++++++ Netcdf Summary infomation ++++++++++ 
	attr["time_coverage_start"] = str(dt.datetime(ds['time.year'].min(), 1, 1))
	attr["time_coverage_end"]   = str(dt.datetime(ds['time.year'].max() , 12, 31))
	


	return attr

def dsmaker(ds, var, results, keys, method):
	"""
	Build a summary of relevant paramters
	args
		ds: xarray ds
			Dataset containing the infomation im intepereting
		var: str
			name of the variable
	return
		ds 	xarray dataset
	"""
	# sys.exit()
	tm = [dt.datetime(ds['time.year'].max() , 12, 31)]
	times = OrderedDict()
	# tm    = [dt.datetime(yr , 12, 31) for yr in start_years]
	# tm    = [date]
	times["time"] = pd.to_datetime(tm)

	times["calendar"] = 'standard'
	times["units"]    = 'days since 1900-01-01 00:00'
	
	times["CFTime"]   = date2num(
		tm, calendar=times["calendar"], units=times["units"])

	dates = times["CFTime"]

	try:
		lat = ds.lat.values
		lon = ds.lon.values
	except AttributeError:
		lat = ds.latitude.values
		lon = ds.longitude.values
	# dates = [dt.datetime(yr , 12, 31) for yr in start_years]
	# ipdb.set_trace()
	# ========== Start making the netcdf ==========
	layers   = OrderedDict()
	encoding = OrderedDict()
	# ========== loop over the keys ==========
	for pos in range(0, len(keys)): 
		try:
			# ipdb.set_trace()
			if type(results[0]) == np.ndarray:
				Val = results[pos][np.newaxis,:, :]
			else:
				# multiple variables
				Val = np.stack([res[pos] for res in results]) 
			ky = keys[pos]

			# build xarray dataset
			DA=xr.DataArray(Val,
				dims = ['time', 'latitude', 'longitude'], 
				coords = {'time': dates,'latitude': lat, 'longitude': lon},
				attrs = ({
					'_FillValue':9.96921e+36,
					'units'     :"1",
					'standard_name':ky,
					'long_name':"%s %s" % (method, ky)
					}),
			)

			DA.longitude.attrs['units'] = 'degrees_east'
			DA.latitude.attrs['units']  = 'degrees_north'
			DA.time.attrs["calendar"]   = times["calendar"]
			DA.time.attrs["units"]      = times["units"]
			layers[ky] = DA
			encoding[ky] = ({'shuffle':True, 
				# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
				'zlib':True,
				'complevel':5})
		
		except Exception as e:
			warn.warn("Code failed with: \n %s \n Going Interactive" % e)
			ipdb.set_trace()
			raise e
	return layers, encoding

def MultipleComparisons(trends, kys, aplha = 0.10, MCmethod="fdr_bh"):
	"""
	Takes the results of an existing trend detection aproach and modifies them to
	account for multiple comparisons.  
	args
		trends: list
			list of numpy arrays containing results of trend analysis
		kys: list 
			list of what is in results
		years:
			years of accumulation 
	
	"""
	if MCmethod == "fdr_by":
		print("Adjusting for multiple comparisons using Benjamini/Yekutieli")
	elif MCmethod == "fdr_bh":
		print("Adjusting for multiple comparisons using Benjamini/Hochberg")
	else:
		warn.warn("unknown MultipleComparisons method, Going Interactive")
		ipdb.set_trace()



	# ========== Locate the p values and reshape them into a 1d array ==========
	# ++++++++++ Find the pvalues ++++++++++
	index      = kys.index("pvalue")
	pvalue     = trends[index]
	isnan      = np.isnan(pvalue)
	
	# ++++++++++ pull out the non nan pvalus ++++++++++
	# pvalue1d = pvalue.flatten()
	pvalue1d   = pvalue[~isnan]
	# isnan1d  = isnan.flatten()
	
	# =========== Perform the MC correction ===========
	pvalue_adj =  smsM.multipletests(pvalue1d, method=MCmethod, alpha=alpha)
	
	# ++++++++++ reformat the data into array ++++++++++
	MCR =  ["Significant", "pvalue_adj"]
	for nm in MCR:
		# make an empty array
		re    = np.zeros(pvalue.shape)
		re[:] = np.NAN
		if nm == "Significant":
			re[~isnan] = pvalue_adj[MCR.index(nm)].astype(int).astype(float)
		else:
			re[~isnan] = pvalue_adj[MCR.index(nm)]
		
		# +++++ add the significant and adjusted pvalues to trends+++++
		trends.append(re)
		kys.append(nm)
	return trends, kys


def datasets():
	"""
	Create the summary of the datasets to be analyised
	"""


	data= OrderedDict()

	data["GIMMS31v11"] = ({
		'fname':"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1982to2017_annualmax.nc",
		'var':"ndvi", "gridres":"GIMMS", "region":"Global", "Periods":["AnnualMax"], "chunk":True
		})
	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_AnnualMax_1999to2018_global_at_1km_compressed.nc",
		'var':"NDVI", "gridres":"COPERN", "region":"Global", "Periods":["AnnualMax"], "chunk":True
		})
	# data["MODISaqua"] = ({
	# 	'fname': sorted(glob.glob("./data/veg/MODIS/aqua/processed/MYD13Q1_A*_final.nc"))[1:],
	# 	'var':"ndvi", "gridres":"MODIS", "region":"Siberia", "Periods":["All"], "chunk":True
	# 	})
	data["GIMMS31v10"] = ({
		'fname':"./data/veg/GIMMS31g/3.GLOBAL.GIMMS31.1982_2015_AnnualMax.nc",
		'var':"ndvi", "gridres":"GIMMS", "region":"Global", "Periods":["AnnualMax"], "chunk":False
		})
	return data


#==============================================================================
# ========================== Call the main functions ==========================
#==============================================================================

if __name__ == '__main__':
	main()