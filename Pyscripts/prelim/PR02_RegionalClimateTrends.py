"""
Prelim script for looking at netcdf files and producing some trends

"""
#==============================================================================

__title__ = "Climate trends"
__author__ = "Arden Burrell"
__version__ = "v1.0(28.01.2019)"
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
from netCDF4 import Dataset, num2date 
from scipy import stats
import xarray as xr
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
# Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================
def main():
	# Load in the netcdf data
	# ds = xr.open_dataset("./data/cli/1.TERRACLIMATE/1.TERRA.tmean.1980.2017v2_RUSSIA_yearmean.nc")
	# ds = xr.open_dataset("./data/cli/1.TERRACLIMATE/1.TERRA.tmean.1980.2017v2_RUSSIA_seasmean.nc")
	data= OrderedDict()
	data["pre"] = ({
		'fname':"./data/cli/1.TERRACLIMATE/TerraClimate_merged_1980to2017_ppt_Russia.nc",
		'var':"ppt"
		})
	data["tas"] = ({
		'fname':"./data/cli/1.TERRACLIMATE/1.TERRA.tmean.1980.2017v2_RUSSIA.nc",
		'var':"tmean"
		})
	for dt in data:
		# ipdb.set_trace()
		trendmapper(data[dt]["fname"], data[dt]["var"])



	 # Reshape to an array with as many rows as years and as many columns as there are pixels
	
	ipdb.set_trace()

#==============================================================================
def trendmapper(fname, var, fdpath=""):
	ds = xr.open_dataset(fname)
	RFinfo = Field_data()#fdpath, den="sDens2017modis")
	seasons = ["Annual", "DJF", "MAM", "JJA", "SON"]
	results = []
	# ========== Create the global attributes ==========
	global_attrs = GlobalAttributes(ds, var)


	for season in seasons:
		# Get the months in the season
		if season == "Annual":
			man_annual = ds[var].groupby('time.year')
		else:
			man_annual = ds[var].where(ds[var]['time.season'] == season).groupby('time.year')
		

		if var == "tmean":
			annual = man_annual.mean(dim='time')
		else:
			annual = man_annual.sum(dim='time')


		trends = _fitvals(annual)
		results.append(trends)

		# a test plot
		# annual.plot(col='year', col_wrap=6)
		# plt.show() 

	layers, encoding = dsmaker(ds, var, results, seasons)
	ds_trend = xr.Dataset(layers, attrs= global_attrs)
	# Build all the plots
	# plt.figure(num=var+r"$_{max}$"+" trend", figsize=(12, 6))
	for num in range(0, len(seasons)):
		ax = plt.subplot(2, 3, num+1, projection=ccrs.PlateCarree())
		# ax = plt.subplot(1, 5, num+1, projection=ccrs.PlateCarree())
		# ax = plt.axes(projection=ccrs.PlateCarree())
		ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
		ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
		ax.add_feature(cpf.RIVERS, zorder=104)
		# add lat long linse
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
			linewidth=1, color='gray', alpha=0.5, linestyle='--')
		gl.xlabels_top = False
		gl.ylabels_right = False
		# if not (num >=2):
		# 	gl.xlabels_bottom = False

		if not (num in [0, 3]):
			gl.ylabels_left = False


		gl.xformatter = LONGITUDE_FORMATTER
		gl.yformatter = LATITUDE_FORMATTER

		# ========== create the colormap ==========
		if var == "tmean":
			vmax =  0.07
			vmin = -0.07
			cmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors)
		elif var =="ppt":
			vmin = -3.0
			vmax =  3.0
			cmap = mpc.ListedColormap(palettable.cmocean.diverging.Curl_20_r.mpl_colors)
		# ========== Make the map ==========
		# ipdb.set_trace()
		ds_trend[seasons[num]].drop('time').plot(ax=ax, transform=ccrs.PlateCarree(),
			cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs={
			"extend":"both",  "pad":0.0075,"fraction":0.125, "shrink":0.74}) #"fraction":0.05,
		# ax.set_title=seasons[num]
		# for vas, cl in zip(RFinfo.RF17.unique().tolist(), ['yx', "r*","k."]):
		ax.plot(RFinfo.lon.values, RFinfo.lat.values, 
			"kx", markersize=4, transform=ccrs.PlateCarree())


	
	plt.subplots_adjust(
		top=0.98,
		bottom=0.02,
		left=0.038,
		right=0.989,
		hspace=0.05,
		wspace=0.037)
	fig = plt.gcf()
	fig.set_size_inches(18.5, 8.5)
	# plt.tight_layout()

	plt.show()

	# ipdb.set_trace()


		# get the value


def _fitvals(annual):
	vals  = annual.values 
	years = annual.year.values

	vals2 = vals.reshape(len(years), -1)
	vals2[np.isnan(vals2)] = 0
	# Do a first-degree polyfit
	regressions = np.polyfit(years, vals2, 1)
	# Get the coefficients back
	trends = regressions[0,:].reshape(vals.shape[1], vals.shape[2])
	return trends
#==============================================================================
def GlobalAttributes(ds, var):
	"""
	Creates the global attributes for the netcdf file that is being written
	these attributes come from :
	https://www.unidata.ucar.edu/software/thredds/current/netcdf-java/metadata/DataDiscoveryAttConvention.html
	args:
		runinfo		Table containing all the details of the individual runs
		ensinfo		Custom class object containing all the infomation about 
					the ensemble being saved
	returns:
		attributes 	Ordered Dictionary cantaining the attribute infomation
	"""
	# ========== Create the ordered dictionary ==========
	attr = OrderedDict()

	# fetch the references for my publications
	# pubs = puplications()
	
	# ========== Fill the Dictionary ==========

	# ++++++++++ Highly recomended ++++++++++ 
	attr["title"]               = "Trend in Climate Variable"
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
	
	# Note. Maybe ad some geographich infomation here


	# Add publication references
	# attr["references"]          = "Results are described in:  %s 	\n  TSS-RESTREND method is described in: %s" % (
	# 	pubs["p%d" % ensinfo.paper], pubs["p1"])
	
	# ++++++++++ Infomation unique to TSS-RESREND ensembles ++++++++++
	# attr["package_version"]     = ",".join(runinfo.iloc[ensinfo.run]["TSS.version"].unique().tolist())
	# attr["package_url"]         = "https://cran.r-project.org/web/packages/TSS.RESTREND/index.html"
	# attr["Vegetation"]          = ",".join(runinfo.iloc[ensinfo.run]["VI.type"].unique().tolist()) 
	# attr["Precipitation"]       = ",".join(runinfo.iloc[ensinfo.run]["rf.type"].unique().tolist()) 

	# ===== Check and see if temperature is included =====
	# if not all(pd.isnull(runinfo.iloc[ensinfo.run].Temperature.tolist())):
	# 	# +++++ contains temperature +++++
	# 	# List of temperature infomation
	# 	temp = ([fnm.split("/")[3].split(".")[1] 
	# 		for fnm in runinfo.iloc[ensinfo.run].Temperature.unique().tolist()])
	# 	# join list to attr
	# 	attr["Temperature"]      = ",".join(temp)

	# # ===== add infomation about CO2 fertilisation =====
	# if (ensinfo.paper == 3) or (ensinfo.paper>=5):
	# 	attr["CO2_method"]       = "Franks et al., (2013) CO2 correction"
	# 	attr["CO2_data"]         = "CMIP5 rcp8.5 forcing data"

	return attr


#==============================================================================

def dsmaker(ds, var, results, seasons):
	"""
	Build a summary of relevant paramters
	args:
		- 	ensinfo
		- 	sig masking
	return
		ds 	xarray dataset
	"""

	date = [dt.datetime(ds['time.year'].max() , 12, 31)]

	# ========== Start making the netcdf ==========
	layers   = OrderedDict()
	encoding = OrderedDict()

	for season, Val in zip(seasons, results):
		# gett the column
		# Grab the data
		# Val  = cf.immkr(results, va_col, plot=False, ret=True)  
		# set the string values
		desc = "trend in %s %s" % (season, var)

		# build xarray dataset
		# ipdb.set_trace()
		DA=xr.DataArray(Val[np.newaxis,:,:],
			dims = ['time', 'latitude', 'longitude'], 
			coords = {'time': date ,'latitude': ds.lat.values, 'longitude': ds.lon.values},
			attrs = ({
				'_FillValue':9.96921e+36,
				'units'     :"1",
				'standard_name':var,
				'long_name':"Trend in %s %s" % (season, var)
				}),
		)

		DA.longitude.attrs['units'] = 'degrees_east'
		DA.latitude.attrs['units']  = 'degrees_north'
		layers[season] = DA
		encoding[season] = ({'shuffle':True, 
			# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
			'zlib':True,
			'complevel':5})
	
	return layers, encoding


def Field_data(den="sDens2017modis"):
	"""
	# Aim of this function is to look at the field data a bit

	To start it just opens the file and returns the lats and longs 
	i can then use these to look up netcdf fils
	"""
	# ========== Load in the relevant data ==========
	fsum = pd.read_csv("./data/field/RF_catsum.csv")
	fsum.sort_values(by=["sn"],inplace=True) 
	fcut = fsum[fsum.sn<64]
	# fd18 = pd.read_csv(fdpath)
	fd17 = pd.read_csv("./data/field/2017data/siteDescriptions.csv")

	# ========== Create and Ordered Dict for important info ==========
	info = OrderedDict()
	info["sn"] = fd17["site number"]
	info["lat"] = fd17.strtY
	info["lon"] = fd17.strtX
	
	# ========== function to return nan when a value is missing ==========
	def _missingvalfix(val):
		try:
			return float(val)
		except Exception as e:
			return np.NAN

	def _fireyear(val):
		try:
			year = float(val)
			if (year <= 2018):
				return year
			else:
				return np.NAN
		except ValueError: #not a simple values
			try:
				return float(str(val[0]).split(" and ")[-1])
			except Exception as e:
				ipdb.set_trace()
				print(e)
				return np.NAN

	info[den] = [_missingvalfix(
		fcut[fcut.sn == sn][den].values) for sn in info['sn']]

	info["RF17"] = [_missingvalfix(
		fcut[fcut.sn == sn]["RF2017"].values) for sn in info['sn']]
	
		
	info["fireyear"] = [_fireyear(
		fd17[fd17["site number"] == sn]["estimated fire year"].values) for sn in info['sn']]
	# ========== Convert to dataframe and replace codes ==========
	RFinfo = pd.DataFrame(info)
	RFinfo["RF17"].replace(0.0, "AR", inplace=True)
	RFinfo["RF17"].replace(1.0, "RF", inplace=True)
	RFinfo["RF17"].replace(2.0, "IR", inplace=True)
	RFinfo["YearsPostFire"] = 2017.0 - RFinfo.fireyear
	return RFinfo
#==============================================================================
if __name__ == '__main__':
	main()