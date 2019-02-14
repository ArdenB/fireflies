"""
Prelim script for looking at netcdf files and producing some trends

These estimates can also be used for P03 climate estimation

"""
#==============================================================================

__title__ = "Global Climate Trends"
__author__ = "Arden Burrell"
__version__ = "v1.0(13.02.2019)"
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
		'fname':"./data/cli/1.TERRACLIMATE/TerraClimate_OptimalAccumulllatedppt_1960to2017_GIMMS.nc",
		'var':"ppt"
		})
	data["tas"] = ({
		'fname':"./data/cli/1.TERRACLIMATE/1.TERRA.tmean.1980.2017v2_RUSSIA.nc",
		'var':"tmean"
		})
	for dt in data:
		# st_yrs = [1982, 1970]#, 1960]
		st_yrs = [1960, 1970, 1982, 1990, 1999]
		plot = False
		# ipdb.set_trace()
		# polyfit
		trendmapper(
			data[dt]["fname"], data[dt]["var"], 
			"scipyols", st_yrs, plot = plot)#, force=True)
		trendmapper(
			data[dt]["fname"], data[dt]["var"], 
			"polyfit", st_yrs, plot = plot)#, force=True)
		trendmapper(
			data[dt]["fname"], data[dt]["var"], 
			"theilsen", st_yrs, plot = plot)#, force=True)


	 # Reshape to an array with as many rows as years and as many columns as there are pixels
	
	ipdb.set_trace()

#==============================================================================
def trendmapper(fname, var, method,start_years, endyr = 2015, fdpath="", force = False, plot=True):


	
	ds = xr.open_dataset(fname)
	fout = './results/netcdf/TerraClimate_%s_%sto%d.nc' % ( var, method, endyr)
	if all([os.path.isfile(fout), not force]):
		warn.warn("Loading existing file, force is needed to overwrite")
		ds_trend = xr.open_dataset(fout)
		kys = [n for n in ds_trend.data_vars]
	else:

		results     = []
		# ========== Create the global attributes ==========
		global_attrs = GlobalAttributes(ds, var)

		# trends = _fitvals(dst)
		# trends = _fitvals(dst[var], method="theilsen")


		for styr in start_years:
			dst = ds[var].sel(time=slice('%d-01-01' % styr, '%d-12-31' % endyr))
			trends, kys = _fitvals(dst, method=method)
			results.append(trends)


		layers, encoding = dsmaker(ds, var, results, kys, start_years, method)
		ds_trend = xr.Dataset(layers, attrs= global_attrs)


		try:
			print("Starting write of data")
			ds_trend.to_netcdf(fout, 
				format         = 'NETCDF4', 
				encoding       = encoding,
				unlimited_dims = ["time"])
		except Exception as e:
			print(e)
			warn.warn(" \n something went wrong with the save, going interactive")
			ipdb.set_trace()
	
	# ========== Build all the plots ==========

	if not plot:
		return True
	pn = 1

	for styp in range(0, len(start_years)):
		for num in range(0, len(kys)):
			# ========== create the colormap ==========
			cmap, vmin, vmax = cbvals(var, kys[num])
			if cmap is None:
				# ipdb.set_trace()
				warn.warn("no colorbar exists for %s, skipping" % (kys[num]))
				continue

			print(styp, num)
			ax = plt.subplot(len(start_years),len(kys), pn, projection=ccrs.PlateCarree())
			ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
			ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
			ax.add_feature(cpf.RIVERS, zorder=104)
			# add lat long linse
			gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				linewidth=1, color='gray', alpha=0.5, linestyle='--')
			gl.xlabels_top = False
			gl.ylabels_right = False
			if num == 0:
				gl.xlabels_bottom = False

			if not ((pn-1) % len(start_years)):
				gl.ylabels_left = False

			gl.xformatter = LONGITUDE_FORMATTER
			gl.yformatter = LATITUDE_FORMATTER


			# ========== Make the map ==========
			# ipdb.set_trace()
			ds_trend[kys[num]].isel(time=styp).plot(ax=ax, transform=ccrs.PlateCarree(),
				cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs={
				"extend":"both"})#,  "pad":0.0075,"fraction":0.125, "shrink":0.74}) #"fraction":0.05,
			pn += 1
			# ax.set_title=seasons[num]
			# for vas, cl in zip(RFinfo.RF17.unique().tolist(), ['yx', "r*","k."]):
			# ax.plot(RFinfo.lon.values, RFinfo.lat.values, 
			# 	"kx", markersize=4, transform=ccrs.PlateCarree())


	
	# plt.subplots_adjust(
	# 	top=0.98,
	# 	bottom=0.02,
	# 	left=0.038,
	# 	right=0.989,
	# 	hspace=0.05,
	# 	wspace=0.037)
	# fig = plt.gcf()
	# fig.set_size_inches(18.5, 8.5)
	# plt.tight_layout()

	plt.show()

	ipdb.set_trace()


		# get the value
def cbvals(var, ky):
	if ky == "slope":
		if var == "tmean":
			vmax =  0.07
			vmin = -0.07
			cmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors)
		elif var =="ppt":
			vmin = -3.0
			vmax =  3.0
			cmap = mpc.ListedColormap(palettable.cmocean.diverging.Curl_20_r.mpl_colors)
		else:
			warn.warn("unknown variable")
			ipdb.set_trace()
	elif ky == "pvalue":
		cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20.mpl_colormap)
		vmin = 0.0
		vmax = 1.0
	elif ky == "rsquared":
		# cmap = mpc.ListedColormap(palettable.matplotlib,Viridis_20.mpl_colors)
		# plt.viridis
		vmin = 0.0
		vmax = 1.0
		# cmap =  
	else:
		cmap = None
		vmin = None
		vmax = None
	# elif ky == "intercept":

	return cmap, vmin, vmax

def _fitvals(dvt, method="polyfit"):
	"""
	Takes the ds[var] and performs some form of regression on it
	"""
	vals  = dvt.values 
	years = pd.to_datetime(dvt.time.values).year
	vals2 = vals.reshape(len(years), -1)

	# Do a first-degree polyfit
	t0 = pd.Timestamp.now()
	print("testing with %s from %d to %d starting at: %s" % (
		method, pd.to_datetime(dvt.time.values).year.min(), 
		pd.to_datetime(dvt.time.values).year.max(), str(t0)))

	if method=="polyfit":
		vals2[np.isnan(vals2)] = 0
		regressions = np.polyfit(years, vals2, 1)
		regressions[regressions== 0] = np.NAN
		trends = [regressions[0,:].reshape(vals.shape[1], vals.shape[2])]
		kys = ["slope"]
		# stacked = dvt.stack(allpoints=['latitude','longitude'])
		# trend = stacked.groupby('allpoints').apply(scipyolsXR)
		# trends = trend.unstack('allpoints')
		# trend = stacked.groupby('allpoints').apply(linear_trend)
		# kys = ["slope", "intercept", "rsquared", "pvalue"]
	elif method == "theilsen":
		regressions = alongaxFAST(vals2, scipyTheilSen)
		trds = regressions.reshape(4, vals.shape[1], vals.shape[2])
		trends = []
		for n in range(0, trds.shape[0]):
			trends.append(trds[n, :, :])
		kys = ["slope", "intercept", "rho", "pvalue"]
	elif method == "scipyols":
		# regressions = alongax(vals2, scipyols)
		regressions = alongaxFAST(vals2, scipyols)
		trds = regressions.reshape(4, vals.shape[1], vals.shape[2])
		trends = []
		for n in range(0, trds.shape[0]):
			trends.append(trds[n, :, :])

		kys = ["slope", "intercept", "rsquared", "pvalue"]

	tdelta = pd.Timestamp.now() - t0
	print("\n Time taken to get regression coefficients using %s: %s" % (method, str(tdelta)))
	# ipdb.set_trace()
	return trends, kys


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
	


	return attr


#==============================================================================

def dsmaker(ds, var, results, keys, start_years, method):
	"""
	Build a summary of relevant paramters
	args:
		- 	ensinfo
		- 	sig masking
	return
		ds 	xarray dataset
	"""
	# sys.exit()
	# date = [dt.datetime(ds['time.year'].max() , 12, 31)]
	dates = pd.to_datetime([dt.datetime(yr , 12, 31) for yr in start_years])
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
		layers[ky] = DA
		encoding[ky] = ({'shuffle':True, 
			# 'chunksizes':[1, ensinfo.lats.shape[0], 100],
			'zlib':True,
			'complevel':5})
	
	return layers, encoding


# @jit
def alongaxFAST(array, myfunc, lineflick=10000):
	""" Fastest wave i've yet found to loop over an entire netcdf file
	array 2d numpy array
	myfunc function i want to apply
	lineflick frequency that i want to see the lines, increasing this number 
		increases speed
	returns
		res 2d array with the results
	"""
	# build an empyt array to hold the result
	# res = np.zeros((array.shape[1], 4))
	res = np.zeros((4, array.shape[1]))
	res[:] = np.NAN

	# locate and remove any nan rows
	ana = ~bn.anynan(array, axis=0)
	array2 = array[:, ana]

	# build a holder
	vals = np.zeros((4, array2.shape[1]))

	for line in range(0, array2.shape[1]):
		if (line % lineflick == 0):
			string = ("\rcalculating regression for line: %d of %d" % 
						(line, array2.shape[1]))
			sys.stdout.write(string)
			sys.stdout.flush()

		out = myfunc(array2[:, line])		
		# vals.append(out)
		vals[:, line] = out
	res[:, ana] = vals
	return res

# @jit
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
		# if bn.allnan(array):
		# 	return np.array([np.NAN, np.NAN, np.NAN, np.NAN])

		slope, intercept, _, _ = stats.mstats.theilslopes(
			array, np.arange(array.shape[0]))
		rho, pval = stats.spearmanr(
			array, np.arange(array.shape[0]))
		# change = (slope*array.shape[0])
		return np.array([slope, intercept, rho, pval])

	except Exception as e:
		print(e)
		warn.warn("unhandeled Error has occured")
		ipdb.set_trace()
		return np.array([np.NAN, np.NAN, np.NAN, np.NAN])

# @jit
def scipyols(array):
	"""
	Function for rapid OLS with time. the regression is done with 
	an independent variable rangeing from 0 to array.shape to make
	the intercept the start which simplifies calculation
	args:
		array 		np : numpy array of annual max VI over time 
	return
		result 		np : change(total change between start and end)
						 slopem intercept, rsquared, pvalue, std_error
	"""
	# +++++ Get the OLS +++++
	try:
		# if bn.allnan(array):
			# return np.array([np.NAN, np.NAN, np.NAN, np.NAN])

		slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(array.shape[0]), array)
		# +++++ calculate the total change +++++
		# change = (slope*array.shape[0])
		# +++++ return the results +++++
		return np.array([slope, intercept, r_value**2, p_value])
	except Exception as e:
		# print(e)
		# warn.warn("unhandeled Error has occured")
		# ipdb.set_trace()
		return np.array([np.NAN, np.NAN, np.NAN, np.NAN])


#==============================================================================
if __name__ == '__main__':
	main()