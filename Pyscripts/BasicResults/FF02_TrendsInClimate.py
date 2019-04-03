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
from dask.diagnostics import ProgressBar
from numba import jit
import bottleneck as bn
import scipy as sp
import glob
from scipy import stats
import statsmodels.stats.multitest as smsM

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
	# =========== Create the summary of the datasets to be analyised ==========
	data= OrderedDict()


	data["Terrraclim_ppt"] = ({
		'fname':"./data/cli/1.TERRACLIMATE/TerraClimate_stacked_ppt_1977to2017_ppt_yearsum_20window.nc",
		'var':"ppt", "gridres":"terraclimate", "region":"Global", "Periods":["AnnualMax"]
		})
	data["Terrraclim_tmean"] = ({
		'fname':"./data/cli/1.TERRACLIMATE/TerraClimate_stacked_tmean_1977to2017_annualmean_20yearwindow.nc",
		'var':"tmean", "gridres":"terraclimate", "region":"Global", "Periods":["AnnualMax"]
		})

	fname = "./data/cli/1.TERRACLIMATE/TerraClimate_stacked_ppt_1977to2017_ppt_yearsum_20window_trend2.nc"
	Plot_Trend(xr.open_dataset(fname), ["slope"], "Terrraclim_ppt", "ppt", "polyfit", "Terraclimate", "Global")
	fname = "./data/cli/1.TERRACLIMATE/TerraClimate_stacked_tmean_1977to2017_annualmean_20yearwindow_slope.nc"
	Plot_Trend(xr.open_dataset(fname), ["slope"], "Terrraclim_tmean", "tmean", "polyfit", "Terraclimate", "Global")
	for method in ["polyfit", "scipyols", "theilsen"]:
		for dt in data:
			print (dt, method)
			# ========== FIt a theilsen slope estimation ==========
			trendmapper(dt, 
					data[dt]["fname"], data[dt]["var"], method, 
					data[dt]["gridres"], data[dt]["region"])
		


def trendmapper(
	dataset, fname, var, method, gridres, region, fdpath="", force = False, plot=True):
	"""
	Master function for trend finder
	args:
		dataset: str
			the code for the dataset
		fname: str
			filename of the netcdf file to be opened
		var: str
			name of the variable in the netcdf 

	"""
	
	# ========== open the dataset and pull the values
	if type(fname) == str:
		ds       = xr.open_dataset(fname)
		global_attrs = GlobalAttributes(ds, var)
	else:
		if dataset == "MODISaqua":
			fouts = "./data/veg/MODIS/aqua/%s_AnnualMax.nc" % dataset
		# files to be removed 
		cleanup = []
		if not os.path.isfile(fouts):
			for fls in fname:
				# ========== open the file for a given year ==========
				fileouts = fls[:-3]+"_AnnualMax.nc"
				cleanup.append(fileouts)

				if not os.path.isfile(fileouts):
					dsmf = xr.open_dataset(fls, chunks={"latitude":480})
					global_attrs = GlobalAttributes(dsmf, var)

					# ========== get the max value for the year ==========

					# ipdb.set_trace()
					dsp   = dsmf.groupby("time.year").max(dim="time")
					tm = [dt.datetime(int(year) , 6, 30) for year in dsp.year]
					dsp = dsp.rename({"year":"time"})
					dsp["time"] = pd.to_datetime(tm)
					dsp.attrs = global_attrs

					# ipdb.set_trace()

					print("starting write")
					encoding = OrderedDict()
					# ipdb.set_trace()
					# sys.exit()
					encoding[var] = ({'shuffle':True, 
						# 'chunksizes':[1, dsp.latitude.shape[0], dsp.longitude.shape[0]],
						'zlib':True,
						'complevel':6})

					delayed_obj = dsp.to_netcdf(fileouts, 
						format         = 'NETCDF4', 
						encoding       = encoding,
						unlimited_dims = ["time"], 
						compute=False)
					with ProgressBar():
						results = delayed_obj.compute()
					dsp.close()
					dsmf.close()
			# ========== Make a joined file name ==========
			jfname = " ".join(cleanup) 
			# print mergetime
			subp.call(
				"cdo -P 4 -b F64 mergetime %s %s" % (jfname, fouts),
				shell=True
				)
			# Open the saved dataset
			ds = xr.open_dataset(fouts)
			# remove the interum files
			for fles in cleanup:
				os.remove(fles)


		else:ds = xr.open_dataset(fouts)

		# warn.warn(" i need to save it out and reload everything")
		# ipdb.set_trace()

	yr_start = pd.to_datetime(ds.time.min().values).year 
	endyr   = pd.to_datetime(ds.time.max().values).year 
	# ipdb.set_trace()

	# ========== Create the outfile name ==========
	fout = './results/netcdf/%s_%s_%s_%sto%d_%s%s.nc' % (
		dataset,  var, method, yr_start, endyr,region, gridres)

	# ========== Check if the file already exists ==========
	if all([os.path.isfile(fout), not force]):
		warn.warn("Loading existing file, force is needed to overwrite")
		ds_trend = xr.open_dataset(fout)
		kys = [n for n in ds_trend.data_vars]
	else:
		results     = []

		# ========== Create the global attributes ==========
		dst = ds[var]
		if dataset == "GIMMS31v10":
			dst /= 10000.0

		# ========== Calculate the trend ==========
		if (dst.nbytes * 1e-9) <  16:
			trends, kys = _fitvals(dst, method=method)
		else:
			trends, kys = _multifitvals(dst, method=method)
		# Correct for multiple comparisons
		if "pvalue" in kys:
			trends, kys = MultipleComparisons(trends, kys, aplha = 0.10)

		results.append(trends)

		layers, encoding = dsmaker(ds, var, results, kys, method)
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
	
	if plot:
		Plot_Trend(ds_trend, kys, dataset, var, method, gridres, region)


		# get the value
#==============================================================================


def Plot_Trend(ds_trend, kys, dataset, var, method, gridres, region):
	"""
	Function to build global trend maps
	"""
	# ========== Build all the plots ==========
	# +++++ Plot number +++++
	pn = 1

	# ========== create the colormap ==========
	cmap, vmin, vmax = cbvals(var, "slope")
	# plt.figure(1, dpi=600)
	ax = plt.subplot(projection=ccrs.PlateCarree())
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(cpf.COASTLINE, zorder=101)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	# add lat long linse
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
		linewidth=1, color='gray', alpha=0.5, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_right = False

	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER

	ds_trend.slope.plot(
		transform=ccrs.PlateCarree(), ax=ax, 
		cmap=cmap, vmin=vmin, vmax=vmax)
	fig = plt.gcf()
	fig.set_size_inches(41, 20)   
	plt.savefig("./plots/Meeting/%s_%s_%s_slope_%s.png" % (dataset, method, var, region))#, dp1=400)
	# plt.savefig("./plots/Meeting/%s_%s_%s_slope_%s.pdf" % (dataset, method, var, region))#, dp1=400)
	plt.show()
	# plt.show()
	# plt.coloes

	ax = plt.subplot(projection=ccrs.PlateCarree())
	ax.add_feature(cpf.OCEAN, facecolor="w", alpha=1, zorder=100)
	ax.add_feature(cpf.COASTLINE, zorder=101)
	ax.add_feature(cpf.BORDERS, linestyle='--', zorder=102)
	ax.add_feature(cpf.LAKES, alpha=0.5, zorder=103)
	ax.add_feature(cpf.RIVERS, zorder=104)
	# add lat long linse
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
		linewidth=1, color='gray', alpha=0.5, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_right = False

	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER
	try:

		ds_trend.slope.sel({'latitude':slice(56, 49), "longitude":slice(103, 123)}).plot(
			transform=ccrs.PlateCarree(), ax=ax, 
			cmap=cmap, vmin=vmin, vmax=vmax)
	except ValueError:
		ds_trend.slope.sel({'lat':slice(56, 49), "lon":slice(103, 123)}).plot(
			transform=ccrs.PlateCarree(), ax=ax, 
			cmap=cmap, vmin=vmin, vmax=vmax)
	plt.savefig("./plots/Meeting/%s_%s_%s_slope_StudyArea.png" % (dataset, method, var))#, dp1=400)
	plt.savefig("./plots/Meeting/%s_%s_%s_slope_StudyArea.pdf" % (dataset, method, var))#, dp1=400)
	plt.show()

	ipdb.set_trace()

def cbvals(var, ky):

	"""Function to store all the colorbar infomation i need """
	cmap = None
	vmin = None
	vmax = None
	if ky == "slope":
		if var == "tmean":
			vmax =  0.07
			vmin = -0.07
			cmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors)
		elif var =="ppt":
			vmin = -5.0
			vmax =  5.0
			cmap = mpc.ListedColormap(palettable.cmocean.diverging.Curl_20_r.mpl_colors)
	elif ky == "pvalue":
		cmap = mpc.ListedColormap(palettable.matplotlib.Inferno_20.hex_colors)
		vmin = 0.0
		vmax = 1.0
	elif ky == "rsquared":
		cmap = mpc.ListedColormap(palettable.matplotlib.Viridis_20.hex_colors)
		vmin = 0.0
		vmax = 1.0
		# cmap =  
	elif ky == "intercept":
		cmap = mpc.ListedColormap(palettable.cmocean.sequential.Ice_20_r.mpl_colors)
		if var == "tmean":
			# vmax =  0.07
			# vmin = -0.07
			# cmap = mpc.ListedColormap(palettable.cmocean.diverging.Balance_20.mpl_colors)
			# ipdb.set_trace()
			pass
		elif var =="ppt":
			vmin = 0
			vmax = 1000
			# cmap = mpc.ListedColormap(palettable.cmocean.diverging.Curl_20_r.mpl_colors)

	return cmap, vmin, vmax


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

def MultipleComparisons(trends, kys, aplha = 0.10, MCmethod="fdr_by"):
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
	pvalue_adj =  smsM.multipletests(pvalue1d, method=MCmethod, alpha=0.10)
	
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


def _fitvals(dvt, method="polyfit"):
	"""
	Takes the ds[var] and performs some form of regression on it
	args 
		dvt: xarray data array
			the values to be regressed
		method: str
			the regression approach to take
	"""
	# ========== Get the values ==========
	vals  = dvt.values 
	# except MemoryError:
	# 	ipdb.set_trace()


	# ========== Convert the time into years ==========
	try:
		years = pd.to_datetime(dvt.time.values).year
		t0 = pd.Timestamp.now()
		print("testing with %s from %d to %d starting at: %s" % (
			method, pd.to_datetime(dvt.time.values).year.min(), 
			pd.to_datetime(dvt.time.values).year.max(), str(t0)))

	except AttributeError:
		years = pd.to_datetime(dvt.year.values).year
		t0 = pd.Timestamp.now()
		print("testing with %s from %d to %d starting at: %s" % (
			method, pd.to_datetime(dvt.year.values).year.min(), 
			pd.to_datetime(dvt.year.values).year.max(), str(t0)))
	
	# ========== Reshape the datainto two dims  ==========
	vals2 = vals.reshape(len(years), -1)


	# ========== pass the results to the specific regression function  ==========
	if method=="polyfit":
		# Do a first-degree polyfit
		vals2[np.isnan(vals2)] = 0
		regressions = np.polyfit(years, vals2, 1)
		ipdb.set_trace()
		regressions[regressions== 0] = np.NAN
		trends = [regressions[0,:].reshape(vals.shape[1], vals.shape[2])]
		kys = ["slope"]
		ipdb.set_trace()

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

	# ========== build a holder ==========
	vals = np.zeros((4, array2.shape[1]))

	# ========== get the starting time ==========
	t0 = pd.Timestamp.now()

	for line in range(0, array2.shape[1]):
		if (line % lineflick == 0):
			string = ("\rRegression climate: line: %d of %d" % 
						(line, array2.shape[1]))
			if line > 0:
				# TIME PER LINEFLICK
				lfx = (pd.Timestamp.now()-t0)/line
				lft = str((lfx*lineflick))
				trm = str(((array2.shape[1]-line)*(lfx)))

				string += (" t/%d lines: %s. ~eta: %s" % (
					lineflick,lft, trm) )

			sys.stdout.write(string)
			sys.stdout.flush()

		out = myfunc(array2[:, line])		
		# vals.append(out)
		vals[:, line] = out
	res[:, ana] = vals
	return res

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



if __name__ == '__main__':
	main()