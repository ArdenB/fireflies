
"""
Prelim script for looking at netcdf files and producing some trends
Broken into three parts
	Part 1 looks at vegetation trends
	Part 2 looks at climate trends
	Part 3 interigates the field data
"""
#==============================================================================

__title__ = "Vegetation and climate trends"
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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Import debugging packages 
import ipdb
# +++++ Import my packages +++++
# import MyModules.CoreFunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================
def main(args):
	# ========== Get the key infomation from the args ==========
	fdpath = args.fdpath 
	warn.warn(
		'''
		This is currently only in alpha testing form
		I will replace all the variables and infomation 
		for experiments in a dataframe so i can look at
		different aproaches
		''')
	
	# ========== Pull out info needed from the field data ==========
	for dens in ["sDens2017Ls", "sDens2017modis"]:
		RFinfo = Field_data(fdpath, den=dens)
		# RFinfo = Field_data(fdpath, den="fracThresh2017Ls")

		# ========== Compare the overall site infomation ==========
		print("Using density data: %s" % dens)
		for DS in ["NDVI", "LAI"]:
			r2, tau = VI_trend(RFinfo, DS,den=dens, plot=False)
			r2, tau = VI_trend(RFinfo, DS,den=dens, fireyear=True)

	ipdb.set_trace()

#==============================================================================
def VI_trend(RFinfo,var, den, fireyear=False, plot=True):
	"""
	This is a function for looking for any correspondense between 
	sites and observed vi trends

	Note:
		This is a niave approach. I'm ignoring key infomation about the
		timing of the fire, the intensity of the fire etc etc. This is more
		of a proof of concept 
	"""

	# warn.warn(
	# 	'''
	# 	This is currently only in alpha testing form
	# 	i'm going to using a simple trend test without
	# 	any consideration of significance. i used cdo
	# 	regres on copernicious NDVI data to start. 
	# 	''')
	
	# warn.warn(
	# 	'''
	# 	This approach is currently ambivilant to when fire 
	# 	events occured. This may need to be incoperated with 
	# 	some form of breakpoint detection (Chow test or MVregression)
	# 	''')

	# ========== Load in the trend data using xarray ==========
	if fireyear:
		# COnsidering the fireyear
		ncin = "./data/veg/COPERN/%s_anmax_Russia.nc" % var
	else:
		# Only looking at years since the fire
		ncin = "./data/veg/COPERN/%s_anmax_Russia_cdoregres.nc" % var
	ds   = xr.open_dataset(ncin)


	# ========== Find the recuitment failure in the netcdf ==========
	if fireyear:
		# warn.warn('''
		# 	Theilsen implemented quickly. check in the future 
		# 	''')
		# RFshort = RFinfo[RFinfo.fireyear<=2012] 
		testnm = "Postfire Theilsen slope"
		VItrend = []
		for index, row in RFinfo.iterrows():
			if row.fireyear<=2012:
				sty   = '%d-01-01' % (int(row.fireyear))
				array = ds[var].sel(
					{"lat":row.lat, "lon":row.lon},
					method="nearest").sel(time=slice(sty, '2017-12-31'))
				VItrend.append(scipyTheilSen(array))
			else:
				VItrend.append(np.NAN)

	else:
		testnm = "1999-2017 OLS slope"
		VItrend = [float(ds[var].sel(
			{"lat":row.lat, "lon":row.lon}, method="nearest").values) for index, row in RFinfo.iterrows()]

	RFinfo["VItrend"] = VItrend
	RFinfo.dropna(inplace=True, subset=['sn', 'lat', 'lon', den, 'RF17', 'VItrend'])

	
	slope, intercept, r_value, p_value, std_err = stats.linregress(x=RFinfo[den], y=RFinfo.VItrend)
	# r2val = r_val**2
	tau, p_value = stats.kendalltau(x=RFinfo[den], y=RFinfo.VItrend)
	print(var, den, testnm)
	print("r-squared:", r_value**2)
	print("kendalltau:", tau)

	# make a quick plot

	if plot:
		# ========== Map of the regional trend data ==========
		# build the figure and grid
		if not fireyear:

			plt.figure(1, figsize=(8, 4))
			ax = plt.axes(projection=ccrs.PlateCarree())
			# add lat long linse
			gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				linewidth=1, color='gray', alpha=0.5, linestyle='--')
			gl.xlabels_top = False
			gl.ylabels_right = False
			gl.xformatter = LONGITUDE_FORMATTER
			gl.yformatter = LATITUDE_FORMATTER
			# define the colorbar
			cmap = mpc.ListedColormap(palettable.colorbrewer.diverging.PRGn_6.mpl_colors)
			# gl.xlabel_style = {'size': 15, 'color': 'gray'}
			# gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
			ds[var].plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap)  

		# ========== Scatter plot of the trend vs field data ==========
		sns.lmplot( x=den, y="VItrend", 
			data=RFinfo, fit_reg=False, 
			hue='RF17', height=4, aspect=2)
		plt.show()

	return r_value**2, tau


def Field_data(fdpath, den="sDens2017Ls"):
	"""
	# Aim of this function is to look at the field data a bit

	To start it just opens the file and returns the lats and longs 
	i can then use these to look up netcdf fils
	"""
	# ========== Load in the relevant data ==========
	fsum = pd.read_csv("./data/field/RF_catsum.csv")
	fsum.sort_values(by=["sn"],inplace=True) 
	fcut = fsum[fsum.sn<64]
	fd18 = pd.read_csv(fdpath)
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
	return RFinfo

#==============================================================================

# @jit
def scipyTheilSen(array):
	"""
	Function for rapid TehilSen slop estimation with time. 
	the regression is done with  an independent variable 
	rangeing from 0 to array.shape to make the intercept 
	the start which simplifies calculation
	
	args:
		array 		np : numpy array of annual max VI over time 
	return
		result 		np : slope, intercept
	"""
	try:
		slope, intercept, _, _ = stats.mstats.theilslopes(array)
		return slope #, intercept
	except Exception as e:
	 	print(e) 
	 	ipdb.set_trace()



#==============================================================================
if __name__ == '__main__':
	# ========== Set the args Description ==========
	description='Passed argumants'
	parser = argparse.ArgumentParser(description=description)
	
	# ========== Add additional arguments ==========
	parser.add_argument(
		'--fdpath', type=str, action="store", 
		default="./data/field/2018data/siteDescriptions18.csv", 
		help='The path to the field results')
	parser.add_argument(
		'--path2', type=str, default=None, 
		help='The path to the second runs results')
	# parser.add_argument(
	# 	"--gparts", type=int, default=None,   
	# 	help="the max original partnumber")
	args = parser.parse_args() 
	
	# ========== Call the main function ==========
	main(args)


