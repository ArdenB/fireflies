
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
import xarray as xr
# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 
import seaborn as sns
from scipy import stats
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
	
	# ========== Pull out info needed from the field data ==========
	dens = "sDens2017modis"
	RFinfo = Field_data(fdpath, den=dens)
	# RFinfo = Field_data(fdpath, den="fracThresh2017Ls")

	# ========== Compare the overall site infomation ==========
	warn.warn(
		'''
		This is currently only in alpha testing form
		I will replace all the variables and infomation 
		for experiments in a dataframe so i can look at
		different aproaches
		''')
	r2, tau = VI_trend(RFinfo, "NDVI",den=dens, plot=True)
	r2, tau = VI_trend(RFinfo, "LAI", den=dens, plot=True)


	ipdb.set_trace()

#==============================================================================
def VI_trend(RFinfo,var, den, plot=True):
	"""
	This is a function for looking for any correspondense between 
	sites and observed vi trends

	Note:
		This is a niave approach. I'm ignoring key infomation about the
		timing of the fire, the intensity of the fire etc etc. This is more
		of a proof of concept 
	"""

	warn.warn(
		'''
		This is currently only in alpha testing form
		i'm going to using a simple trend test without
		any consideration of significance. i used cdo
		regres on copernicious NDVI data to start. 
		''')
	
	warn.warn(
		'''
		This approach is currently ambivilant to when fire 
		events occured. This may need to be incoperated with 
		some form of breakpoint detection (Chow test or MVregression)
		''')

	# ========== Load in the trend data using xarray ==========
	ncin = "./data/veg/COPERN/%s_anmax_Russia_cdoregres.nc" % var
	ds   = xr.open_dataset(ncin)


	# ========== Find the recuitment failure in the netcdf ==========
	NDVItrend = [float(ds[var].sel(
		{"lat":row.lat, "lon":row.lon}, method="nearest").values) for index, row in RFinfo.iterrows()]

	RFinfo["VItrend"] = NDVItrend
	RFinfo.dropna(inplace=True)

	
	slope, intercept, r_value, p_value, std_err = stats.linregress(x=RFinfo[den], y=RFinfo.VItrend)
	# r2val = r_val**2
	tau, p_value = stats.kendalltau(x=RFinfo[den], y=RFinfo.VItrend)
	print("r-squared:", r_value**2)
	print("kendalltau:", tau)

	# make a quick plot

	if plot:
		# plot regional trend data
		plt.figure(1)
		ds[var].plot()  

		sns.lmplot( x=den, y="VItrend", data=RFinfo, fit_reg=False, hue='RF17')
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

	info[den] = [_missingvalfix(
		fcut[fcut.sn == sn][den].values) for sn in info['sn']]

	info["RF17"] = [_missingvalfix(
		fcut[fcut.sn == sn]["RF2017"].values) for sn in info['sn']]
	
	# ========== Convert to dataframe and replace codes ==========
	RFinfo = pd.DataFrame(info)
	RFinfo["RF17"].replace(0.0, "AR", inplace=True)
	RFinfo["RF17"].replace(1.0, "RF", inplace=True)
	RFinfo["RF17"].replace(2.0, "IR", inplace=True)
	return RFinfo

#==============================================================================


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


