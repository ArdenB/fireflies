
"""
Prelim script for looking at netcdf files and producing some trends
Broken into three parts
	Part 1 pull out the NDVI from the relevant sites
"""
#==============================================================================

__title__ = "Time Series Chow Test"
__author__ = "Arden Burrell"
__version__ = "v1.0(27.02.2019)"
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Import plotting and colorpackages
from statsmodels.sandbox.regression.predstd import wls_prediction_std
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
# +++++ Import my packages +++++
import myfunctions.stats as sf
# import MyModules.CoreFunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

#==============================================================================
def main():
	# ========== Get the key infomation from the args ==========
	# fdpath = args.fdpath 
	warn.warn(
		'''
		This is currently only in alpha testing form
		I will replace all the variables and infomation 
		for experiments in a dataframe so i can look at
		different aproaches
		''')

	# ========== set the filnames ==========
	data= OrderedDict()
	data["MODISaqua"] = ({
		"fname":"./data/veg/MODIS/aqua/processed/MYD13Q1_A*_final.nc",
		'var':"ndvi", "gridres":"250m", "region":"SIBERIA", "timestep":"16day", 
		"start":2002, "end":2018
		})
	data["COPERN"] = ({
		'fname':"./data/veg/COPERN/NDVI_MonthlyMax_1999to2018_at_1kmRUSSIA.nc",
		'var':"NDVI", "gridres":"1km", "region":"RUSSIA", "timestep":"Monthly",
		"start":1999, "end":2018
		})
	data["GIMMS"] = ({
		"fname":"./data/veg/GIMMS31g/GIMMS31v1/timecorrected/ndvi3g_geo_v1_1_1981to2017_mergetime_compressed.nc",
		'var':"ndvi", "gridres":"8km", "region":"Global", "timestep":"16day", 
		"start":1981, "end":2017
		})

	win = 3.0 # Window from the start and end of the time series

	for syear in [2017, 2018]:
		# ========== Pull out info needed from the field data ==========
		SiteInfo = Field_data(year = syear)
		Firedict = OrderedDict()
		Chowdict = OrderedDict()
		# ========== Loop over each of the included vegetation datasets ==========
		for dsn in data:
			# ========== Get the veg values ==========
			infile = ("./data/field/exportedNDVI/NDVI_%dsites_%s_%dto%d_%s_"
				% (syear, dsn,data[dsn]["start"], data[dsn]["end"], data[dsn]["gridres"]))
			df = pd.read_csv(infile+"AnnualMax.csv", index_col="sn")

			FireBreak = []

			# ========== loop over the value ==========
			for index, row in SiteInfo.iterrows():
				# ===== Check if the fire year is in the range ======
				if np.isnan(row["fireyear"]):
					# fire year is problematic from data
					FireBreak.append([np.NAN, np.NAN, np.NAN, np.NAN, np.NAN])
				elif row["fireyear"] < (data[dsn]["start"]+win):
					# too close to the start
					FireBreak.append([np.NAN, np.NAN, np.NAN, np.NAN, np.NAN])
				elif row["fireyear"] > (data[dsn]["end"]-win):
					# too close to the end
					FireBreak.append([np.NAN, np.NAN, np.NAN, np.NAN, np.NAN])
				else:
					firedate = dt.datetime(int(row["fireyear"]) , 1, 1)
					vi    = df.loc[index].values
					warn.warn("Use a for loop to implement multiple breapoints in the future")
					time = pd.to_datetime(df.loc[index].index) #independant variable
					# Create the dummy variable
					dummy = np.zeros(vi.shape)    
					dummy[ time > pd.to_datetime(firedate)] = 1

					dfs = pd.DataFrame({
						"loc":np.arange(vi.shape[0]),
						"ndvi":vi, 
						"dummy":dummy}, index = time)

					# X = dfs[[loc]]
					res    = smf.ols(formula='ndvi ~ loc', data=dfs).fit() 
					res2   = smf.ols(formula='ndvi ~ loc + dummy', data=dfs).fit() 
					chow   = sf.ChowTest(df.loc[index], firedate)
					ana    = pd.DataFrame(sm.stats.anova_lm(res, res2))
					anasig = ana.iloc[1,-1]
					# if anasig < 0.05:
					# 	dfs.ndvi.plot()
					# 	plt.show()
					# 	ipdb.set_trace()

					# ========== build the output ==========
					results = [chow["F_value"], chow["p_value"], res.f_pvalue, res2.f_pvalue, anasig]
					FireBreak.append(results)
			dfx = pd.DataFrame(
				np.vstack(FireBreak), 
				columns=["ChowF", "ChowP", "lmTot_p", "lmFire_p", "Anova_p"], 
				index=SiteInfo.index) 
			dfx["RF"] = SiteInfo.RF 
			Firedict[dsn] = dfx
			Chowdict[dsn] = dfx["ChowP"]
		dfchow       = pd.DataFrame(Chowdict) 
		dfchow["RF"] = SiteInfo.RF 
		ipdb.set_trace()
	
	ipdb.set_trace()

#==============================================================================
def regplot(res, dfs):
	prstd, iv_l, iv_u = wls_prediction_std(res)

	fig, ax = plt.subplots(figsize=(8,6))

	ax.plot(dfs["ndvi"], 'o', label="NDVI")
	# ax.plot(x, y_true, 'b-', label="True")
	ax.plot(res.fittedvalues, 'r--.', label="Predicted")
	ax.plot(iv_u, 'r--')
	ax.plot(iv_l, 'r--')
	legend = ax.legend(loc="best")

	plt.show()


def Field_data(year = 2018):
	"""
	# Aim of this function is to look at the field data a bit

	To start it just opens the file and returns the lats and longs 
	i can then use these to look up netcdf fils
	"""
	# ========== Load in the relevant data ==========
	if year == 2018:
		fd18 = pd.read_csv("./data/field/2018data/siteDescriptions18.csv")
	else:
		fd18 = pd.read_csv("./data/field/2018data/siteDescriptions17.csv")

	fd18.sort_values(by=["site number"],inplace=True) 
	# ========== Create and Ordered Dict for important info ==========
	info = OrderedDict()
	info["sn"]  = fd18["site number"]
	try:
		info["lat"] = fd18.lat
		info["lon"] = fd18.lon
		info["RF"]  = fd18.rcrtmnt
	except AttributeError:
		info["lat"] = fd18.strtY
		info["lon"] = fd18.strtX
		info["RF"]  = fd18.recruitment
	
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
				year = float(str(val[0]).split(" and ")[0])
				if year < 1980:
					warn.warn("wrong year is being returned")
					year = float(str(val).split(" ")[0])
					# ipdb.set_trace()

				return year
			except Exception as e:
				# ipdb.set_trace()
				# print(e)
				print(val)
				return np.NAN

	# info[den] = [_missingvalfix(
	# 	fcut[fcut.sn == sn][den].values) for sn in info['sn']]

	# info["RF17"] = [_missingvalfix(
	# 	fcut[fcut.sn == sn]["RF2017"].values) for sn in info['sn']]
	
		
	info["fireyear"] = [_fireyear(fyv) for fyv in fd18["estimated fire year"].values]
	# ========== Convert to dataframe and replace codes ==========
	RFinfo = pd.DataFrame(info).set_index("sn")
	# ipdb.set_trace()
	# RFinfo["RF17"].replace(0.0, "AR", inplace=True)
	# RFinfo["RF17"].replace(1.0, "RF", inplace=True)
	# RFinfo["RF17"].replace(2.0, "IR", inplace=True)
	# RFinfo["YearsPostFire"] = 2017.0 - RFinfo.fireyear
	return RFinfo

#==============================================================================

if __name__ == '__main__':
	main()