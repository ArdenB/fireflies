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
	# ========== Make a list of files ==========
	files = glob.glob("./data/field/exportedNDVI/NDVI_2018sites_*MonthlyMax.csv")

	# ========== Dict to hold the NDVI ==========
	NDVI = OrderedDict()
	site = 4

	# ========== Make a list of files ==========
	for fn in files:
		# ========== Get the dataset name ==========
		dataset = fn.split("_")[2] +"_" +fn.split("_")[-2]
		# ipdb.set_trace()

		# ========== Read the data ==========
		df       = pd.read_csv(fn, index_col="sn", parse_dates=True)
		df[df<0] = np.NAN

		# ========== Get the site values ==========
		sv = df.loc[site]#[~df.loc[site].isnull()] 
		sv.index = pd.to_datetime(sv.index)

		# ========== Add them to the NDVI dict ==========
		NDVI[dataset] = sv
	
	ax = plt.subplot(xlim=(pd.to_datetime("1981-01-01"), pd.to_datetime("2019-04-04")))
	for ds in NDVI:	
		NDVI[ds].plot(sharex=True, sharey=True, ax=ax, label=ds)
	plt.legend()
	# plt.savefig()
	plt.show()
	ipdb.set_trace()


if __name__ == '__main__':
	main()