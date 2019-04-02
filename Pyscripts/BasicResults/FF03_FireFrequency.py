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
from scipy import stats
import xarray as xr
from numba import jit
# import bottleneck as bn
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
# import cartopy.crs as ccrs
# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main():
	
	# ========== Set up the params ==========
	arraysize = 10000 # size of the area to test
	mat       = 50.0  # how long before a forest reaches maturity
	germ      = 20.0  # how long before a burnt site can germinate
	burnfrac  = 0.050  # how much burns
	firefreq  = 5    # how often the fires happen
	years     = 100   # number of years to loop over


	# ========== Make an array ==========
	array = np.zeros(arraysize)

	# ========== Make the entire array mature forest ==========
	array[:] = mat

	# ========== Create empty lists to hold the variables ==========
	ymean = []
	fmat  = []
	fgerm = []
	# ========== start the loop ==========
	for year in range(0, years):
		# Loop over every year in case i want to add major fire events
		# print(year)
		if year % firefreq == 0: 
			# FIre year
			array = firetime(array, mat, germ, burnfrac)
		else:
			# non fire year
			array = firetime(array, mat, germ, 0.0)
		# Mean years
		ymean.append(np.mean(array))
		# Fraction of mature forest
		fmat.append(np.sum(array==mat)/float(arraysize))
		# Fraction of germinating forest
		fgerm.append(np.sum(array>germ)/float(arraysize))
	obs = OrderedDict()
	obs["MeanAge"]             = ymean
	obs["MatureFraction"]      = fmat
	obs["GerminatingFraction"] = fgerm
	df = pd.DataFrame(obs)
	# df.MeanAge.plot()
	# df.MatureFraction.plot()
	df.plot(subplots=True)
	plt.show()

	# ipdb.set_trace()

#==============================================================================

def firetime(array, mat, germ, burnfrac):
	"""
	takes in an array and modifies it based on the inputs
	args:
		array: np array	
			array of the years of the forest
		mat: float
			years to reach complete maturity 
		germ: float
			years to reach germination point 
		burnfrac: float
			the percentage of the data that is burnt each year
	"""
	# ========== Find the mature part of the array ==========
	mature = array == mat
	# ipdb.set_trace()
	# ========== Add one year to non mature forests ========== 
	array[~ mature] += 1.0

	# ========== Burn a fraction of the forest ==========
	bf = array.shape[0]*burnfrac
	array[np.logical_and(mature, (np.cumsum(mature)<bf))] = 0.0

	# ========== Return the array ==========
	return array

#==============================================================================

if __name__ == '__main__':
	main()