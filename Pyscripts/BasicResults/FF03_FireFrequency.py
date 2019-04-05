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
	
	# ========== Set up the params ==========
	arraysize   = 10000   # size of the area to test
	mat         = 40.0    # how long before a forest reaches maturity
	germ        = 10.0    # how long before a burnt site can germinate
	# burnfrac  = 0.10    # how much burns
	burnfrac    = BurntAreaFraction(year=2016)/2
	
	nburnfrac   = 0.0     # how much burns in other years
	# nburnfrac   = BurntAreaFraction(year=2018)/2.0     # how much burns in other years
	# nburnfrac = np.mean([BurntAreaFraction(year=int(yr)) for yr in [2015, 2017, 2018]])     # how much burns in other years

	firefreqL   = [25, 20, 15, 11, 5, 4, 1]       # how often the fires happen
	years       = 200     # number of years to loop over
	RFfrac      = 0.00 # The fraction that will fail to recuit after a fire

	# ========== Create empty lists to hold the variables ==========
	obsMA = OrderedDict() 
	obsMF = OrderedDict() 
	obsGF = OrderedDict() 

	# ========== Loop over the fire frequency list ==========
	for firefreq in firefreqL:
		
		print("Testing with a %d year fire frequency" % firefreq)
		
		# ========== Make an array ==========
		array   = np.zeros(arraysize)
		rucfail = np.ones( arraysize)

		# ========== Make the entire array mature forest ==========
		array[:] = mat

		# ========== Create the empty arrays ==========
		ymean  = []
		fmat   = []
		fgerm  = []
		rfhold = 0 #the left over fraction of RF
		# ========== start the loop ==========
		for year in range(0, years):
			# Loop over every year in case i want to add major fire events
			# print(year)
			if year % firefreq == 0: 
				# FIre year
				array, rucfail, rfhold = firetime(array, mat, germ, burnfrac, rucfail, RFfrac, rfhold)
			else:
				# non fire year
				array, rucfail, rfhold = firetime(array, mat, germ, nburnfrac, rucfail, RFfrac, rfhold)
			# Mean years
			ymean.append(np.mean(array))
			# Fraction of mature forest\
			fmat.append(np.sum(array>=mat)/float(arraysize))

			# Fraction of germinating forest
			fgerm.append(np.sum(array>germ)/float(arraysize))
			# if year>60 and firefreq == 1:
			# 	ipdb.set_trace()

		obsMA["FF_%dyr" % firefreq] = ymean
		obsMF["FF_%dyr" % firefreq] = fmat
		obsGF["FF_%dyr" % firefreq] = fgerm
	
	obs = OrderedDict()
	obs["MeanAge"]             = pd.DataFrame(obsMA)
	obs["MatureFraction"]      = pd.DataFrame(obsMF)
	obs["GerminatingFraction"] = pd.DataFrame(obsGF)
	for kys in obs.keys(): 
		print(kys)   
		# plt.figure(kys)
		obs[kys].plot(title=kys)
	
	plt.show()

	# ipdb.set_trace()

#==============================================================================

def firetime(array, mat_f, germ, burnfrac, rucfail, RFfrac, rfhold):
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
		rucfail: np array
			Lay to hold info about rf. 1=Abundant, 2 = RF
		RFfrac: float
			Fraction of burnt are that fails to recuit
	"""
	# ========== Calculate the fraction to burn ==========
	bf = array.shape[0]*burnfrac
	
	# ========== Find the mature part of the array ==========
	# mature = array == mat
	mat = np.max(array)
	mature = array >= mat

	while np.sum(mature) <= bf:
		mat -=  1
		if mat == 0:
			warn.warn("Have hit RF mat 0, going interactive")
			mature = array > 0
			break
		mature = array >= mat

	# ========== Add one year to non mature forests ========== 
	# array[~ mature] += 1.0
	array += 1.0

	# ========== Burn a fraction of the forest ==========
	burnloc = np.logical_and(mature, (np.cumsum(mature)<=bf))
	array[burnloc] = 0.0

	# ========== Add a recuitment failure adjustment ==========
	rfloc  =  (bf * RFfrac) + rfhold # number of place that fail to recuit
	if not (rfloc % 1) == 0:
		rfhold = np.round(rfloc % 1, decimals=5)
	else:
		rfhold = 0
	prevrf = np.shape(array)[0] - np.sum(rucfail) #areas that have already failed to recuit

	# ========== include the RF ==========
	rucfail[np.logical_and(burnloc, ((np.cumsum(burnloc)<= (prevrf +rfloc))))] = 0
	array *= rucfail

	# ========== Return the array ==========
	return array, rucfail, rfhold

#==============================================================================

def BurntAreaFraction(year = 2015):
	# ========== Path to the dataset ==========
	path = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/M0044633/"
	glb  = "*_%d*.nc" % year

	# ========== Get a list of the files ==========
	files = glob.glob(path+glb)  
	if files == []:
		warrn.warn("selected year has no burnt area to look at")
		raise ValueError

	# ========== Get a list of the files ==========
	parts = []
	time  = []
	for fn in files:
		dsp = xr.open_dataset(fn)
		da = dsp.BA_DEKAD.sel({'lat':slice(55.5, 51.5), "lon":slice(107, 114)})
		da = da.where(da <=1.0)
		try:
			tm = [dt.datetime(int( fn[-35:-31]), int(fn[-31:-29] ), int(fn[-29:-27]))]
		except Exception as e:
			ipdb.set_trace()
			raise e
		time.append(pd.to_datetime(tm))
		BAvalues = da.values
		
		if not BAvalues.shape[0] == 1:
			BAvalues = np.expand_dims(BAvalues, axis=0)

		parts.append(BAvalues)
	# ax = plt.subplot(projection=ccrs.PlateCarree())
	# da.plot(ax=ax, transform=ccrs.PlateCarree())
	# plt.show()
	try:
		burnt = np.sum(np.stack(parts, axis=0), axis=0)
	except Exception as e:
		ipdb.set_trace()
		raise e
	BAF = bn.nansum(burnt)/np.sum(~np.isnan(burnt)).astype(float)
	print("the fraction of burnt area for %d is %f" % (year, BAF))
	# ipdb.set_trace()
	# plt.imshow(burnt[0, :,:]);plt.show()
	return BAF


if __name__ == '__main__':
	main()