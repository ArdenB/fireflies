"""
Script goal, 

Build evaluation maps of GEE data

"""
#==============================================================================

__title__ = "GEE Movie Fixer"
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
import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob

from collections import OrderedDict
from scipy import stats
from numba import jit

# Import the Earth Engine Python Package
# import ee
# import ee.mapclient
# from ee import batch

# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import palettable 

# import fiona
# fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
# fiona.drvsupport.supported_drivers['KML'] = 'rw' # enable KML support which is disabled by default


# import moviepy.editor as mpe
# import skvideo.io     as skv
# import skimage as ski
# from moviepy.video.io.bindings import mplfig_to_npimage


# import seaborn as sns
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket

# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
import ipdb

# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
# import MyModules.PlotFunctions as pf
# import MyModules.NetCDFFunctions as ncf

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

# ==============================================================================
def main():
	# ========== Read in the Site data ==========
	df = pd.read_csv("./results/ProjectSentinal/Fire sites and forest loss.csv") 
	df = renamer(df)
	# ========== Read in the Site data ==========
	cordname    = "./data/other/GEE_sitelist.csv"
	site_coords = pd.read_csv(cordname, index_col=0)

	# ========== CHeck the data makes sense ==========
	dfchecker(df, site_coords)


	# ========== Get out some basic infomation ==========

	# ========== Get some site specific infomation ==========
	for site in site_coords.name.values:
		out = disturbanceBuilder(site, df, site_coords)
		if not out is None:
			ipdb.set_trace()			

	ipdb.set_trace()

# ==============================================================================
# Data interrogation
# ==============================================================================
def disturbanceBuilder(site, df, site_coords):
	"""Function to i1nterogate the disturbances"""
	# ========== subset the dataset so only site data is present ==========
	dfs = df[df.site == site]

	# ========== Check the number of obs ==========
	print(site, dfs.shape[0])
	Info = OrderedDict()
	# Info["site"] = site
	if dfs.shape[0] == 0:
		return None
	elif dfs.shape[0] > 1:
		# ========== Loop over the user obs ==========
		for index, row in dfs:
			ipdb.set_trace()	
	else:
		row = dfs.iloc[0]
		Info["TotalDisturbance"] = 
		print
	ipdb.set_trace()
# ==============================================================================
# Raw Data correction and processing 
# ==============================================================================
def renamer(df):
	"""Function to rename my dataframe columns """

	rename = OrderedDict()
	rename['What is the name of the site?']         = "site"
	rename['Is the site clearly evergreen forest?'] = "SiteCon"
	rename["What fraction is Evergreen forest?"]    = "ConFracBase"
	rename["Has the fraction of forest changed significantly before the first disturbance event?"] = "ConFracDis"

	distcount = 1
	# ========== rename the disturbance columns ==========
	for col in df.columns.values:
		if col == "Would you like to log any disturbance events?" or col.startswith("Log another disturbance"):
			rename[col] = "Dist%02d" % distcount
			distcount  += 1 #add to the disturbance counter
			df[col]     = df[col].map({'Yes': 1,'yes': 1,'No': np.NAN, 'no': np.NAN})


	# rename[""]
	# rename[""]
	return df.rename(columns=rename)

def dfchecker(df, site_coords):
	
	"""Function to check and sort the data, fixing any errors along the way 
		args:
			df: 	pd df csv from google forms """
	# ========== Check site names make sense ==========
	for sitename in df.site.values:
		if not sitename in site_coords.name.values:
			warn.warn("This site name is a problems: %s" % sitename)
			ipdb.set_trace()

	# ========== Fix the percentages ==========
	df.replace("<25%", "<30%")

# ==============================================================================
if __name__ == '__main__':
	main()