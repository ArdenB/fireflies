"""
Script goal, 

Make the climate maps

"""
#==============================================================================

__title__ = "Climate stat calculator"
__author__ = "Arden Burrell"
__version__ = "v1.0(11.11.2020)"
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
# import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob
import shutil
import time
import subprocess as subp
from dask.diagnostics import ProgressBar

from collections import OrderedDict
from cdo import *

# from scipy import stats
# from numba import jit


# from netCDF4 import Dataset, num2date, date2num 
# from scipy import stats
# import statsmodels.stats.multitest as smsM

# Import plotting and colorpackages
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl

import palettable 

# import seaborn as sns
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string
from statsmodels.stats.weightstats import DescrStatsW


# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 

# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# import pdb as ipdb
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================
def main():
	# ========== find the paths ==========
	dpath, cpath = syspath()
	cmaps = ([palettable.colorbrewer.diverging.RdBu_11.mpl_colormap, palettable.colorbrewer.diverging.PuOr_11_r.mpl_colormap])
	# ========== find the climate path ==========
	for va, vrng, cmap in zip(["ppt", "tmean"], [[-10.0, 10.0],[-0.10, 0.10]], cmaps):
		for gp in ["annual", "DJF", "MAM", "JJA", "SON"]:
			plt.figure(f"{gp} {va}")
			ds = xr.open_dataset(f"{cpath}TerraClim_{va}_{gp}trend_1985to2015.nc")
			ds.slope.plot(cmap=cmap, vmin=vrng[0], vmax=vrng[1])
			plt.show()
		breakpoint()
			# for sn in []

	breakpoint()


#==============================================================================

def syspath():
	# ========== Create the system specific paths ==========
	sysname   = os.uname()[1]
	backpath = None
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h"
		dpath = "/mnt/d/Data51"
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive"
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
		dpath = "/media/ubuntu/Harbinger/Data51"
	# elif 'ccrc.unsw.edu.au' in sysname:
	# 	dpath = "/srv/ccrc/data51/z3466821"
	elif sysname == 'DESKTOP-T77KK56':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
		backpath = "/mnt/f/fireflies"
		chunksize = 500
	elif sysname == 'DESKTOP-KMJEPJ8':
		dpath = "./data"
		backpath = "/mnt/g/fireflies"
		chunksize = 500
	elif sysname == 'arden-worstation':
		# WHRC linux distro
		dpath = "./data"
		cpath= "/media/arden/SeagateMassStorage/Data51/Climate/TerraClimate/"
		# dpath= "/media/arden/Harbinger/Data51/BurntArea"
	elif sysname == 'LAPTOP-8C4IGM68':
		dpath     = "./data"
		backpath = "/mnt/d/fireflies"
	else:
		ipdb.set_trace()
	return dpath, cpath

# ==============================================================================

if __name__ == '__main__':
	main()