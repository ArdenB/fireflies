
"""
Script goal, 

Make the graphs

"""
#==============================================================================

__title__ = "BA grapher"
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
import dask

from collections import OrderedDict
# from cdo import *

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
import seaborn as sns

# import seaborn as sns
import cartopy as ct
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
import matplotlib.colors as mpc
import matplotlib.patheffects as pe
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import string
from statsmodels.stats.weightstats import DescrStatsW
import pickle
from sklearn import metrics as sklMet

# ========== Import ml packages ==========
import sklearn as skl
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import permutation_importance
from sklearn import metrics as sklMet

# ========== Import my dunctions ==========
import myfunctions.corefunctions as cf
import myfunctions.PlotFunctions as pf 

import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
# import pdb as ipdb
import ipdb

print("numpy version   : ", np.__version__)
print("pandas version  : ", pd.__version__)
print("xarray version  : ", xr.__version__)
print("cartopy version : ", ct.__version__)
print("sklearn version : ", skl.__version__)

#==============================================================================
def main():
	# ========== find the paths ==========
	dpath, cpath = syspath()
	ppath = "./plots/ShortPaper/PF00_BAassessment/"
	cf.pymkdir(ppath)
	# ========== all thes files are form 3.BAeval S02_Form
	dfan    = pd.read_csv(f"{ppath}S02_FormAna_dfan.csv", index_col=0)
	dfps    = pd.read_csv(f"{ppath}S02_FormAna_dfps.csv", index_col=0)
	dfscore = pickle.load(open(f"{ppath}S02_FormAna_dfscore.p", "rb"))
	data    = pickle.load(open(f"{ppath}S02_FormAna_data.p", "rb"))
	TEvents = pickle.load(open(f"{ppath}S02_FormAna_TEvents.p", "rb"))

	# ========== Make the plot for the paper ==========
	_plotmaker(dfan, dfps, dfscore, data, TEvents)
	# breakpoint()



def _plotmaker(dfan, dfps, dfscore, data, TEvents, forest=False):
	ppath = "./plots/ShortPaper/PF00_BAassessment/"
	cf.pymkdir(ppath)

	if not forest:
		dfps.drop("ForestryEst", inplace=True)
	# breakpoint()

	# ========== Setup the figure ==========
	# 'family' : 'normal',
	font = {'weight' : 'bold', #,
	        'size'   : 11}
	mpl.rc('font', **font)
	sns.set_style("whitegrid")
	plt.rc('legend',fontsize=11)
	plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})

	fig, ax1 = plt.subplots(
		1, 1,figsize=(8,7), num=("Accuracy assessment"), dpi=130)
	
	# fontsize=20
	cmapHex = palettable.colorbrewer.qualitative.Paired_6.hex_colors

	dfps.plot.bar(rot=0, color=cmapHex[1:], ax=ax1) #"Correct Non Detection",
	ax1.legend(
		["Correct Detection",   "Spatial under est.", "Spatial over est.", "False Negative", "False Positive"])
	ax1.set_ylabel(f"% of total events", fontweight='bold')
	labels = []
	for label in ax1.get_xticklabels():
		try:
			s1 = data[label.get_text()]["pubname"]
		except KeyError:
			s1 = "Forestry Est."
		s2 = TEvents[label.get_text()]
		labels.append(f"{s1} (n={s2})")
		# f"{data[label.get_text()]["pubname"]} (n={TEvents[label.get_text()]})"
	# labels = [f"{ data[label.get_text()]["pubname"]} (n={TEvents[data[label.get_text()]]})" for label in ax1.get_xticklabels()]
	ax1.set_xticklabels(labels, rotation=30, ha="right")



	# dfan.plot.bar(rot=0, color=cmapHex, ax=ax2)
	# ax2.legend(["Correct Non Detection", "Correct Detection", "Spatial under est.", "Spatial over est.", "False Negative", "False Positive"])
	# ax2.set_ylabel("Mean annual accuracy (%)", fontweight='bold')
	# labels2 = []
	# for label in ax2.get_xticklabels():
	# 	try:
	# 		s1 = data[label.get_text()]["pubname"]
	# 	except KeyError:
	# 		s1 = "Forestry Est."
	# 	s2 = data[label.get_text()]["end"] - data[label.get_text()]["start"] +1
	# 	labels2.append(f"{s1} ({s2} yrs.)")
	# 	# f"{data[label.get_text()]["pubname"]} (n={TEvents[label.get_text()]})"
	# # labels = [f"{ data[label.get_text()]["pubname"]} (n={TEvents[data[label.get_text()]]})" for label in ax1.get_xticklabels()]
	# ax2.set_xticklabels(labels2)

	# ========== Convert to dataframe ==========
	fig.tight_layout()

	plotfname = f"{ppath}PF00_BAassessment_mod."
	for fmt in ["pdf", "png"]:
		print(f"Starting {fmt} plot save at:{pd.Timestamp.now()}")
		plt.savefig(plotfname+fmt)#, dpi=dpi)
	
	# print("Starting plot show at:", pd.Timestamp.now())
	plt.show()

	if not (plotfname is None):
		maininfo = "Plot from %s (%s):%s by %s, %s" % (__title__, __file__, 
			__version__, __author__, dt.datetime.today().strftime("(%Y %m %d)"))
		gitinfo = pf.gitmetadata()
		infomation = [maininfo, plotfname, gitinfo]
		cf.writemetadata(plotfname, infomation)
	
	# ipdb.set_trace()
	# breakpoint()

def syspath():
	# ========== Create the system specific paths ==========
	sysname   = os.uname()[1]
	backpath = None
	if sysname == 'DESKTOP-UA7CT9Q':
		# spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/h"
		dpath = "/mnt/d/Data51"
		cpath = "/mnt/g/Data51/Climate/TerraClimate/"
	elif sysname == "ubuntu":
		# Work PC
		# dpath = "/media/ubuntu/Seagate Backup Plus Drive"
		# spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
		dpath = "/media/ubuntu/Harbinger/Data51"
	# elif 'ccrc.unsw.edu.au' in sysname:
	# 	dpath = "/srv/ccrc/data51/z3466821"
	elif sysname == 'DESKTOP-N9QFN7K':
		# The windows desktop at WHRC
		# dpath = "/mnt/f/Data51/BurntArea"
		dpath = "./data"
		cpath = "/mnt/f/Data51/Climate/TerraClimate/"
	elif sysname == 'DESKTOP-KMJEPJ8':
		dpath = "./data"
		# backpath = "/mnt/g/fireflies"
		cpath = "/mnt/i/Data51/Climate/TerraClimate/"
	elif sysname == 'arden-worstation':
		# WHRC linux distro
		dpath = "./data"
		cpath= "/media/arden/SeagateMassStorage/Data51/Climate/TerraClimate/"
		# dpath= "/media/arden/Harbinger/Data51/BurntArea"
	elif sysname == 'LAPTOP-8C4IGM68':
		dpath     = "./data"
		# backpath = "/mnt/d/fireflies"
		cpath = "/mnt/d/Data51/Climate/TerraClimate/"
	else:
		ipdb.set_trace()
	
	return dpath, cpath
# ==============================================================================
if __name__ == '__main__':
	main()