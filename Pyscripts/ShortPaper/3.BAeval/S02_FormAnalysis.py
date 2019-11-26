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
import geopy.distance as geodis

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

	# ========== Pull out the field data ==========
	fd = Field_data(df, site_coords)



	# ========== Get some site specific infomation ==========
	sitesummary = OrderedDict()
	for site in site_coords.name.values:
		out = DisturbanceCounter(site, df, site_coords)
		if not out is None:
			sitesummary[site] = out			

	dfsum = pd.DataFrame(sitesummary).transpose()
	dfsum["RECRU"] = fd.loc[dfsum.index].Recruitment.values
	plt.figure(1)
	dfsum.boxplot(column="MeanSitefire", by="RECRU")
	plt.figure(2)
	dfsum.boxplot(column="MeanSiteDist", by="RECRU")
	plt.show()
	ipdb.set_trace()
	# ========== Get out some basic infomation ==========

# ==============================================================================
# Data interrogation
# ==============================================================================
def DisturbanceCounter(site, df, site_coords, skipdec=True):
	"""
	Function to i1nterogate the disturbances
	args : 
		site:			str
			name of the site
		df:				pd.dataframe
			the results of the google form
		site_coords:	pd.dataframe
			the site_coords i've created so far
	"""
	
	def _rowcounter(row):
		""" Internal function to sumarise the disturbances in a row of the dataframe """
		# +++++ calculate the total number of disturbances and fires +++++
		TotDist  = row[[ky.startswith("Dist") for ky in row.keys().values]].sum()
		TotFire  = 0 

		# +++++ Same but at the site level +++++
		SiteDist = 0
		Sitefire = 0
		
		for event in range (1, 8):
			# ===== Subset the info for the specific event =====
			disinf = row[[ky.endswith("%02d" % event) for ky in row.keys().values]]

			if disinf.empty:
				continue
			elif np.isnan(disinf[0]):
				break
			else:
				TotFire  = bn.nansum([ TotFire, disinf["IsFire%02d" % event]])
				SiteDist = bn.nansum([SiteDist, disinf["SiteImpacted%02d" % event]])
				Sitefire = bn.nansum([Sitefire, (disinf["SiteImpacted%02d" % event] * disinf["IsFire%02d" % event])])

		return np.array([TotDist, TotFire, SiteDist, Sitefire]), ["TotDist", "TotFire", "SiteDist", "Sitefire"]
	
	# ========== subset the dataset so only site data is present ==========
	dfs = df[df.site == site]

	
	# ========== Check the number of obs ==========
	if dfs.shape[0] == 0:
		return None
	else:
		print(site, dfs.shape[0], ~(dfs.SiteConif.values.sum() == 0))
	

	Info = OrderedDict()
	if dfs.SiteConif.values.sum() == 0:
		# +++++ site is not conifereous forest +++++
		return None

	elif dfs.shape[0] > 1:
		# ========== Loop over the user obs ==========
		obs = []
		for index, row in dfs.iterrows():
			vals, kys = _rowcounter(row)
			obs.append(vals)
		 
		obsnp   = np.vstack(obs)
		distnum = np.hstack([np.mean(obsnp, axis=0), np.min(obsnp, axis=0), np.max(obsnp, axis=0), dfs.SiteConif.values.mean()])
		# ipdb.set_trace()	
	else:
		# ===== subset the row =====
		row = dfs.iloc[0]

		# ===== get the values =====
		distnum     = np.zeros(13)
		distnum[:]  = np.NAN
		vals, kys   = _rowcounter(row)
		distnum[:4] = vals
		distnum[-1] = dfs.SiteConif.values.mean()
	# ========== convert the results to a pandas series ==========
	# +++++ Make the keys +++++
	fullkys = []
	for vari in ["Mean", "Min", "Max"]:
		for ky in kys:
			fullkys.append(vari+ky)
	fullkys.append("SiteConif")
	return pd.Series(distnum, index=fullkys)
# ==============================================================================
# Raw Data correction and processing 
# ==============================================================================
def renamer(df):
	"""Function to rename my dataframe columns """

	rename = OrderedDict()
	rename['What is the name of the site?']         = "site"
	rename['Is the site clearly evergreen forest?'] = "SiteConif"
	rename["What fraction is Evergreen forest?"]    = "ConifFracBase"
	rename["Has the fraction of forest changed significantly before the first disturbance event?"] = "ConifFracDis"

	distcount = 1
	# ========== rename the disturbance columns ==========
	for col in df.columns.values:
		if col == "Would you like to log any disturbance events?" or col.startswith("Log another disturbance"):
			rename[col] = "Dist%02d" % distcount
			# +++++ change to a sumable float col +++++
			df[col]     = df[col].map({'Yes': 1,'yes': 1,'No': np.NAN, 'no': np.NAN})
		elif col.startswith("What is the Disturbance?"):
			rename[col] = "IsFire%02d" % distcount
			df[col]     = df[col].map({'Fire': 1, "Land use": 0})
		elif col.startswith("Was the site impacted?"):
			rename[col] = "SiteImpacted%02d" % distcount
			df[col]     = df[col].map({'Yes': 1,'yes': 1,'No': 0, 'no': 0})
			distcount  += 1 #add to the disturbance counter

	# rename[""]
	# rename[""]

	# ========== Fix known bad values  ==========
	df = df.replace("Site 09", "Site09")
	df = df.replace("<25%", "<30%")
	df = df.rename(columns=rename)

	# ========== Fix column values ==========
	df["SiteConif"] = df["SiteConif"].map({'Yes': 1,'yes': 1,'No': 0, 'no': 0})

	return df

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
	# df.replace("<25%", "<30%")


def Field_data(df, site_coords):
	"""
	# Aim of this function is to look at the field data and pull out the RF status 
	"""
	def _fireyear(val):
		# ========== setup the item to return ==========
		years    = np.zeros(4)
		years[:] = np.NAN
		
		# ========== deal with no data ==========
		try:
			if np.isnan(val):
				return years
		except:pass


		
		# ========== Fix the string ==========
		val = val.replace(" crown fire", "")
		val = val.replace("?", "")
		val = val.replace("and", "")
		val = val.replace(" ", "")
		val = val.replace("crownhigh-severity", "")
		val = val.replace("-crownfire", "")
		val = val.replace("-high-severitysurfacefire", "")
		val = val.replace("(03)", "2003")

		if val ==  "morethan10yearsago" or val == 'burnedtwice':
			return years

		# ========== Convert to years ==========
		year = int(val)
		if (year <= 2018) and (year >= 1980):
			years[0] = year
			return years
		elif len(val)%4 == 0:
			for nx in range(0, len(val), 4):
				year = int(val[nx:nx+4])
				if (year <= 2018) and (year >= 1980):
					years[nx//4] = year
				else:
					ipdb.set_trace()
				return years
		else:
			ipdb.set_trace()
			return np.NAN

	def _fielddataclean(row, fieldyear):
		siteinfo = OrderedDict()
		try:
			row = row.squeeze()
		except:
			pass

		if row is None:
			# This was a new 2019 SIte
			siteinfo["Recruitment"] = "RF"
			siteinfo["FBurn01"]     = 2015
			siteinfo["FBurn02"]     = np.NAN
			siteinfo["FBurn03"]     = np.NAN
			siteinfo["FBurn04"]     = np.NAN
			siteinfo["AltName"]     = np.NAN
		else:
			siteinfo["Recruitment"] = row["RF"]
			fires = _fireyear( row["estimated fire year"])
			siteinfo["FBurn01"]     = fires[0]
			siteinfo["FBurn02"]     = fires[1]
			siteinfo["FBurn03"]     = fires[2]
			siteinfo["FBurn04"]     = fires[3]
			if fieldyear == 2019:
				siteinfo["AltName"] = "Site%02d" % row["site number"]
			else:
				siteinfo["AltName"] = np.NAN

		return siteinfo

	def _fdfix(RFinfo):
		
		RFinfo.RF[    RFinfo["RF"].str.contains("poor")] = "RF"  #"no regeneration"
		RFinfo.RF[    RFinfo["RF"].str.contains("no regeneration")] = "RF" 
		RFinfo.RF[RFinfo["RF"].str.contains("singular")] = "IR"  
		for repstring in ["sufficient", "sufficent", "sifficient"]:
			RFinfo.RF[RFinfo["RF"].str.contains(repstring)] = "IR" 
		for repstring in ["abundunt", "abundant","sufficient", "sufficent", "sifficient"]:
			RFinfo.RF[RFinfo["RF"].str.contains(repstring)] = "AR" 
		return RFinfo

	fd18 = pd.read_csv("./data/field/2018data/siteDescriptions18.csv")
	fd17 = pd.read_csv("./data/field/2017data/siteDescriptions17.csv")

	fd18 = fd18.sort_values(by=["site number"]).reset_index(drop=True)
	fd18 = fd18.rename(columns={"rcrtmnt":"RF"}) 
	fd18 = _fdfix(fd18)
	fd17 = fd17.sort_values(by=["site number"]).reset_index(drop=True)
	fd17 = fd17.rename(columns={"strtY":"lat", "strtX":"lon", "recruitment":"RF"}) 
	fd17 = _fdfix(fd17)

	# ========== Loop over the Sites ==========
	info = OrderedDict()
	for index, row in site_coords.iterrows():
		# ========== Sort out the distance thing ==========
		if row.set == 2019:
			if row["name"] == "TestBurn":
				siteinfo = _fielddataclean(None, row.set)
			else:
				# ++++++++++ Calculate distance ++++++++++
				dist18 = np.array([geodis.distance((row.lat, row.lon), (lat, lon)).km for lat, lon in zip(fd18.lat, fd18.lon)])
				dist17 = np.array([geodis.distance((row.lat, row.lon), (lat, lon)).km for lat, lon in zip(fd17.lat, fd17.lon)])
				
				if np.min(dist18) <= np.min(dist17):
					siteinfo = _fielddataclean(fd18.iloc[np.argmin(dist18)].squeeze(), row.set)
				else:
					siteinfo = _fielddataclean(fd17.iloc[np.argmin(dist17)].squeeze(), row.set)
				
		elif row.set == 2018:
			sn = int(row["name"].strip("Site"))
			siteinfo = _fielddataclean(fd18.iloc[np.where(fd18["site number"].values == sn)[0]], row.set)
		elif row.set == 2017:
			sn = int(row["name"].strip("Site"))
			siteinfo = _fielddataclean(fd17.iloc[np.where(fd17["site number"].values == sn)[0]], row.set)
		
		info[row["name"]] = siteinfo
	# ========== Create and Ordered Dict for important info ==========
	return pd.DataFrame(info).transpose()

# ==============================================================================
if __name__ == '__main__':
	main()