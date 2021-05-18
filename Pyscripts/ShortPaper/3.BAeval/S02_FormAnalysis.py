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
# import geopandas as gpd
import argparse
import datetime as dt
import warnings as warn
import xarray as xr
import bottleneck as bn
import scipy as sp
import glob
import dask
from dask.diagnostics import ProgressBar

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
# import geopy.distance as geodis

# import moviepy.editor as mpe
# import skvideo.io     as skv
# import skimage as ski
# from moviepy.video.io.bindings import mplfig_to_npimage
import pickle

import seaborn as sns
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
# import ipdb
# ipdb.set_trace()
if not os.uname()[1] =='arden-H97N-WIFI':
	import ipdb
else:
	import pdb as ipdb
# except Exception as e:
# 	print(str(e))
# 	warn.warn("failed to import ipdb")
# ipdb.set_trace()

# +++++ Import my packages +++++
import myfunctions.corefunctions as cf 
import myfunctions.PlotFunctions as pf 
# import MyModules.NetCDFFunctions as ncf

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

# ==============================================================================
def main():
	# ========== Read in the Site data ==========
	df = pd.read_csv("./results/ProjectSentinal/Fire sites and forest loss.csv") 
	df = renamer(df)

	dfq = pd.read_csv("./results/ProjectSentinal/Fire histories of all sites_QS.csv") 
	#Check and see if its in 
	clnm = ["Site - fieldwork (2018)", "Site - fieldwork (2017)"]
	sitenum = dfq[clnm].sum(axis=1).astype(int).values
	sitenm = [f"Site{stnu:02d}" for stnu in sitenum]
	nonsites = []
	for sn, sna in zip(sitenm, dfq["Site - fieldwork (2019)"].values):
		if sn in df.site.astype(str).values or sna in df.site.astype(str).values:
			continue
		nonsites.append((sn, sna))
	print(len(nonsites))
	
	# ========== Read in the Site data ==========
	cordname    = "./data/other/GEE_sitelist.csv"
	site_coords = pd.read_csv(cordname, index_col=0)
	site_coords["lonb_min"] = np.min([site_coords["lonb_COP_min"], site_coords["lonb_MOD_min"]], axis=0)
	site_coords["latb_min"] = np.min([site_coords["latb_COP_min"], site_coords["latb_MOD_min"]], axis=0)
	site_coords["lonb_max"] = np.max([site_coords["lonb_COP_max"], site_coords["lonb_MOD_max"]], axis=0)
	site_coords["latb_max"] = np.max([site_coords["latb_COP_max"], site_coords["latb_MOD_max"]], axis=0)
	# breakpoint()


	# ========== CHeck the data makes sense ==========
	dfchecker(df, site_coords)

	# ========== Pull out the field data ==========
	fd = Field_data(df, site_coords, force=True)

	# FieldHyptest(fd)
	# ipdb.set_trace()

	# ========== Find the save lists ==========
	spath, dpath = syspath()
	data         = datasets(dpath)

	# ========== Get some site specific infomation ==========
	siteseries  = OrderedDict()
	for site in site_coords.name.values:
		out = DisturbanceSeries(site, df, site_coords, spath)			
		# out = DisturbanceCounter(site, df, site_coords)
		if not out is None:
			siteseries[site]  =  out
			# sitesummary[site] = out
	
	dfsum = DisturbanceCounter(siteseries, df, site_coords, skipdec=False)#
	dfsum["RECRU"] = fd.loc[dfsum.index].Recruitment.values

	# # dfsum.boxplot(column="PostFLdist", by="RECRU")
	# # dfsum.boxplot(column="PostFLfire", by="RECRU")
	# ========== WOrk out how the histories compare ==========
	dtscore, dtsum = SiteHistScore(siteseries, site_coords, data, fd, spath, dpath)
	# breakpoint()
	# ipdb.set_trace()
	try :
		ax2 = sns.swarmplot(y="PostFLdist", x="RECRU", data=dfsum, order=["AR", "IR", "RF"])
		plt.figure(2)
		ax3 = sns.swarmplot(y="PostFLfire", x="RECRU", data=dfsum, order=["AR", "IR", "RF"])
		plt.show()
	except:
		ipdb.set_trace()

	# ========== Get out some basic infomation ==========
	ipdb.set_trace()
	
# ==============================================================================
# Field data investigation
# ==============================================================================
def FieldHyptest(fd):
	
	"""Script for looking at the field data"""
	
	fds = ~np.isnan(fd[["FBurn01", "FBurn02", "FBurn03", "FBurn04"]].astype(float))

	# ========= Build the plot ===========
	cmapHex = palettable.colorbrewer.qualitative.Set1_3_r.hex_colors
		
	ax  =  sns.swarmplot(y=fds.sum(axis=1), x=fd["Recruitment"], order=["AR", "IR", "RF"], palette=cmapHex)
	plt.ylabel("Number of Fire events")
	plt.xlabel("Recuitment pathway")
	plt.show()
	
	# Produce the plot 
	ipdb.set_trace()

# ==============================================================================
# BA product scorer 
# ==============================================================================
def SiteHistScore(siteseries, site_coords, data, fd, spath, dpath):

	"""
	Function to score the infomation in the field, BA and FL datasets 
	"""
	def _datasetopener(data, dsn, dfc, spath, dpath):
		"""Opens the dataset and extracts the correct location"""
		# ========== CHeck how to open the dataset ==========
		print (dsn)
		if "*" in data[dsn]["fname"]:
			fnames = glob.glob(data[dsn]["fname"])
			# ===== open the dataset =====
			ds = xr.open_mfdataset(fnames, chunks=data[dsn]["chunks"], combine='nested', concat_dim="time")
			if ds.latitude[0] < ds.latitude[-1]:
				ds = ds.sortby("latitude", ascending=False)
			# ========== fiz a time issue ==========
			if dsn == "COPERN_BA":
				ds["time"] = pd.date_range('2014-06-30', periods=6, freq='A')
		else:
			# ========== open the dataset ==========
			ds = xr.open_dataset(data[dsn]["fname"], chunks=data[dsn]["chunks"])

		# breakpoint()
		hist = ds[data[dsn]["var"]].sel(
			{"latitude":dfc.lat, "longitude":dfc.lon},method="nearest").groupby('time.year').max()
		with ProgressBar():
			hist = hist.compute()

		if dsn in ["Hansen", "Hansen_AFM"]:
			hist = (hist>0).astype(float)

		with ProgressBar():
			dsbox = ds[data[dsn]["var"]].sel(dict(latitude=slice(dfc.lat.max(), dfc.lat.min()), longitude=slice(dfc.lon.min(), dfc.lon.max()))).compute()
		
		siteh = OrderedDict()
		boxh  = OrderedDict()
		for nx in range(0, dfc.shape[0]):

			# ========== region search apprach here ==========
			dsba = dsbox.sel(dict(
				latitude=slice(dfc.latb_max[nx], dfc.latb_min[nx]), 
				longitude=slice(dfc.lonb_min[nx], dfc.lonb_max[nx])))
			boxh[dfc["name"][nx]] = (dsba>0).any(dim=["longitude", "latitude"]).astype(float).values * dsba.time.dt.year.values
			
			# ========== pull out just the site history ==========
			siteh[dfc["name"][nx]] = hist[:, nx, nx].values * hist.year.values

		# ========== Return the results ==========
		try:
			dfSH = pd.DataFrame(siteh).transpose()
			dfSH = dfSH.replace(0, np.NaN)
			dfBH = pd.DataFrame(boxh).transpose()
			dfBH = dfBH.replace(0, np.NaN)
			if not all((dfSH > 0).sum(axis=1) <= (dfBH > 0).sum(axis=1)):
				breakpoint()
		except:
			ipdb.set_trace()
		# breakpoint()
		return dfSH, dfBH
	
	def _SiteScorere(siteseries, yrst, yrfn, dsn, dfsh, dfbh):
		Accuracy = OrderedDict()
		for site in siteseries.keys():
			# score
			FP = 0
			FN = 0
			CD = 0
			DD = 0
			SDo = 0
			SDu = 0
			CND = 0

			# Manual clas
			dfSEN = siteseries[site]

			# +++++ from the disturbance data +++++
			try:
				dfda  = dfsh.loc[site].dropna().values
				dfba  = dfbh.loc[site].dropna().values
			except Exception as er:
				print(str(er))
				breakpoint()

			# ++++++++++ Manual Clasification ++++++++++
			# Get the burns as years
			if dsn == "Hansen":
				bnyr = pd.DatetimeIndex(dfSEN.date[np.logical_and(dfSEN.SImp ==1, dfSEN.SiteLoss == 1)].values).year.values				
				abny = pd.DatetimeIndex(dfSEN.date[(dfSEN.SiteLoss ==1)].values).year.values
			elif dsn == "Hansen_AFM":
				# breakpoint()
				bnyr = pd.DatetimeIndex(dfSEN.date[np.logical_and(dfSEN.Fire ==1, np.logical_and(dfSEN.SImp ==1, dfSEN.SiteLoss == 1))].values).year.values
				abny = pd.DatetimeIndex(dfSEN.date[(dfSEN.Fire ==1)].values).year.values

			else:
				bnyr = pd.DatetimeIndex(dfSEN.date[np.logical_and(dfSEN.Fire ==1, dfSEN.SImp ==1)].values).year.values
				#Creat a search for spatial uncertianty
				abny = pd.DatetimeIndex(dfSEN.date[(dfSEN.Fire ==1)].values).year.values
			# get the ones in the correct range
			bnyr = bnyr[bnyr >= yrst]
			bnyr = bnyr[bnyr <= yrfn]
			abny = abny[abny >= yrst]
			abny = abny[abny <= yrfn]

			if len(bnyr) == 0 and len(dfda) == 0:
				CND += 1
			elif len(abny) == 0 and len(dfda) > 0: #No adjucent breaks either
				# breakpoint()
				FP = len(dfda)

			elif len(bnyr) > 0 and len(dfba) == 0: 
				# No fires in the box
				FN = len(bnyr)
			else:
				cv = np.hstack([bnyr, dfda])
				# if dsn == "esacci":
				# 	breakpoint()
				for year in np.unique(np.hstack([bnyr, dfda])):
					if not year in cv:
						# Test to see if a number has already been delt with
						continue
					elif year in dfda and year in bnyr:
						# same year in both datasets
						# breakpoint()
						CD += 1

					elif year in bnyr:
						# its a burn that happened
						if year in dfba:
							# Check to see if it might just be a spatial missmach
							SDu += 1
						elif (year+1 in dfda and year+1 in cv) or (year-1 in dfda and year-1 in cv):
							#Check for an off by one error, cv to prevent dulicates
							CD += 1
							# DD += 1
							# breakpoint()
							cv[cv == (year+1)] = np.NaN
							cv[cv == (year-1)] = np.NaN
						
						else:
							# It was missed in the RS
							FN += 1

					elif year in dfda:
						# a fire in the RS data
						if year in abny:
							# Check to see if it might just be a spatial missmach
							SDo += 1
						elif (year+1 in bnyr and year+1 in cv) or (year-1 in bnyr and year-1 in cv):
							#Check for an off by one error, cv to prevent dulicates
							# DD += 1
							CD += 1
							# breakpoint()
							cv[cv == (year+1)] = np.NaN
							cv[cv == (year-1)] = np.NaN
						else:
							# It was missed in the RS
							FP += 1

					else:
						ipdb.set_trace()
					cv[cv == year] = np.NaN


			# if CND > 0:
			# 	FP = np.NAN
			# 	FN = np.NAN
			# 	CD = np.NAN
			# 	DD = np.NAN
			# 	SD = np.NAN

			# if all([FP == 0, FN == 0, CD == 0, DD==0]):
			
			Accuracy[site]= ({
				"CorrectNonDetection":CND, "CorrectDetection": CD, "SpatialDetectionUnder":SDu,  "SpatialDetectionOver":SDo,  "FalseNegative":FN, #"DelayedDetection": DD,
				"FalsePositive":FP, "TotalEvents":np.sum([CD, SDu, SDo, FN, FP, CND])})

			# breakpoint()

		# ========== Add the results of the dataset to the dictionary ===========
		# breakpoint()

		acu = pd.DataFrame(Accuracy).transpose()
		return acu
	# ========== get the coords ==========
	cords = OrderedDict()
	for index, row in site_coords.iterrows():
		sitex = row["name"]
		if not sitex in siteseries.keys():
			continue
		else:
			# ========== Set up the box ==========
			cords[sitex] = row[["name", "lat", "lon", "lonb_max", "lonb_min", "latb_max", "latb_min"]]

	# =========== pull out the dataset infomation ==========
	dfc = pd.DataFrame(cords).transpose()
	dfscore = OrderedDict()
	for dsn in data:
		with dask.config.set(**{'array.slicing.split_large_chunks': True}):
			dfsh, dfbh = _datasetopener(data, dsn, dfc, spath, dpath)
		# dfsh = dfsh.replace(0, np.NaN)
		# ========== loop over the sites ==========
		acu = _SiteScorere(siteseries, data[dsn]["start"], data[dsn]["end"], dsn, dfsh, dfbh)
		dfscore[dsn] = acu

	# =========== pull out the Forestry Estimates ==========
	acu = _SiteScorere(siteseries, 1987, 2018, "ForestryEst", fd[["FBurn01","FBurn02","FBurn03","FBurn04"]], fd[["FBurn01","FBurn02","FBurn03","FBurn04"]])
	dfscore["ForestryEst"] = acu

	# ========== Process the datasets to get some form of accuracy estimate ==========
	processed = OrderedDict()
	annual    = OrderedDict()
	TEvents   = OrderedDict()
	for dsn in dfscore:
		# ========== loop over the datasets ==========
		dfds = dfscore[dsn]#.dropna()

		events = dfds.drop(["TotalEvents", "CorrectNonDetection"], axis=1)
		TEvents[dsn] = events.sum(axis=1).sum()

		# ========== Pull out the results ===========
		processed[dsn] = ( events.div(events.sum(axis=1), axis=0) * 100).mean(axis=0)#.drop("TotalEvents")

		# ========== Qork out the annual accuracy ===========
		if dsn == "ForestryEst":
			continue
		nyears = data[dsn]["end"] - data[dsn]["start"] +1
		events["CorrectNonDetection"] = nyears-  events.sum(axis=1)
		annual[dsn] = (events / nyears).mean(axis=0) * 100
		# breakpoint()

	dfan = pd.DataFrame(annual).transpose()
	dfan = dfan[["CorrectNonDetection"]+dfan.columns.values[:-1].tolist()]
	dfps = pd.DataFrame(processed).transpose()
	
	def _plotmaker(dfan, dfps, dfscore, data, TEvents):
		ppath = "./plots/ShortPaper/PF00_BAassessment/"
		cf.pymkdir(ppath)

		# ========== Setup the figure ==========
		font = {'family' : 'normal',
		        'weight' : 'bold', #,
		        'size'   : 8}
		mpl.rc('font', **font)
		sns.set_style("whitegrid")
		plt.rc('legend',fontsize=6)
		plt.rcParams.update({'axes.titleweight':"bold", 'axes.titlesize':8})

		fig, (ax1, ax2) = plt.subplots(
			2, 1,figsize=(15,10), num=("Accuracy assessment"), dpi=130)
		
		# fontsize=20
		cmapHex = palettable.colorbrewer.qualitative.Paired_6.hex_colors

		dfps.plot.bar(rot=0, color=cmapHex[1:], ax=ax1) #"Correct Non Detection",
		ax1.legend(
			["Correct Detection",   "Spatial under est.", "Spatial over est.", "False Negative", "False Positive"])
		ax1.set_ylabel("% of total measurments", fontweight='bold')
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
		ax1.set_xticklabels(labels)


	
		dfan.plot.bar(rot=0, color=cmapHex, ax=ax2)
		ax2.legend(["Correct Non Detection", "Correct Detection", "Spatial under est.", "Spatial over est.", "False Negative", "False Positive"])
		ax2.set_ylabel("Mean annual accuracy (%)", fontweight='bold')
		labels2 = []
		for label in ax2.get_xticklabels():
			try:
				s1 = data[label.get_text()]["pubname"]
			except KeyError:
				s1 = "Forestry Est."
			s2 = data[label.get_text()]["end"] - data[label.get_text()]["start"] +1
			labels2.append(f"{s1} ({s2} yrs.)")
			# f"{data[label.get_text()]["pubname"]} (n={TEvents[label.get_text()]})"
		# labels = [f"{ data[label.get_text()]["pubname"]} (n={TEvents[data[label.get_text()]]})" for label in ax1.get_xticklabels()]
		ax2.set_xticklabels(labels2)

		# ========== Convert to dataframe ==========
		plotfname = f"{ppath}PF00_BAassessment."
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
		
		for df, dfn in zip([dfan, dfps], ["dfan", "dfps"]):	df.to_csv(f"{ppath}S02_FormAna_{dfn}.csv")
		
		for ite, itn in zip([dfscore, data, TEvents],["dfscore", "data", "TEvents"]):	pickle.dump(ite, open(f"{ppath}S02_FormAna_{itn}.p", "wb"))
		ipdb.set_trace()
		breakpoint()

	_plotmaker(dfan, dfps, dfscore, data, TEvents)
	breakpoint()
	return dfscore, dfps

# ==============================================================================
# Time Series builder
# ==============================================================================

def DisturbanceSeries(site, df, site_coords, spath, skipdec=False):
	"""
	Function to i1nterogate the disturbances and return some from of time series
	args : 
		site:			str
			name of the site
		df:				pd.dataframe
			the results of the google form
		site_coords:	pd.dataframe
			the site_coords i've created so far
	"""
	def _standloss(dfs, dates, frame, IsFire, SiteDis, StandRep, site, spath):

		# ========== Container to hold the standloss guesses ==========
		stloss = []
		for index, row in dfs.iterrows():
			# ========== check if the dates are accessable ==========

			datecsv = spath + "%s/LANDSAT_5_7_8_%s_SNRGB_timeinfo.csv" % (site, site)
			if os.path.isfile(datecsv):
				dft = pd.read_csv(datecsv, index_col=0, parse_dates=True)
			else:
				dft = None	
			for event in range(0, frame.loc[index].values.shape[0]):
				evinfo = OrderedDict()
				evinfo["Frame"] = frame.loc[index][event]
				evinfo["date"]  = dates.loc[index][event]
				evinfo["Fire"]  = IsFire.loc[index][event]
				evinfo["SImp"]  = 1.0
				evinfo["Agree"] = row["Name"]

				# ========== Check the dates ==========
				if not dft is None and not np.isnan(frame.loc[index][event]):
					tme = pd.Timestamp(dft.loc[frame.loc[index][event]].date[:10])
					if tme == evinfo["date"]:
						pass
					elif abs(tme - evinfo["date"]) <pd.Timedelta("90 days"): 
						evinfo["date"] = tme
					else:
						print("The date is incorrect", tme, evinfo["date"])
						warn.warn("\n\n\n\n\n\nFrame date checking is needed here!!!!!!!!!\n\n\n\n")
						ipdb.set_trace()
				
				if np.isnan(frame.loc[index][event]):
					evinfo["Agree"]    = None
					evinfo["SImp"]     = np.NaN
					evinfo["SiteLoss"] = np.NaN
				elif not SiteDis.loc[index][event] == 1:
					# ===== Skip if not inpacting the site ==========
					evinfo["SImp"]     = 0.0
					evinfo["SiteLoss"] = 0.0
				elif (IsFire.loc[index][event] == 1 and StandRep.loc[index][event] == 1):
					# ===== Add stand replacing fires =====
					evinfo["SiteLoss"] = 1.0
				elif IsFire.loc[index][event] == 0:
					evinfo["SiteLoss"] = 1.0
				else:
					# evinfo["Agree"]    = None
					# evinfo["SImp"]     = np.NaN
					evinfo["SiteLoss"] = 0
				# if site == 'Site08':
				# 	ipdb.set_trace()
				stloss.append(evinfo)

		dfSite = pd.DataFrame(stloss)
		return dfSite.sort_values(by=["Frame"]).reset_index(drop=True)
	
	def _eventGroup(dfs, dfSite, frame):
		"""Loop over the rows and look for similar events """
		# ========== add an event group column ==========
		dfSite["group"] = 0
		dfSite["group"][np.isnan(dfSite["Frame"])] = np.NaN
		EG = 1
		for row in range(0, dfSite.shape[0]):
			# +++++ Skip the bad rows +++++
			if np.isnan(dfSite["group"][row]):
				break
			elif dfSite["group"][row] > 0:
				pass
			else:
				fdif = abs(dfSite.Frame - dfSite.loc[row].Frame) < 5
				dtyp = dfSite.Fire == dfSite.loc[row].Fire
				grouped = np.logical_and(fdif, dtyp)
				if any(dfSite["group"][grouped]> 0):
					warn.warn("Shit is weird here")
					ipdb.set_trace()

				dfSite["group"][grouped] = EG
				EG += 1
		# ========== Simplify the grouped events ==========
		sim_events = []
		for group in range(1, EG):
			evinfo = OrderedDict()
			dfsub = dfSite[dfSite.group == group]
			evinfo["Frame"]    = dfsub.Frame.min()
			evinfo["date"]     = dfsub.date.min()
			evinfo["Fire"]     = dfsub.Fire.max()
			evinfo["SImp"]     = dfsub.SImp.max()
			evinfo["Agree"]    = float(dfsub.Agree.unique().shape[0])/dfs.shape[0]
			evinfo["SiteLoss"] = dfsub.SiteLoss.max()
			sim_events.append(evinfo)

		# ========== add empty rows ==========
		while len(sim_events) < frame.shape[1]:
			evinfo = OrderedDict()
			evinfo["Frame"]    = np.NaN
			evinfo["date"]     = pd.Timestamp(np.NaN)
			evinfo["Fire"]     = np.NaN
			evinfo["SImp"]     = np.NaN
			evinfo["Agree"]    = np.NaN
			evinfo["SiteLoss"] = np.NaN
			sim_events.append(evinfo)
		return pd.DataFrame(sim_events)

	# ========== subset the dataset so only site data is present ==========
	dfs = df[df.site == site]

	
	# ========== Check the number of obs ==========
	if dfs.shape[0] == 0:
		return None
	elif skipdec and (dfs.SiteConif.values.sum() == 0):
		# +++++ site is not conifereous forest +++++
		return None
	else:
		pass
	
	def _colky(dfs, instr):
		# ========== Function to check if a string ecists in a columns ==========
		locs = []
		for ky in dfs.columns:
			locs.append(ky.startswith(instr))
		return dfs.columns.values[locs]

	# # ========== Loop over each of the events ==========
	dates    = dfs[_colky(dfs, "Date")]
	frame    = dfs[_colky(dfs, "Frame")]
	IsFire   = dfs[_colky(dfs, "IsFire")]
	SiteDis  = dfs[_colky(dfs, "SiteImpacted")]
	StandRep = dfs[_colky(dfs, "StandRep")]
	
	# ========== Find the stand loss event ==========
	dfSite  = _standloss(dfs, dates, frame, IsFire, SiteDis, StandRep, site, spath)

	if dfs.shape[0] > 1:
		dfSite = _eventGroup(dfs, dfSite, frame)
	else:
		dfSite["Agree"] = (~dfSite.Agree.isnull()).astype(float)
		dfSite["Agree"][dfSite["Agree"]==0] = np.NaN
	
	dfSite["PostStLs"] = 0
	try:
		dfSite["PostStLs"][np.min(np.where(dfSite.SiteLoss.values == 1)):] = 1
	except Exception:
		# No stand replacing fire 
		pass
	dfSite["PostStLs"][np.isnan(dfSite.Frame)] = np.NaN
	if dfSite.shape[0] > frame.shape[1]:
		warn.warn("This loc has more than normal number of disturbances")
		print(dfSite.shape)
	return dfSite
	

# ==============================================================================
# Borad data interrogation
# ==============================================================================

def DisturbanceCounter(siteseries, df, site_coords, skipdec=True):
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
	
	sitesummary = OrderedDict()

	for site in siteseries: 

		# ========== subset the dataset so only site data is present ==========
		dfs = df[df.site == site]
		
		# ========== Check the number of obs ==========
		if dfs.shape[0] == 0:
			continue
		elif skipdec and (dfs.SiteConif.values.sum() == 0):
			# +++++ site is not conifereous forest +++++
			continue
		else:
			print(site, dfs.shape[0], ~(dfs.SiteConif.values.sum() == 0))


		# ========== work out the number of disturbances after the stand loss ==========
		SiteTS = siteseries[site]
		stdist = np.logical_and(SiteTS.PostStLs==1, SiteTS.SImp==1)

		if any(stdist):
			PLSiD = np.sum(stdist) - 1
			PLSiF = np.sum(np.logical_and(stdist, SiteTS.Fire==1)) - SiteTS.Fire[np.where(stdist)[0][0]]
			
			
		else:
			PLSiD = np.NaN
			PLSiF = np.NaN

		# ========== Work out the total number of disturbances ==========
		if dfs.shape[0] > 1:
			# ========== Loop over the user obs ==========
			obs = []
			for index, row in dfs.iterrows():
				vals, kys = _rowcounter(row)
				obs.append(vals)
			 
			obsnp   = np.vstack(obs)
			distnum = np.hstack([
				np.mean(obsnp, axis=0), 
				np.min(obsnp, axis=0), 
				np.max(obsnp, axis=0), 
				PLSiD, PLSiF,
				dfs.SiteConif.values.mean()])
			# ipdb.set_trace()	
		else:
			# ===== subset the row =====
			row = dfs.iloc[0]

			# ===== get the values =====
			distnum     = np.zeros(15)
			distnum[:]  = np.NAN
			vals, kys   = _rowcounter(row)
			distnum[:4] = vals
			distnum[-3] = PLSiD 
			distnum[-2] = PLSiF
			distnum[-1] = dfs.SiteConif.values.mean()
		
		# ========== convert the results to a pandas series ==========
		# +++++ Make the keys +++++
		fullkys = []
		for vari in ["Mean", "Min", "Max"]:
			for ky in kys:
				fullkys.append(vari+ky)
		for cname in  ["PostFLdist", "PostFLfire", "SiteConif"]:
			fullkys.append(cname)
		sitesummary[site] =  pd.Series(distnum, index=fullkys)

	return pd.DataFrame(sitesummary).transpose()

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
		elif col.startswith("What was the frame number?"):
			rename[col] = "Frame%02d" % distcount
		elif col.startswith("When did the event happen?"):
			rename[col] = "Date%02d" % distcount
			df[col]     = pd.to_datetime(df[col])
		elif col.startswith("If fire, was it stand replacing?"):
			rename[col] = "StandRep%02d" % distcount
			df[col]     = df[col].map({
				'Yes': 1,'Yes at the Site, no in other places in the box': 1,
				'No': 0, 'No at the site, yes in other parts of the box': 0})
		elif col.startswith("Any Additional Comments"):
			distcount  += 1 #add to the disturbance counter
			
			# ipdb.set_trace()

	# rename[""]
	# rename[""]

	# ========== Fix known bad values  ==========
	df = df.replace("Site 09", "Site09")
	df = df.replace("<25%", "<30%")
	df = df.rename(columns=rename)

	# ========== Fix column values ==========
	df["SiteConif"] = df["SiteConif"].map({'Yes': 1,'yes': 1,'No': 0, 'no': 0})
	# ipdb.set_trace()
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

def Field_data(df, site_coords, force=False):
	"""
	# Aim of this function is to look at the field data and pull out the RF status 
	"""
	if os.path.isfile("./data/field/ProcessedFD.csv") and not force:
		dfds = pd.read_csv("./data/field/ProcessedFD.csv", index_col=0)
		return dfds
	else:
		import geopy.distance as geodis

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
		
		RFinfo["RF"][RFinfo["RF"].str.contains("poor")] = "RF"  #"no regeneration"
		RFinfo["RF"][RFinfo["RF"].str.contains("no regeneration")] = "RF" 
		
		RFinfo["RF"][RFinfo["RF"].str.contains("singular")] = "IR"  
		
		for repstring in ["sufficient", "sufficent", "sifficient"]:
			RFinfo["RF"][RFinfo["RF"].str.contains(repstring)] = "IR" 
		
		for repstring in ["abundunt", "abundant"]:
			RFinfo["RF"][RFinfo["RF"].str.contains(repstring)] = "AR" 
		
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
	dfds = pd.DataFrame(info).transpose()
	dfds.to_csv("./data/field/ProcessedFD.csv", header=True)
	return dfds

# ==============================================================================
# Containers for infomation
# ==============================================================================

def datasets(dpath):
	# ========== set the filnames ==========
	# breakpoint()
	data= OrderedDict()
	# 
	data["MODIS"] = ({
		"fname":dpath + "MODIS/MODIS_MCD64A1.006_500m_aid0001_reprocessedBAv2.nc",
		'var':"BA", "gridres":"500m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'longitude': 1000, 'latitude': 10000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/MODIS/MASK/MCD12Q1.006_500m_aid0001v2.nc",
		"pubname":"MCD64A1"
		})
	data["COPERN_BA"] = ({
		'fname':dpath + "COPERN_BA/processed/COPERN_BA_gls_*_SensorGapFix.nc",
		'var':"BA", "gridres":"300m", "region":"Global", "timestep":"AnnualMax",
		"start":2014, "end":2019,"rasterio":False, "chunks":{'time':1,'longitude': 1000, 'latitude': 10000}, 
		"rename":{"lon":"longitude", "lat":"latitude"}, "pubname":"CGLS-BA"
		})
	data["esacci"] = ({
		"fname":dpath + "esacci/processed/esacci_FireCCI_*_burntarea.nc",
		'var':"BA", "gridres":"250m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': 1000, 'latitude': 10000},
		"rename":None, "maskfn":"/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/esacci/processed/esacci_landseamask.nc",
		"pubname":"FireCCI51"
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	data["Hansen"] = ({
		"fname":dpath + "HANSEN/lossyear/Hansen_GFC-2018-v1.6_*_totalloss_SIBERIAatesacci.nc",
		'var':"lossyear", "gridres":"250m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': 1000, 'latitude': 10000},
		"rename":None, "pubname":"HansenGFC"
		# "rename":{"band":"time","x":"longitude", "y":"latitude"}
		})
	data["Hansen_AFM"] = ({
		"fname":dpath + "HANSEN/lossyear/Hansen_GFC-2018-v1.6_*_totalloss_SIBERIAatesacci_MODISAFmasked.nc",
		'var':"lossyear", "gridres":"250m", "region":"Siberia", "timestep":"Annual", 
		"start":2001, "end":2018, "rasterio":False, "chunks":{'time':1, 'longitude': 1000, 'latitude': 10000},
		"rename":None, "pubname":"HansenGFC-MAF"
		})
	return data


def syspath():
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-CSHARFM':
		# LAPTOP
		spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		# dpath = "/mnt/e"
		ipdb.set_trace()
	elif sysname == "owner":
		spath = "/mnt/c/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/mnt/d/Data51/BurntArea/"
	elif sysname == 'DESKTOP-KMJEPJ8':
		spath = "/mnt/c/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/mnt/i/Data51/BurntArea/"
	elif sysname == "ubuntu":
		# Work PC
		dpath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/BurntArea/"
		spath = "/media/ubuntu/Seagate Backup Plus Drive/Data51/VideoExports/"
	elif sysname == 'arden-H97N-WIFI':
		spath = "/mnt/FCBE3028BE2FD9C2/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/media/arden/Harbinger/Data51/BurntArea/"
	elif sysname == 'LAPTOP-8C4IGM68':
		spath = "/mnt/c/Users/user/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/mnt/i/Data51/BurntArea/"
	elif sysname == 'DESKTOP-UA7CT9Q':
		spath = "/mnt/c/Users/arden/Google Drive/UoL/FIREFLIES/VideoExports/"
		dpath = "/mnt/d/Data51/BurntArea/"
	else:
		ipdb.set_trace()
	return spath, dpath

# ==============================================================================

if __name__ == '__main__':
	main()