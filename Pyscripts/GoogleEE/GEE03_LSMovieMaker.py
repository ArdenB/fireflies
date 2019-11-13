"""
Script goal, 

Test out the google earth engine to see what i can do
	- find a landsat collection for a single point 

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
import shutil
import time

from collections import OrderedDict
from scipy import stats
from numba import jit

# Import the Earth Engine Python Package
import ee
import ee.mapclient
from ee import batch

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

import imageio 
import moviepy.editor as mpe
# import skvideo.io     as skv
import skimage as ski
import skimage.exposure as exposure
from moviepy.video.io.bindings import mplfig_to_npimage


# import seaborn as sns
import matplotlib as mpl 
import cartopy.crs as ccrs
import cartopy.feature as cpf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
import socket
import myfunctions.corefunctions as cf 

# import cartopy.feature as cpf
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# # Import debugging packages 
import ipdb

print("numpy version  : ", np.__version__)
print("pandas version : ", pd.__version__)
print("xarray version : ", xr.__version__)

#==============================================================================

def main(args):


	# ========== Load the list of all sites ==========
	dpath       = syspath()
	cordname    = "./data/other/GEE_sitelist.csv"
	site_coords = pd.read_csv(cordname, index_col=0)
	force       = args.force
	verbose     = args.verbose
	multi       = True

	# ========== Check the site ==========
	for site, coords in site_coords.iterrows():
		if not args.site is None:
			if not args.site == coords.name:
				continue
		# if site in ["Burn2015 UP", "G10T1-0"]:#, "G11T1-0", "G12T1-0", "G13T1-0"]:
		# 	warn.warn("Skiping site until i have a better test protocol, this will need to be fixed")
		# 	continue
		print("\n" + site +"\n")

		scheck = SiteChecker(dpath, coords["name"], force, multi)
		
		if scheck == False:
			continue
		else:
			# ========== Process the images ==========
			try:
				images, bandcombo, bandlist, datelist = IMcleaner(dpath, site, scheck, coords, verbose, test = False)
			except IOError:
				continue

			# ipdb.set_trace()
			mpl.use('agg')
			if multi:
				MovieMaker(images, dpath, site, scheck, coords, bandlist, datelist, ["NRG", "RGB", "SNR"])
			else:
				for bands in bandcombo:
					MovieMaker(images, dpath, site, scheck, coords, bandlist, datelist, bands)
			# ipdb.set_trace()

	ipdb.set_trace()
	sys.exit()

#==============================================================================

def IMcleaner(dpath, site, scheck, coords, verbose, test = False, dsinfom = "LANDSAT_5_7_8"):
	"""
	Function to clean the data up and make the visualisation better
	"""
	# ========== setup the key params ==========
	SF       = 0.0001 # Scale Factor
	fnames   = scheck["fnames"]
	spath    = dpath + "UoL/FIREFLIES/VideoExports/%s" % coords["name"]
	comple   = True #Boolean statment, changed to false if a var fails
	fails    = []

	# ========== create a check ==========
	datelist = scheck["dates"].copy()
	datelist["QCpass"] = True
	
	# ========== Work out how many bands ==========
	bandcombo, bandlist = _bandcom(fnames["fnames"][0])

	# ========== setup the containers ==========
	images = OrderedDict()
	for ky in bandcombo.keys():
		images[ky] = []

	# ========== Loop over the files ==========
	for nx, row in fnames.iterrows():
		string = "\r processing %d of %d" % (nx, fnames.index.max())
		sys.stdout.write(string)
		sys.stdout.flush()

		# ========== create the fname ==========
		fnn = row[0]

		# ========== get the date and the sensor ==========
		date = pd.Timestamp(scheck["dates"]["date"][nx])
		sens = scheck["dates"]["satellite"][nx]


		# =========== Open the file and scale it ==========
		try:
			da_in  = xr.open_rasterio(fnn).transpose("y", "x", "band").rename({"x":"longitude", "y":"latitude"}) * SF
		except:
			warn.warn("File %d Failed  to open. Setting up repair")

			# ========== make sure the complete file is not written out ==========
			comple = False

			# ========== delete the bad file ==========
			print(" Existing file is broken. deleted")
			os.remove(fnn)

			# ========== append to fails ==========
			fails.append(nx)
			continue
		
		if not bandlist is None:
			da_in["band"] = bandlist

		# =========== create a first fail filter ==========
		nanfrac = np.isnan(da_in).sum().astype(float)/np.prod(da_in.shape)
		if nanfrac > 0.50:
			datelist["QCpass"][nx] = False
			continue

		# ========== Remove duplicate frames ==========
		if (nx > 1) and not verbose:
			# Get the previous frame times
			dateold = pd.Timestamp(scheck["dates"]["date"][nx-1])
			sensold = scheck["dates"]["satellite"][nx-1] 

			tests = ([
				((date - dateold) < pd.Timedelta(1, unit='d')), #same day
				sensold == sens, #same Sensor
				datelist["QCpass"][nx-1]]) # passed the QC 
			if all(tests):
				# There is already a vilid image here
				datelist["QCpass"][nx] = False
				continue

		# ========== Loop over the band combinations ==========
		for bnd in bandcombo:
			def _imagefixer(nx, da_in, bandcombo, bnd, date, sens):
				
				da_sel  = da_in.sel(band=bandcombo[bnd])

				# ========== left here for debugging ===========
				# debugrow =[246]
				# if nx in debugrow:
				# 	# da_sel.plot.imshow(rgb="band")
				# 	# plt.show()
				# 	ipdb.set_trace()

				# ========== Deal with the landsat 7/8 winter problem ==========
				# This may need to become more complicated in the future
				if (date.month in [11, 12, 1, 2, 3]) and (sens in ['LANDSAT_7', 'LANDSAT_8']):
					da_sel =  da_sel.where(np.logical_or((da_in < -0.2), (da_sel>=0)), 0)
				
				# =========== mask out dodgy values ==========
				da_sel =  da_sel.where(da_sel <= 1)
				da_sel =  da_sel.where(da_sel >= 0)
				noNaN  = ~da_sel.reduce(bn.anynan, dim='band')
				da_sel =  da_sel.where(noNaN)
				noNaF  =  noNaN.values.astype(float) 
				noNaF[noNaF ==0] = np.NAN
				
				# =========== Add infomation so i can look it in a dataframe ==========
				gfrac = float((noNaN.sum().values)/noNaN.size)

				# =========== Pull out the values ==========
				img = da_sel.values

				# =========== Convert them to a 1d array ==========
				try:
					pL, pH = np.percentile(img[~np.isnan(img)], (1, 99))

					# rescale the intensity to make the image brighter
					imgo = ski.exposure.rescale_intensity(img, in_range=(pL, pH))

					plot= False
					if plot:
						plt.figure(3)
						plt.imshow(imgo) 
						plt.show()

						hist = ski.exposure.histogram(img[~np.isnan(img)])
						histo = ski.exposure.histogram(imgo[~np.isnan(imgo)]) 
						plt.figure(2)
						plt.plot(histo[1], histo[0])
						plt.show()
					
					da_out  =  da_sel.copy(data=imgo)

				except IndexError:
					# meanMod = np.array([np.NAN, np.NAN, np.NAN])
					da_out  = da_sel.copy()
				
				# ========== convert to smart xarray dataset ==========
				da_out = da_out.expand_dims("time")
				da_out = da_out.assign_coords(time=[date])

				return da_out
			
			# ========== adjust the image brightness ==========
			da_img =  _imagefixer(nx, da_in, bandcombo, bnd, date, sens)
			# if image fails some test: pass
			images[bnd].append(da_img)
	
	if comple:
		# ========== Setup the datelist for a return ==========
		datelist = (datelist[datelist.QCpass == True]).reset_index(drop=False,inplace=False)
		return images, bandcombo, bandlist, datelist
	else:
		print("%d files failed to load. the complete file list has been deleted" % len(fails))
		# ========== save the failed numbers ==========
		fail = np.array(fails)
		np.save(spath+"/raw/failed_geotifs.npy", fail)


		# ========== generate new files ==========
		import subprocess as subp
		subp.call("ipython ./Pyscripts/GoogleEE/GEE02_LandsatMovie.py -- --site %s" % (site), shell=True)

		# ========== delete the complete build list ==========
		os.remove("%s/raw/%s_CompleteFileList.csv" % (spath, site))

		# ========== raise and error ==========
		raise IOError

#==============================================================================
def MovieMaker(images, dpath, site, scheck, coords, bandlist, datelist, bands):
	""" Function to build the movie """

	failed = 0
	while failed <2:
		spath    = dpath + "UoL/FIREFLIES/VideoExports/%s" % coords["name"]
		
		# for bands in bandcombo:
		print("\n starting %s at:" % bands, pd.Timestamp.now())

		# ========== Create a single dataarray for the raster images ===========
		sets = OrderedDict()
		if type(bands) == str:
			imstack     = images[bands]
			sets[bands] =  xr.concat(imstack, dim="time")
			fnout = "%s/LANDSAT_5_7_8_%s_%s.mp4" % (spath, coords["name"], bands) 
		elif type(bands) == list:
			bndnm = "multi_" + "_".join(bands)
			for bnd in bands:
				imstack   = images[bnd]
				sets[bnd] =  xr.concat(imstack, dim="time")
			fnout = "%s/LANDSAT_5_7_8_%s_%s.mp4" % (spath, coords["name"], bndnm) 
		else:
			ipdb.set_trace()


		# ========== Loop over each frame of the video ==========
		nx = []

		def frame_maker(index):

			# ========== Pull the infomation from the pandas part of the loop ==========
			indx  = int(index) 
			info  = datelist.iloc[int(indx)] #rowinfo[1]
			
			# # ========== Check the dates i'm exporting ==========
			# nx.append(frame.time.values)

			# ========== create and internal subplot ==========
			def _subplotmaker(ax, bnds, spt):
				# ========== Get the data for the frame ==========
				frame = sets[bnds].isel(time=int(indx))
				# ========== Set the colors ==========
				# if bnds == "NRG":
				color = "blue"
				# else:
				# 	color = "purple"
				# ========== Grab the data ==========
				frame.plot.imshow(ax=ax, rgb="band")# , transform=ccrs.PlateCarree())

				## =========== Setup the annimation ===========
				ax.set_title(spt)

				ax.scatter(coords.lon, coords.lat, 5, c=color, marker='+')#, transform=ccrs.PlateCarree())
				
				# ========== Set up the box ==========
				blonO = np.min([coords["lonb_COP_min"], coords["lonb_MOD_min"]])
				blatO = np.min([coords["latb_COP_min"], coords["latb_MOD_min"]])
				blonM = np.max([coords["lonb_COP_max"], coords["lonb_MOD_max"]])
				blatM = np.max([coords["latb_COP_max"], coords["latb_MOD_max"]])

				rect = mpl.patches.Rectangle(
					(blonO,blatO),
					blonM-blonO,
					blatM-blatO,linewidth=2,edgecolor=color,facecolor='none')
				ax.add_patch(rect)
				# +++++ change the number od ticks
				ax.xaxis.set_major_locator(plt.MaxNLocator(5))

				
			# ========== Build the plots ==========
			if type(bands) == str:
				# Set up the figure
				fig, axs = plt.subplots(1, figsize=(11,10))
				#  create the title 
				spt = "%s %s %s frame %d" % (bands, info.satellite, info.date.split(" ")[0], datelist.iloc[indx]["index"])
				# make the figure
				_subplotmaker(axs, bands, spt)
				plt.axis('scaled')
			else:
				# Set up the figure
				fig, axs = plt.subplots(1,len(bands), sharey=True, figsize=(20,8),)
				# +++++ Loop over the figure combo +++++
				for ax, bnds, in zip(axs, bands):
					# make the figure
					_subplotmaker(ax, bnds, bnds)
					ax.set_aspect('equal')

				# Get rid of the excess lats
				for ax in axs.flat:
					ax.label_outer()

				# ========== Change parms for the entire plot =========
				fig.suptitle("%s %s - Frame%03d" % (
					info.satellite, pd.Timestamp(info.date).strftime('%d-%m-%Y'), datelist.iloc[indx]["index"]))
				# ipdb.set_trace()
				plt.axis('scaled')
				# +++++ Make the images bigger by eleminating space +++++
				fig.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0, wspace=0, hspace=0) #top = 1, bottom = 1, right = 1, left = 1, 
				plt.tight_layout()
				plt.margins(0,0)


			return mplfig_to_npimage(fig)

		# ========== Eposrt the videos ==========
		mov = mpe.VideoClip(frame_maker, duration=int(datelist.shape[0]))
		# plays the clip (and its mask and sound) twice faster
		# newclip = clip.fl_time(lambda: 2*t, apply_to=['mask','audio'])

		# fnout = "%s/LANDSAT_5_7_8_%s_complete.txt" % (spath, coords["name"]) 
		print("Starting Write of the data at:", pd.Timestamp.now())
		try:
			mov.write_videofile(fnout, fps=1)
			return
		except Exception as ex:
			warn.warn(str(ex))
			print("Movie making failed. This will need to be redone")
			failed +=1
	raise ValueError


#==============================================================================
def SiteChecker(dpath, site, force, multi,
	program = "LANDSAT", dschoice = "SR", 
	dsinfom = "LANDSAT_5_7_8", dsbands = "SNRGB"):
	"""
	Function to check if a site needs to be done
	args:
		spath: 		str
		site:		str
		force:		bool
		program: 	str

	"""
	# THIS CODE I CAN USE IF THE BATCH EXPORT EVER WORKS
	# # =========== build a date check ==========
	# ymd = fnn.split("RGB_")[-1][:8] 
	# if not ( ymd  == date.strftime(format="%Y%m%d")):
	# 	print("date is missing")
	# 	warn.warn("date is missing")	

	spath = dpath + "UoL/FIREFLIES/VideoExports/"
	# ========== Check if the video has already been made ==========
	if not force:
		if multi:
			if os.path.isfile(spath+site+"/LANDSAT_5_7_8_%s_multi_NRG_RGB_SNR.mp4" % (site)):
				print("File already exists for %s, Going to next site. Use force to overwrite video" % site)
				return False
		else:
			ipdb.set_trace()
	# TO DO

	# ========== Look for the coords file ==========
	csv_cr = "%s%s/%s_%s_gridinfo.csv" % (spath, site, program, site)
	csv_nm = "%s%s/%s_%s_%s_timeinfo.csv" % (spath, site, dsinfom, site, dsbands)

	# ========== Check if the request has been sent to the cloud ==========
	if not os.path.isfile(csv_nm):
		print("%s has not been sent to the cloud. Going to next site")
		return False
	else:
		# ========== read the csv files ==========
		df_dates = pd.read_csv(csv_nm, index_col=0, parse_dates=True)
		df_cords = pd.read_csv(csv_cr)#, index_col=0, parse_dates=True)

	# ========== Check to see if all the files are in the right location ==========
	dfn_nm   = "%s%s/raw/%s_CompleteFileList.csv" % (spath, site, site)
	if not os.path.isfile(dfn_nm):
		try:
			df_names = filemover(dpath, spath, site, dsinfom, dsbands, df_dates, dfn_nm)
		except IOError:
			return False
	else:
		df_names = pd.read_csv(dfn_nm, index_col=0)
		if not (spath+site+"/raw/" ==   df_names.iloc[1].values[0].strip(df_names.iloc[1].values[0].split("/")[-1])):
			fnameslist = [spath+site+"/raw/"+gnm.split("/")[-1] for gnm in df_names.fnames.values]
			df_names["fnames"] = fnameslist


	return({"fnames":df_names, "dates":df_dates, "coords":df_cords})
	# ipdb.set_trace()
	# sys.exit()

#==============================================================================
def filemover(dpath, spath, site, dsinfom, dsbands, df, dfn_nm):
	""" Function to check if all the files i need are in the correct location"""

	# ========== Make the raw path and movepath ==========
	rpath   = dpath + "FIREFLIES_geotifs/"
	mpath   = "%s%s/raw/" % (spath, site)
	partial = False
	Wloops  = 0


	while Wloops<=1:
		Wloops += 1

		# ========== Find the files ==========
		fnames = sorted(glob.glob(rpath+'%s_%s_%s_*.tif' % (dsinfom, site, dsbands)))
		
		if os.path.isfile(mpath+"failed_geotifs.npy"):
			# ========== load the fails ==========
			fails = np.load(mpath+"failed_geotifs.npy")

			def filetester(fnn):
				da_in  = xr.open_rasterio(fnn).transpose("y", "x", "band").rename({"x":"longitude", "y":"latitude"})
				da_in == None

			if len(fnames) == fails.shape[0]:
				for fnn in fnames:
					filetester(fnn)
				print( "All of the rebuilt file appear to be valid")

				# ========== Move the files ==========
				print("Starting %s file relocation at:" % site, pd.Timestamp.now())
				for fx in fnames:
					shutil.move(fx, mpath)
				
				# ========== Store the file names ==========
				mfnames  = sorted(glob.glob(mpath+'%s_%s_%s_*.tif' % (dsinfom, site, dsbands)))

				df_names = pd.DataFrame({"fnames":mfnames})
				if df_names.shape[0] == df.shape[0]:
					df_names.to_csv(dfn_nm)
				else:
					warn.warn("something has gone wrong here")
					ipdb.set_trace()

				# ========== Sleep to allow files to move ==========
				print("Waiting so files have a chance to move. Wait started at:", pd.Timestamp.now())
				time.sleep(120)
				return df_names


		# ========== Check to see if they have all downloaded ==========
		if len(fnames) == df.shape[0]:
			# ========== Make the path ==========
			cf.pymkdir(mpath)
			
			# ========== Move the files ==========
			print("Starting %s file relocation at:" % site, pd.Timestamp.now())
			for fx in fnames:
				shutil.move(fx, mpath)
			
			# ========== Store the file names ==========
			mfnames  = sorted(glob.glob(mpath+'%s_%s_%s_*.tif' % (dsinfom, site, dsbands)))

			df_names = pd.DataFrame({"fnames":mfnames})
			if df_names.shape[0] == df.shape[0]:
				df_names.to_csv(dfn_nm)
			else:
				warn.warn("something has gone wrong here")
				ipdb.set_trace()

			# ========== Sleep to allow files to move ==========
			print("Waiting so files have a chance to move. Wait started at:", pd.Timestamp.now())
			time.sleep(60)
			return df_names
		elif len(fnames) == 0:
			print(site, " is waiting for files to download. No files currently in folder")
			raise IOError
		else:
			# ========== Look for failures ==========
			for nnx in range(0, df.shape[0]):
				fnstest = sorted(glob.glob(rpath+'%s_%s_%s_%04d*' % (dsinfom, site, dsbands, nnx)))
				if len(fnstest)>1:
					warn.warn("Duplicate Files exist at: %d" % nnx)
					# delete duplicates of the form (1), (2) etc
					for fnt in fnstest:
						if " (" in fnt:
							os.remove(fnt)
							print("Excess file deleted at: %d" % nnx)

				elif len(fnstest) == 0:
					print(nnx)
					warn.warn("Missing file: %d" % nnx)
					ipdb.set_trace()
	print("WHile loops exceeded")
	ipdb.set_trace()
	sys.exit()
#==============================================================================
def _bandcom(fn):
	""" Returns a dict of infomation about the geotifs"""
	bandcombo = OrderedDict()
	da_ts  = xr.open_rasterio(fn).transpose("y", "x", "band").rename({"x":"longitude", "y":"latitude"})

	if da_ts.shape[2] == 3:
		bandcombo["RGB"] = [1, 2, 3]	
		bandlist = None
	elif da_ts.shape[2] == 4:
		bandcombo["NRG"] = [1, 2, 3]
		bandcombo["RGB"] = [2, 3, 4]
		bandlist = None
	elif da_ts.shape[2] == 7:
		# bandlist = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa']
		bandlist = ['pixel_qa', 'SWIR2','SWIR1', 'NIR', 'R', 'G', 'B']

		bandcombo["SNR"] = ['SWIR1', 'NIR', 'R']
		bandcombo["NRG"] = ['NIR'  , 'R'  , 'G']
		bandcombo["RGB"] = [  'R'  , 'G'  , 'B']

	else:
		warn.warn("unknown band structure")
		ipdb.set_trace()
	return bandcombo, bandlist

def syspath():
	""" Gets the system path """
	# ========== Create the system specific paths ==========
	# ========== Create the system specific paths ==========
	sysname = os.uname()[1]
	if sysname == 'DESKTOP-CSHARFM':
		# LAPTOP
		dpath = "/mnt/c/Users/arden/Google Drive/"

	elif sysname == "owner":
		dpath = "/mnt/c/Users/user/Google Drive/"
	else:
		warn.warn("Paths not created for this computer")
		# spath =  "/media/ubuntu/Seagate Backup Plus Drive"
		ipdb.set_trace()
	return dpath

#==============================================================================
if __name__ == '__main__':
	# ========== Set the args Description ==========
	description='Script to make movies'
	parser = argparse.ArgumentParser(description=description)
	
	# ========== Add additional arguments ==========
	parser.add_argument(
		"-s", "--site", type=str, default=None, help="Site to work with ")
	# parser.add_argument(
	# 	"--gparts", type=int, default=None,   
	# 	help="the max partnumber that has not been redone")
	parser.add_argument(
		"-f", "--force", action="store_true",
		help="the max partnumber that has not been redone")
	parser.add_argument(
		"-v", "--verbose", action="store_true",
		help="make a video with all framse, will increase video length")
	args = parser.parse_args() 
	
	# ========== Call the main function ==========
	main(args)