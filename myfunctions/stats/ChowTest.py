import numpy as np
from scipy.stats import f
import pandas as pd 
import xarray
import ipdb
import warnings as warn
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _split(da, date, period="Y"):
	"""
	args:
		da: xarray data array
			the dataarray containing the values with a time index
		date: pandas datetime
			date of the breakpoint
		period: datetime frequency code
			shift till after the breakpoint, based on temporal frequency of data
	Returns
		y1: Array like y-values for data preceeding the breakpoint
		x1: Array like x-values for data preceeding the breakpoint
		y2: Array like y-values for data occuring after the breakpoint
		x2: Array like x-values for data occuring after the breakpoint
	"""
	# ========== Split the time series ==========
	y1 = da.sel(time=slice(None, date)).values
	y2 = da.sel(time=slice(
		date,None)).values

	x1 = da.sel(time=slice(None, date)).time.values.astype('datetime64[%s]'% period).astype(float)
	x2 = da.sel(time=slice(date,None)).time.values.astype('datetime64[%s]'% period).astype(float)

	return y1, y2, x1, x2
	

def _f_value(da, date, period):
	"""This is the f_value function for the Chow Break test package
	Args:
		da: xarray data array
			the dataarray containing the values with a time index
		date: pandas datetime
			date of the breakpoint
		period: datetime frequency code
			shift till after the breakpoint, based on temporal frequency of data

	Returns:
		F-value: Float value of chow break test
	"""
	def _find_rss (y, x):
		"""This is the subfunction to find the residual sum of squares for a given set of data
		Args:
			y: Array like y-values for data subset
			x: Array like x-values for data subset

		Returns:
			rss: Returns residual sum of squares of the linear equation represented by that data
			length: The number of n terms that the data represents
		"""
		# ========== Old versions of this approach ==========
		# X = sm.add_constant(x)#
		# A = np.vstack([x, np.ones(len(x))]).T
		# rss = np.linalg.lstsq(A, y, rcond=None)[1]
		# df = pd.DataFrame({"y":y, "x":x})
		# results = sm.OLS(y,sm.add_constant(np.arange(x.shape[0]))).fit()

		# ========== Current version ==========
		# FUTURE, use smf to allow for multivariate approaches 
		results     = sm.OLS(y,sm.add_constant(x)).fit()
		rss     = results.ssr 
		length  = len(y)
		return (rss, length)

	# ===== Split the series into sub sections =====
	y1, y2, x1, x2 = _split(da, date, period)
	# ipdb.set_trace()

	# ===== get the rss =====
	rss_total, n_total = _find_rss(
		da.values, 		da.time.values.astype('datetime64[%s]'% period).astype(float))
	rss_1, n_1 = _find_rss(y1, x1)
	rss_2, n_2 = _find_rss(y2, x2)
	

	chow_nom = (rss_total - (rss_1 + rss_2)) / 2
	chow_denom = (rss_1 + rss_2) / (n_1 + n_2 - 4)
	return chow_nom / chow_denom


def _p_value(da, date, period, **kwargs):
	"""
	Calculates the p value for the results
	args:
		da: xarray data array
			the dataarray containing the values with a time index
		date: pandas datetime
			date of the breakpoint
		period: datetime offest
			shift till after the breakpoint, based on temporal frequency of data
	returns:
		F-value: Float value of chow break test
		P-value: Float value of chow break test
	"""

	F = _f_value(da, date, period, **kwargs)
	if not F:
		return 1
	
	df1 = 2
	df2 = da.size - 4
	# The survival function (1-cdf) is more precise than using 1-cdf,
	# this helps when p-values are very close to zero.
	# -f.logsf would be another alternative to directly get -log(pval) instead.
	p_val = f.sf(F, df1, df2)
	return F, p_val


def ChowTest(ts, date, period="Y", **kwargs):
	"""
	Chow Test
	args:
		ts: pandas time series or x array dataseties
			the complete unsegmented
		date: object that can be cohesed into pd.datetime
			The date of the suspected breakpoint
	returns:
		fval: pvalue of the chow test
		pval: fvalue of the chow test

	"""

	warn.warn("ChowTest has not been implemented yet.  ")
	warn.warn("Data checks are still under development")
	# ========== Check the input varialbes ==========
	# Breakpoint date
	if type(date) == dt.datetime:
		date = pd.to_datetime(date)
	elif type(date) == pd._libs.tslibs.timestamps.Timestamp:
		pass
	else:
		warn.warn("Data checks are still under development")

	# TimeSeries	
	if type(ts) == pd.core.series.Series:
		ts.index = pd.to_datetime(ts.index)
		ts.index.name = "time"
		da = ts.to_xarray()

	fval, pval = _p_value(da, date, period, **kwargs)

	return {"F_value":fval, "p_value":pval}