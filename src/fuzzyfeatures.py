from __future__ import unicode_literals
from fuzzywuzzy import fuzz, process
import numpy as np
import pandas as pd
def qratio(a,b):
	try:
		return fuzz.QRatio(a,b)
	except:
		return np.NaN

def sratio(a,b):
	try:
		return fuzz.ratio(a,b)
	except:
		return np.NaN

def pratio(a,b):

	try:
		return fuzz.partial_ratio(a,b)
	except:
		return np.NaN

def tokensort(a,b):
	try:
		return fuzz.token_sort_ratio(a,b)
	except:
		return np.NaN

def tokenset(a,b):
	try:
		return fuzz.token_set_ratio(a,b)
	except:
		return np.NaN
