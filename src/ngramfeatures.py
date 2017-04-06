#!/Users/follin/Documents/projects/kaggle/StackExchangeTransferLearn/venv/bin/python
from __future__ import unicode_literals
import numpy as np
from repoze.lru import lru_cache
from nltk import word_tokenize, ngrams
from exceptions import *
from collections import Counter
from chardet import detect





def common_weighted_fraction(x,y, tfidf = None):
	if tfidf is None:
		tfidf = TFIDF(stops, n =1, eps = 10, min_count = 1)
	common_words = common_ngrams(x,y)
	numerator = 2 * np.sum([tfidf(w) for w in common_words])
	denominator = np.sum([tfidf(w) for w in x]) + np.sum([tfidf(w) for w in y]) + 0.0

	return numerator/denominator


def get_ngrams(x, n = 1):
	try:
		return [i for i in ngrams(x, n)]
	except TypeError:
		return []
	except:
		return []

def tokenize(x, n = 1):
	try:
		return [word for word in word_tokenize(x.lower())]
	except AttributeError:
		return []
	except UnicodeDecodeError:
		try:
			return [word for word in word_tokenize(unicode(x, encoding = 'utf-8'))]
		except:
			encoding = decect(x)['encoding']
			print 'Autodetected encoding {} at {} \% confidence'.format(encoding, detect(x)['confidence'])
			print r'\t ...'+x
			return [word for word in word_tokenize(unicode(x, encoding = encoding))]


### String methods





def common_ngrams(x,y):
	try:
		set1 = set(x)
	except:
		return np.NaN
	try:
		set2 = set(y)
	except:
		return np.NaN
	return set1.intersection(set2)




def questionlength(string_array, unique = False):
	if isinstance(string_array, list):
		if unique:
			out = len(set(string_array))
		else:
			out = len(string_array)
		return out
	else:
		return 0




### Helper Functions
#stops = [stopwords.words('english')]

