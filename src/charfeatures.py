from __future__ import unicode_literals
import numpy as np
from repoze.lru import lru_cache
from nltk import word_tokenize, ngrams
from exceptions import *
import difflib

def qlength(sentence):
	if isinstance(sentence, unicode):
		return len(sentence)
	if isinstance(sentence, list):
		try:
			return len(''.join(sentence))
		except:
			return np.NaN
	else:
		return np.NaN

def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()