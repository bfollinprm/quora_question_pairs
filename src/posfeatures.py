from __future__ import unicode_literals
from nltk import pos_tag, word_tokenize
import nltk
import numpy as np
from exceptions import *
from repoze.lru import lru_cache
from scipy.spatial import distance
from chardet import detect
import cPickle as pickle



class Interrogatives(object):
	def __init__(self, file = None):
		interrogatives = [
			'which', 'what', 'whose',  ### determiners
			'who', 'whom','whose', ##personal pronoun
			'what', 'which', # impersonal pronouns
			'where', #location
			'when', #time
			'how', #manner
			'why', #reason
			'whether', #choice between
			'does','has','can','shall',
			'is','was',
			'should'
			] 
		if file is not None:
			try:
				self.interrogatives = pickle.load(open(file, 'rb'))
			except:
				self.interrogatives = dict(zip(interrogatives, np.arange(len(interrogatives))))
		else:
			self.interrogatives = dict(zip(interrogatives, np.arange(len(interrogatives))))
		if file is not None:
			pickle.dump(self.interrogatives, open(file, 'wb'))

	def __call__(self, question):
		self.v = None
		try:
			[self._upcount(word) for word in word_tokenize(question)]
		except:
			return np.zeros(20)
		return self.v
	def _upcount(self, word):
		i = self.interrogatives.get(word.lower(), -1)
		if self.v is None:
			self.v = np.zeros(20)
		else:
			pass
		if i > 0:
			self.v[i] += 1
		return self.v

def interrogatives(q, vectorizer = None):
	return vectorizer(q)

def difference(x,y):
	return x-y


def cosine(x, y):
	try:
		return distance.cosine(x,y)
	except ValueError:
		return np.NaN
	except:
		return np.NaN

def braycurtis(x,y):
	try:
		return distance.braycurtis(x,y)
	except ValueError:
		return np.NaN
	except:
		return np.NaN

def euclidean(x,y):
	try:
		return distance.euclidean(x,y)
	except ValueError:
		return np.NaN
	except:
		return np.NaN

def cityblock(x,y):
	try:
		return distance.cityblock(x,y)
	except ValueError:
		return np.NaN
	except:
		return np.NaN

def minkowski(x,y, p = 3):
	try:
		return distance.minkowski(x,y, p)
	except ValueError:
		return np.NaN
	except:
		return np.NaN

def correlation(x,y):
	try:
		return distance.correlation(x,y)
	except ValueError:
		return np.NaN
	except:
		return np.NaN



class POS_Tagger(object):
	def __init__(self, corpus = None, file = None):

		if file is not None:
			try:
				self.pos_hash = pickle.load(open(file, 'rb'))
			except:
				corpus = [s for s in corpus if isinstance(s, unicode)]
				self.pos_hash = {k: v for d in 
						[{word: tag[0] for (word, tag) in get_tags(s) if tag[0] in ['N','V']} for s in corpus] 
												for k,v in d.items()}

		else:
			corpus = [s for s in corpus if isinstance(s, unicode)]
			self.pos_hash = {k: v for d in 
					[{word: tag[0] for (word, tag) in get_tags(s) if tag[0] in ['N','V']} for s in corpus] 
											for k,v in d.items()}
		if file is not None:
			pickle.dump(self.pos_hash, open(file, 'wb'))

	def __call__(self, word):
		tag = self.pos_hash.get(word, ' ')
		return tag

def tokenize(x):
	try:
		return [word for word in word_tokenize(x)]
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
def get_tags(sentence):
	if isinstance(sentence, list):
		return pos_tag(sentence)
	elif isinstance(sentence, unicode):
		return pos_tag(tokenize(sentence))
	else:
		return sentence


def unique_nums(sentence):
	try:
		if isinstance(sentence, list):
			return (set([word for word in sentence if word.isdigit()]))
		elif isinstance(sentence, unicode):
			return (set([word for word in tokenize(sentence) if word.isdigit()]))
		else:
			raise InputError('sentence', 'is not a string or list of strings')
	except InputError:
		return np.NaN

def num_common_unique_nums(sentence1, sentence2):
	try:
		set1 = unique_nums(sentence1)
		set2 = unique_nums(sentence2)
		return len(set1.intersection(set2))
	except:
		return np.NaN

def num_common_unique_nums_weighted(sentence1, sentence2, tfidf = None):
	try:
		set1 = unique_nums(sentence1)
		set2 = unique_nums(sentence2)
		return sum([tfidf(x) for x in set1.intersection(set2)])
	except:
		return np.NaN



def unique_caps(sentence):
	try:
		if isinstance(sentence, list):
			return (set([word for word in sentence if word[0].isupper()]))
		elif isinstance(sentence, unicode):
			return (set([word for word in tokenize(sentence) if word[0].isupper()]))
		else:

			raise InputError('sentence', 'is not a string or list of strings')
	except InputError:
		return np.NaN

def num_common_unique_caps(sentence1, sentence2):
	try:
		set1 = unique_caps(sentence1)
		set2 = unique_caps(sentence2)
		return len(set1.intersection(set2))
	except:
		return np.NaN

def num_common_unique_caps_weighted(sentence1, sentence2, tfidf = None):
	try:
		set1 = unique_caps(sentence1)
		set2 = unique_caps(sentence2)
		return sum([tfidf(x) for x in set1.intersection(set2)])
	except:
		return np.NaN


def unique_nouns(sentence, tagger = None):
	try:
		if isinstance(sentence, list):
			return set([word for word in sentence if tagger(word) == 'N'])
		elif isinstance(sentence, unicode):
			return set([word for word in tokenize(sentence) if tagger(word) == 'N'])
		else:
			raise InputError('sentence', 'is not a string or list of strings')
	except InputError:
		return np.NaN


def unique_verbs(sentence, tagger = None):
	try:
		if isinstance(sentence, list):
			return set([word for word in sentence if tagger(word) == 'V'])
		elif isinstance(sentence, unicode):
			return set([word for word in tokenize(sentence) if tagger(word) == 'V'])
		else:
			raise InputError('sentence', 'is not a string or list of strings')
	except InputError:
		return np.NaN

def num_common_unique_nouns(sentence1, sentence2, tagger = None):
	try:
		set1 = unique_nouns(sentence1, tagger = tagger)
		set2 = unique_nouns(sentence2, tagger = tagger)
		return len(set1.intersection(set2))
	except:
		return np.NaN

def num_common_unique_nouns_weighted(sentence1, sentence2, tfidf = None, tagger = None):
	try:
		set1 = unique_nouns(sentence1, tagger = tagger)
		set2 = unique_nouns(sentence2, tagger = tagger)
		return sum([tfidf(x) for x in set1.intersection(set2)])
	except:
		return np.NaN	

def num_common_unique_verbs(sentence1, sentence2, tagger = None):
	try:
		set1 = unique_verbs(sentence1, tagger = tagger)
		set2 = unique_verbs(sentence2, tagger = tagger)
		return len(set1.intersection(set2))
	except:
		return np.NaN

def num_common_unique_verbs_weighted(sentence1, sentence2, tfidf = None, tagger = None):
	try:
		set1 = unique_verbs(sentence1, tagger = tagger)
		set2 = unique_verbs(sentence2, tagger = tagger)
		return sum([tfidf(x) for x in set1.intersection(set2)])
	except:
		return np.NaN	
