from __future__ import unicode_literals
import numpy as np
import pandas as pd
import os
import os.path as osp
from exceptions import *
from numpy import zeros, array, sqrt, dot
from repoze.lru import lru_cache
from scipy.spatial import distance #cosine, braycurtis, euclidean, cityblock, minkowski, correlation
from gensim.models import  Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
from scipy.misc import imresize
#from numpy.linalg import dot

# class QuoraWord2Vec(object):
# 	def __init__(self, binaryfile = None, corpus = None):
# 		if binaryfile is None:
# 			try: 
# 				parentdir = osp.join(osp.abspath(osp.join(os.getcwd(), os.pardir)), 'QuoraQuestionPairs')
# 				self.model = Word2Vec.load(osp.join(parentdir, 'data','QuoraQuestions.bin'))
# 			except:
# 				try:
# 					print 'building word2vecs from Quora Questions'
# 					corpus = [s.split() for s in corpus if isinstance(s, unicode)]
# 					print corpus
# 					self.model = Word2Vec(corpus, size = 300, window = 5, min_count = 1, workers = 4, iter = 10)
# 					self.model.train(corpus)
# 					parentdir = osp.join(osp.abspath(osp.join(os.getcwd(), os.pardir)), 'QuoraQuestionPairs')
# 					self.model.save(osp.join(parentdir, 'data','QuoraQuestions.bin'))
# 				except ImportError:
# 					raise ImportError('model', 'no word2vec file found')
# 		elif isinstance(binaryfile, str):
# 			try:
# 				self.model = Word2Vec.load(osp.join(binaryfile))
# 			except:

# 				raise ImportError('model', 'no word2vec file found @ %s'%binaryfile)
# 		self.word_vectors = self.model.wv
# 	def __call__(self, word):
# 		try:
# 			vec = array(self.word_vectors[word])

# 			return vec/sqrt(dot(vec, vec))
# 		except:
# 			#print 'word {} not in vocabulary, returning null vector'.format(word)
# 			return zeros(300)

# class GoogleWord2Vec(object):
# 	def __init__(self, binaryfile = None):
# 		if binaryfile is None:
# 			parentdir = osp.join(osp.abspath(osp.join(os.getcwd(), os.pardir)), 'QuoraQuestionPairs')
# 			try:
# 				self.word_vectors = KeyedVectors.load_word2vec_format(osp.join(parentdir, 'data','GoogleNews.bin'), binary = True)
# 			except:
# 				raise ImportError('model', 'no word2vec file found @ %s'%osp.join(parentdir, 'data','GoogleNews.bin'))
# 		else:
# 			try:
# 				self.word_vectors = KeyedVectors.load_word2vec_format(binaryfile, binary = True)
# 			except:
# 				raise ImportError('model', 'no word2vec file found @ %s'%binaryfile)

# 	def __call__(self, word):
# 		try:
# 			vec = array(self.word_vectors[word])
# 			return vec/sqrt(dot(vec, vec))
# 		except:
# 			# word not in vocabulary, returning null vector
# 			return zeros(300)



def vectorize(x, vectorizer = None):
    if vectorizer is None:
        vectorizer  = GoogleWord2Vec()
    if isinstance(x, list):
        try:
            return array([vectorizer(el) for el in x])
        except:
            return array([zeros(300) for el in x])
    elif isinstance(x, float):
    	return zeros(300)
    elif isinstance(x, unicode):
        x = x.split()
        try:
            return array([vectorizer(el) for el in x])
        except:
            return array([zeros(300) for el in x])
    else:
        return zeros(300)


def wmdistance(x, vectorizer = None):
    if vectorizer is None:
        vectorizer  = GoogleWord2Vec()
   
   	v1 = x.q1_nostops
   	v2 = x.q2_nostops
   	return vectorizer.wmdistance(v1, v2)


def sumvector(x):
	try:
		y = np.sum(x, axis = 0)
		if y.size != 300:
			if x.size == 300:
				return x
			else:
				return zeros(300)
		else:
			return y
	except:
		return zeros(300)

def meanvector(x):
	try:
		y = np.mean(x, axis = 0)
		if y.size != 300:
			if x.size == 300:
				return x
			else:
				return zeros(300)
		else:
			return y
	except:
		return zeros(300)

def stdvector(x):
	try:
		y = np.std(x, axis = 0)
		if y.size != 300:
			return zeros(300)
		else:
			return y
	except:
		return zeros(300)


def difference(x,y):
	try: 
		return x-y
	except:
		return np.zeros(300)

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

def similarity(x,y):
	try:
		out = distance.cosine(x,y)
		if out < 100:
			return out
		else:
			return 0
	except:
		return 0

def matrix_similarity(q1vecs, q2vecs):
	#try:
		matrix = np.array([similarity(x,y) for x in q1vecs for y in q2vecs])
		matrix = np.resize(matrix, 1000)
		return matrix
	#except:
	# 	return np.NaN

