from local_exceptions import InputError
import numpy as np
import pandas as pd
from feature_utilities import *

class FeatureSet(object):
	def __init__(self, container):
		from data import DataContainer
		if isinstance(container, DataContainer):
			self.q1 = container.question1
			self.q2 = container.question2
			self.feat = container.features
		else:
			raise InputError('{}'.format(container), 'object is not a Data Container')


class CharFeatures(FeatureSet):
	def __init__(self, container, prefix = ''):
		'''
		Initializes a character features instance. Options are:

		prefix:		Something to prefix the feature names with, if you run multiple instances of this 
					class on your data.	
		'''
		super(CharFeatures, self).__init__(container)
		self.prefix = prefix
	def __call__(self):
		'''
		returns character features. They are:
		1. string length of question 1 (without spaces)
		2. string length of quesetion 2 (without spaces)
		3. length difference between the two strings
		4. Whether the questions are exactly equivalent after removing whitespace and lowercasting
		5. total length of common string of q1 and q2
		6. ratio of similar to dissimilar portions of the string.
		'''
		self.feat[self.prefix + 'len_q1'] = self.q1.apply(stringlength)
		self.feat[self.prefix + 'len_q2'] = self.q2.apply(stringlength)
		self.feat[self.prefix + 'len_diff'] = (
			self.feat[self.prefix + 'len_q1'] 
			- self.feat[self.prefix + 'len_q2']
			).apply(abs)
		self.feat[self.prefix+'are_equiv'] = (
			self.q1.apply(lambda x: x.replace(' ', '').lower() if isinstance(x, unicode) else '')
			== self.q2.apply(lambda x: x.replace(' ', '').lower() if isinstance(x, unicode) else '')
			)
		questiontuples = zip(
			self.q1.apply(lambda x: x.replace(' ', '').lower() if isinstance(x, unicode) else ''), 
			self.q2.apply(lambda x: x.replace(' ', '').lower() if isinstance(x, unicode) else '')
			)
		self.feat[self.prefix+'match_len'] = np.array([common_string_length(x,y) for x,y in questiontuples])
		self.feat[self.prefix+'char_match_ratio'] = np.array(
			[diff_ratio(x,y) for x,y in zip(self.q1.values, self.q2.values)]
			)
		return self.feat


class NGramFeatures(FeatureSet):
	def __init__(self, container, idf_encoder = None, prefix = '', tokenizer = None):
		'''
		Initializes an ngram features instance. Options are:

		idf_encoder:	IDFWeights method that computes ngram weights. If none, builds one from 
						questions in DataContainer. If False, weights everything equally. MAKE SURE 
						VOCAB MATCHES WITH TOKENIZER OUTPUT!

		tokenizer: preprocessers.Tokenize() instance to break up corpus. If none, default to 1-gram word tokenization. 

		prefix:		Something to prefix the feature names with, if you run multiple instances of this 
					class on your data.
		'''
		from encoders import IDFWeights
		from preprocessers import Tokenize
		if tokenizer is None:
			self.tokenizer = Tokenize()
		elif isinstance(tokenizer, Tokenize):
			self.tokenizer = tokenizer
		else:
			raise ImputError('','')


		self.weight_words = True
		super(NGramFeatures, self).__init__(container)
		self.prefix = prefix
		if idf_encoder is None:
			self.idf_encoder = IDFWeights(corpus = self.q1.tolist() + self.q2.tolist(), n = 1)
		elif isinstance(idf_encoder, IDFWeights):
			self.idf_encoder = idf_encoder
		elif not idf_encoder:
			self.weight_words = False
		else:
			raise InputError({}.format(idf_encoder), 'object is not an IDF encoder')
		self.q1 = self.q1.apply(lambda x: tokenizer(x) if isinstance(x, unicode) else '').values
		self.q2 = self.q2.apply(lambda x: tokenizer(x) if isinstance(x, unicode) else '').values

	def __call__(self):
		'''
		returns Ngram features. They are:
		1. Number of ngrams in question 1
		2. number of ngrams in question 2
		3. number of unique ngrams in question 1
		4. number of unique ngrams in question 2
		5. number of unique common ngrams between Q1 and Q2
		6. ratio of common to total unique ngrams
		'''
		### ngram counts
		if self.weight_words:
			q1word_count = [sum([self.idf_encoder(w) for w in x]) for x in self.q1]
			q2word_count = [sum([self.idf_encoder(w) for w in x]) for x in self.q2]
			q1unique_count = [sum(self.idf_encoder(w) for w in set(x)) for x in self.q1]
			q2unique_count = [sum(self.idf_encoder(w) for w in set(x)) for x in self.q2]

			# q1unique_count = [sum(set(self.idf_encoder(w) for w in x)) for x in self.q1]
			# q2unique_count = [sum(set(self.idf_encoder(w) for w in x)) for x in self.q2]
			common_words_count = [sum(self.idf_encoder(word) 
				for word in common_ngrams(x,y)) for x,y in zip(self.q1, self.q2)]
		else:
			q1word_count = [len([w for w in x]) for x in self.q1]
			q2word_count = [len([w for w in x]) for x in self.q2]
			q1unique_count = [len(set(w for w in x)) for x in self.q1]
			q2unique_count = [len(set(w for w in x)) for x in self.q2]
			common_words_count = [len([w for w in common_ngrams(x,y)]) for x,y in zip(self.q1, self.q2)]


		self.feat[self.prefix + 'num{}gramsQ1'.format(self.n)] = q1word_count
		self.feat[self.prefix + 'num{}gramsQ2'.format(self.n)] = q2word_count
		self.feat[self.prefix + 'num_unique{}gramsQ1'.format(self.n)] = q1unique_count
		self.feat[self.prefix + 'num_unique{}gramsQ2'.format(self.n)] = q2unique_count
		self.feat[self.prefix + 'common{}grams'.format(self.n)] = common_words_count
		self.feat[self.prefix + 'common{}grams_ratio'.format(self.n)] = 2*np.array(common_words_count)/(
			np.array(q1unique_count) + np.array(q2unique_count)
			)


		return self.feat



class FuzzyFeatures(FeatureSet):
	def __init__(self, container, prefix = ''):
		'''
		Initializes a feature set based on the Levenstein (fuzzy) distance between
		two strings. Options:

		prefix:		Something to prefix the feature names with, if you run multiple instances of this 
					class on your data.

		'''
		super(FuzzyFeatures, self).__init__(container)
		self.q1 = self.q1.apply(lambda x: x if isinstance(x, unicode) else '')
		self.q2 = self.q2.apply(lambda x: x if isinstance(x, unicode) else '')

		self.prefix = prefix
	def __call__(self):
		from fuzzywuzzy import fuzz

		'''
		returns fuzzy logic features. They are:
		1. Qratio similarity
		2. levenstein similarity ratio (sratio)
		3. partial ratio
		4. token sort ratio
		5. token set ratio
		'''
		self.feat[self.prefix + 'QRatio'] = [fuzz.QRatio(x,y) for x,y in zip(self.q1, self.q2)]
		self.feat[self.prefix + 'SRatio'] = [fuzz.ratio(x,y) for x,y in zip(self.q1, self.q2)]
		self.feat[self.prefix + 'PRatio'] = [fuzz.partial_ratio(x,y) for x,y in zip(self.q1, self.q2)]
		self.feat[self.prefix + 'TSort'] = [fuzz.token_sort_ratio(x,y) for x,y in zip(self.q1, self.q2)]
		self.feat[self.prefix + 'TSet'] = [fuzz.token_set_ratio(x,y) for x,y in zip(self.q1, self.q2)]
		return self.feat


class VectorFeatures(FeatureSet):
	'''
	Initializes a vector feature constructor that computes features based on vector representations
	of documents. Options:

	vectorizer: what vector representation to use. Implented classes (in src.encoders) include 
				Doc2Vec, Part of Speech bagging, and TF-IDF vectorization.

	size: 	Number of dimensions to include in the feature set. If size < the output of a vectorizer,
			a truncated SVD transform is implemented that reduces the size of the vector space 
			before output

	corpus: if size is not None, the corpus used to understand the covariance structure for SVD 
			transformation. If corpus is None, corpus is set to be the questions in the data container.

	prefix: Something to prefix the feature names with, if you run multiple instances of this 
			class on your data.

	n_batches:	For large datasets, vectorization can lead to memory issues. n_batches determines the 
				number of jobs to batch the vectorization into to spare your RAM.

	'''
	def __init__(self, container, vectorizer, corpus = None, prefix = '', size = None, n_batches = 10):
		super(VectorFeatures, self).__init__(container)
		from encoders import Encoder
		if not isinstance(vectorizer, Encoder):
			raise InputError('{}'.format(vectorizer), 'Not an encoder instance')
		self.vectorizer = vectorizer
		self.prefix = prefix
		self.size = size
		self.n_batches = n_batches

		if corpus is None:
			self.corpus = container.question1.tolist() + container.question2.tolist()
		else:
			self.corpus = corpus

		if (self.size is not None) & (self.size < self.vectorizer.size):
			from sklearn.decomposition import TruncatedSVD
			self.transformer = TruncatedSVD(self.size)
			mat = self._get_covmat()
			self.transformer.fit(mat)
		else:
			self.size = self.vectorizer.size
			self.transformer = None


	def __call__(self):
		'''
		returns vector features. They are:
		1. cosine similarity
		2. L2 normed distance
		3. L1 normed distance
		4. Bray-Curtis distance
		5. Correlation distance
		6. absolute distance vector between q1 and q2 (ndim features)
		'''
		from scipy.spatial import distance
		self.q1s = np.array_split(self.q1.values, self.n_batches)
		self.q2s = np.array_split(self.q2.values, self.n_batches)
		indices = np.array_split(self.feat.index, self.n_batches)
		dfs = []
		for index, (q1, q2) in zip(indices, zip(self.q1s, self.q2s)):
			df = pd.DataFrame(index = index.values)
			pre = self.prefix
			vec = self.vectorizer
			df[pre + 'cos_dist'] = [distance.cosine(vec(x),vec(y)) for x,y in zip(q1,q2)]
			df[pre + 'euc_dist'] = [distance.euclidean(vec(x),vec(y)) for x,y in zip(q1,q2)]
			df[pre + 'manhattan_dist'] = [distance.cityblock(vec(x),vec(y)) for x,y in zip(q1,q2)]
			df[pre + 'braycurt_dist'] = [distance.braycurtis(vec(x),vec(y)) for x,y in zip(q1,q2)]
			df[pre + 'correlation_dist'] = [distance.correlation(vec(x),vec(y)) for x,y in zip(q1,q2)]
			for i in range(self.size):
				if self.transformer is not None:
					df[pre + 'vec_{}'.format(i)] = abs(
					self.transformer.transform(np.array([vec(x) for x in q1]))[:,i] 
					-self.transformer.transform(np.array([vec(x) for x in q2]))[:,i]
					)
				else:
					df[pre + 'vec_{}'.format(i)] = abs(
					(np.array([vec(x) for x in q1]))
					-(np.array([vec(x) for x in q2]))
					)[:,i]


			dfs +=[df]
		df = pd.concat(dfs)
		return self.feat.join(df)

	def _get_covmat(self):
		dim = self.vectorizer.size
		covmat = np.zeros([dim, dim])
		blocks = np.array_split(range(dim), int(np.sqrt(self.n_batches)+1))
		for i,b1 in enumerate(blocks):
			var1 = np.array([self.vectorizer(x)[b1] for x in self.corpus])
			var1 -= np.mean(var1, axis = 0)
			for j, b2 in enumerate(blocks):
				if j >= i:
					var2 = np.array([self.vectorizer(x)[b2] for x in self.corpus])
					var2 -= np.mean(var2, axis = 0)
					submat = np.cov(var1, var2, rowvar = False)[:len(b1), len(b1):]
					covmat[min(b1):max(b1)+1, min(b2):max(b2)+1] = submat
					covmat[min(b2):max(b2)+1, min(b1):max(b1)+1] = submat.T
		return covmat


class MatrixFeatures(FeatureSet):
	'''
	Initializes matrix feature constructor that computes features based on vector representation of words.
	Options are:
	'''
	def __init__(self, container, word_vectorizer, question_length = 20, prefix = '', n_batches = 10):
		super(MatrixFeatures, self).__init__(container)
		from scipy.ndimage import zoom
		self.vectorizer = word_vectorizer
		self.question_length = question_length
		self.prefix = prefix
		self.n_batches = n_batches
		if not isinstance(vectorizer, TokenEncoder):
			raise InputError('{}'.format(vectorizer), 'Not a word encoder instance')

	def __call__(self):
		self.q1 = np.array_split(self.q1.values, self.n_batches)
		self.q2 = np.array_split(self.q2.values, self.n_batches)
		indices = np.array_split(self.feat.index, self.n_batches)
		dfs = []
		for index, (q1, q2) in zip(indices, zip(self.q1, self.q2)):
			df = pd.DataFrame(index = index.values)
			pre = self.prefix
			vec = self.vectorizer
			q1 = [[vec(w) for w in ngrams(word_tokenize(s), vec.n)] for s in q1]
			q2 = [[vec(w) for w in ngrams(word_tokenize(s), vec.n)] for s in q2]
			matrix = [
				array([vec(a) - vec(b) for a in ngrams(word_tokenize(s1), vec.n) for b in ngrams(word_tokenize(s2), vec.n)]).reshape(
					len(ngrams(word_tokenize(s1), vec.n)), len(ngrams(word_tokenize(s1), vec.n))
				)
			]
			matrix = [zoom(a, zoom = 1.0 * self.question_length/np.array([a.shape[0], a.shape[1]])) for a in matrix]
			for i in arange(self.question_length):
				for j in arange(self.question_length):
					df[pre + 'diff_mat{}{}'.format(i,j)] = [a[i,j] for a in matrix]
			dfs += [df]
		df = pd.concat(dfs)
		return self.feat.join(df)


### Unit tests
if __name__ == '__main__':
	from data import DataContainer
	from encoders import Doc2Vec
	container = DataContainer('../data/train.csv', size = 1000)
	container.clean_questions()
	# extractor = CharFeatures(container)
	# feats = extractor()
	# print feats.columns
	# print feats.sample(1)
	extractor = NGramFeatures(container, idf_encoder= False)
	feats = extractor()
	print feats.columns
	print feats.sample(1)
	extractor = NGramFeatures(container)
	feats = extractor()
	print feats.columns
	print feats.sample(1)
	vectorizer = Doc2Vec(corpus = container.question1.tolist() + container.question2.tolist())
	extractor = VectorFeatures(container, vectorizer = vectorizer, size = 12, n_batches = 1)
	feats = extractor()
	print feats.columns
	print feats.sample(1)



