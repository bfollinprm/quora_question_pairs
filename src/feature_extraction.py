
from exceptions import *
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
	def __init__(self, container, remove_stopwords = False, prefix = ''):
		'''
		Initializes a character features instance. Options are:

		remove_stopwords:	Whether to keep the stopwords or remove them. Stop list currently hardcoded
							at src/feature_utilities

		prefix:		Something to prefix the feature names with, if you run multiple instances of this 
					class on your data.	
		'''
		super(CharFeatures).__init__(self, container)
		if remove_stopwords:
			self.q1 = self.q1.apply(removestopwords)
			self.q2 = self.q2.apply(removestopwords)
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
				self.q1.apply(lambda x: x.replace(' ', '').lower())
				== self.q2.apply(lambda x: x.replace(' ', '').lower())
				)
			questiontuples = zip(
				self.q1.apply(lambda x: x.replace(' ', '').lower()), 
				self.q2.apply(lambda x: x.replace(' ', '').lower())
				)
			self.feat[self.prefix+'match_len'] = np.array([common_string_length(x,y) for x,y in questiontuples])
			self.feat[self.prefix+'char_match_ratio'] = np.array(
				[diff_ratio(x,y) for x,y in zip(self.q1.values, self.q2.values)]
				)
			return self.feat


class NGramFeatures(FeatureSet):
	def __init__(self, container, idf_encoder = None, n = 1, remove_stopwords = False, prefix = ''):
		'''
		Initializes an ngram features instance. Options are:

		idf_encoder:	IDFWeights method that computes ngram weights. If none, builds one from 
						questions in DataContainer. If False, weights everything equally.

		n:	Size of the ngrams

		remove_stopwords:	Whether to keep the stopwords or remove them. Stop list currently hardcoded
							at src/feature_utilities

		prefix:		Something to prefix the feature names with, if you run multiple instances of this 
					class on your data.
		'''
		from encoders import IDFWeights
		from nltk import ngrams, word_tokenize
		self.weight_words = True
		super(CharFeatures).__init__(self, container)
		if remove_stopwords:
			self.q1 = self.q1.apply(removestopwords)
			self.q2 = self.q2.apply(removestopwords)
		self.prefix = prefix
		self.n = n
		if idf_encoder is None:
			self.idf_encoder = IDFWeights(corpus = self.q1.tolist() + self.q2.tolist(), n = n)
		elif isinstance(idf_encoder, IDFWeights):
			self.idf_encoder = idf_encoder
		elif not idf_encoder:
			self.weight_words = False
		else:
			raise InputError({}.format(idf_encoder), 'object is not an IDF encoder')
		if self.n == 1:
			self.q1 = self.q1.apply(lambda x: word_tokenize(x)).values
			self.q2 = self.q2.apply(lambda x: word_tokenize(x)).values
		if self.n > 1:
			self.q1 = self.q1.apply(lambda x: ngrams(word_tokenize(x), n = self.n)).values
			self.q2 = self.q2.apply(lambda x: ngrams(word_tokenize(x), n = self.n)).values

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
			q1unique_count = [sum(set(self.idf_encoder(w) for w in x)) for x in self.q1]
			q2unique_count = [sum(set(self.idf_encoder(w) for w in x)) for x in self.q2]
			common_words_count = [sum(self.idf_encoder(word) for word in common_ngrams(x,y)) for x,y in zip(ngrams1, ngrams2)]
		else:
			q1word_count = [len(x) for x in self.q1]
			q2word_count = [len(x) for x in self.q2]
			q1unique_count = [len(set(x)) for x in self.q1]
			q2unique_count = [len(set(x)) for x in self.q2]
			common_words_count = [len(common_ngrams(x,y)) for x,y in zip(ngrams1, ngrams2)]


		self.feat[self.prefix + 'num{}gramsQ1'.format(self.n)] = q1word_count
		self.feat[self.prefix + 'num{}gramsQ2'.format(self.n)] = q2word_count
		self.feat[self.prefix + 'num_unique{}gramsQ1'.format(self.n)] = q1unique_count
		self.feat[self.prefix + 'num_unique{}gramsQ2'.format(self.n)] = q2unique_count
		self.feat[self.prefix + 'common{}grams'.format(self.n)] = common_words_count
		self.feat[self.prefix + 'common{}grams_ratio'.format(self.n)] = 2*common_words_count/(q1unique_count + q2unique_count)


		return self.feat



### Unit tests
if __name__ = '__main__':
	pass
#### TO DO BELOW


def fuzzy_features(dc, remove_stopwords = True, suffix = ''):
	from data import DataContainer
	import fuzzyfeatures as ff
	from ngramfeatures import removestopwords

	if isinstance(dc, DataContainer):
		from cleaning import lower
		## Check for needed columns

		if remove_stopwords:
			#print 'removing stopwords'
			question1 = dc.question1.apply(removestopwords).apply(lower)
			question2 = dc.question2.apply(removestopwords).apply(lower)
		else:
			question1 = dc.question1.apply(lower)
			question2 = dc.question2.apply(lower)

		#print 'preparing fuzz'

		#print 'saving features'
		dc.features['QRatio'+suffix] = np.array([ff.qratio(x,y) for x,y in zip(question1, question2)])
		dc.features['SRatio'+suffix] = np.array([ff.sratio(x,y) for x,y in zip(question1, question2)])
		dc.features['PRatio'+suffix] = np.array([ff.pratio(x,y) for x,y in zip(question1, question2)])
		dc.features['Tsort'+suffix] = np.array([ff.tokensort(x,y) for x,y in zip(question1, question2)])
		dc.features['Tset'+suffix] = np.array([ff.tokenset(x,y) for x,y in zip(question1, question2)])

		return dc
	else:
		raise InputError('dc', 'input object is not a Data Container')


def vector_features(dc, vectorizer = None, n_principle_directions = 10, suffix = '', batch = True):
	from data import DataContainer
	import vectorfeatures as vf
	from ngramfeatures import removestopwords
	from sklearn.decomposition import PCA
	from numpy import mean
	if vectorizer is None:
		print 'grabbing Google News Word2Vec vectorizer'
		vectorizer = vf.GoogleWord2Vec()
	if vectorizer is 'GoogleNews':
		print 'grabbing Google News Word2Vec vectorizer'
		vectorizer = vf.GoogleWord2Vec()
	if vectorizer is 'QuoraQuestions':
		if isinstance(dc, DataContainer):
			print 'grabbing Quora Questions Word2Vec vectorizer'
			vectorizer = vf.QuoraWord2Vec(corpus = dc.question1.tolist()+dc.question2.tolist())

	batchsize = 10000.

	if isinstance(dc, DataContainer):
		dfs = []
		try:
			a = np.array_split(dc.question1, int(len(dc.question1)/batchsize))
			b = np.array_split(dc.question2, int(len(dc.question2)/batchsize))
		except:
			a = np.array_split(dc.question1, 10)
			b = np.array_split(dc.question2, 10)
		for q1, q2 in zip(a,b):
			temp_dict = {}
			temp_dict['index'] = q1.index

			#print 'removing stopwords'
			question1 = q1.apply(removestopwords).values
			question2 = q2.apply(removestopwords).values

			#print 'vectorizing tokens'
			vecQ1 = np.array([vf.vectorize(x, vectorizer = vectorizer) for x in question1])
			meanQ1 = np.array([vf.meanvector(x) for x in vecQ1])
			stdevQ1 = np.array([vf.stdvector(x) for x in vecQ1])
			vecQ2 = np.array([vf.vectorize(x, vectorizer = vectorizer) for x in question2])
			meanQ2 = np.array([vf.meanvector(x) for x in vecQ2])
			stdevQ2 = np.array([vf.stdvector(x) for x in vecQ2])
			sumQ1 = np.array([vf.sumvector(x) for x in vecQ1])
			sumQ2 = np.array([vf.sumvector(x) for x in vecQ2])

			meandiff = np.array([vf.difference(x,y) for (x,y) in zip(meanQ1, meanQ2)])

			#print 'saving features'
			temp_dict['mean_cosdist'+suffix] = np.array([vf.cosine(x,y) for x,y in zip(meanQ1, meanQ2)])
			temp_dict['mean_l2dist'+suffix] = np.array([vf.euclidean(x,y) for x,y in zip(meanQ1, meanQ2)])
			temp_dict['mean_l1dist'+suffix] = np.array([vf.cityblock(x,y) for x,y in zip(meanQ1, meanQ2)])
			temp_dict['mean_BCdist'+suffix] = np.array([vf.braycurtis(x,y) for x,y in zip(meanQ1, meanQ2)])
			temp_dict['mean_l3dist'+suffix] = np.array([vf.minkowski(x,y, 3) for x,y in zip(meanQ1, meanQ2)])
			temp_dict['mean_l4dist'+suffix] = np.array([vf.minkowski(x,y, 4) for x,y in zip(meanQ1, meanQ2)])
			temp_dict['mean_corrdist'+suffix] = np.array([vf.correlation(x,y) for x,y in zip(meanQ1, meanQ2)])

			stddiff = np.array([vf.difference(x,y) for (x,y) in zip(stdevQ1, stdevQ2)])

			temp_dict['stdev_cosdist'+suffix] = np.array([vf.cosine(x,y) for x,y in zip(stdevQ1, stdevQ2)])
			temp_dict['stdev_l2dist'+suffix] = np.array([vf.euclidean(x,y) for x,y in zip(stdevQ1, stdevQ2)])
			temp_dict['stdev_l1dist'+suffix] = np.array([vf.cityblock(x,y) for x,y in zip(stdevQ1, stdevQ2)])
			temp_dict['stdev_BCdist'+suffix] = np.array([vf.braycurtis(x,y) for x,y in zip(stdevQ1, stdevQ2)])
			temp_dict['stdev_l3dist'+suffix] = np.array([vf.minkowski(x,y, 3) for x,y in zip(stdevQ1, stdevQ2)])
			temp_dict['stdev_l4dist'+suffix] = np.array([vf.minkowski(x,y, 4) for x,y in zip(stdevQ1, stdevQ2)])
			temp_dict['stdev_corrdist'+suffix] = np.array([vf.correlation(x,y) for x,y in zip(stdevQ1, stdevQ2)])


			sumdiff = np.array([vf.difference(x,y) for (x,y) in zip(sumQ1, sumQ2)])

			temp_dict['sum_cosdist'+suffix] = np.array([vf.cosine(x,y) for x,y in zip(sumQ1, sumQ2)])
			temp_dict['sum_l2dist'+suffix] = np.array([vf.euclidean(x,y) for x,y in zip(sumQ1, sumQ2)])
			temp_dict['sum_l1dist'+suffix] = np.array([vf.cityblock(x,y) for x,y in zip(sumQ1, sumQ2)])
			temp_dict['sum_BCdist'+suffix] = np.array([vf.braycurtis(x,y) for x,y in zip(sumQ1, sumQ2)])
			temp_dict['sum_l3dist'+suffix] = np.array([vf.minkowski(x,y, 3) for x,y in zip(sumQ1, sumQ2)])
			temp_dict['sum_l4dist'+suffix] = np.array([vf.minkowski(x,y, 4) for x,y in zip(sumQ1, sumQ2)])
			temp_dict['sum_corrdist'+suffix] = np.array([vf.correlation(x,y) for x,y in zip(sumQ1, sumQ2)])


			feature_selector = PCA(n_components = n_principle_directions,svd_solver = 'randomized', copy = False)
			diffvecs = feature_selector.fit_transform(meandiff)
			for i in range(n_principle_directions):
				try:
					temp_dict['delta_mean_{0}'.format(i)+suffix] = diffvecs[:,i]
				except:
					return np.NaN

			diffvecs = feature_selector.fit_transform(stddiff)
			for i in range(n_principle_directions):
				try:
					temp_dict[ 'delta_stdev_{0}'.format(i)+suffix] = diffvecs[:,i]
				except:
					return np.NaN

			diffvecs = feature_selector.fit_transform(meandiff)
			for i in range(n_principle_directions):
				try:
					temp_dict[ 'delta_mean_{0}'.format(i)+suffix] = diffvecs[:,i]
				except:
					return np.NaN


			matrix = np.array([vf.matrix_similarity(x,y) for x, y in zip(vecQ1, vecQ2)])
			matrix = feature_selector.fit_transform(matrix)

			for i in range(n_principle_directions):
					try:
						temp_dict['sim_mat_{}'.format(i)+suffix] = matrix[:, i]

					except:
						temp_dict['sim_mat_{}'.format(i)+suffix] = matrix[:, i]
			dfs += [pd.DataFrame.from_dict(temp_dict).set_index('index')]

		df = pd.concat(dfs)
		dc.features = dc.features.join(df)
		return dc


	else:
		raise InputError('dc', 'input object is not a Data Container')



def pos_features(dc, tagger = None):
	from data import DataContainer
	from ngramfeatures import TFIDF
	import posfeatures as pf
	if isinstance(dc, DataContainer):



		question1 = dc.question1.values
		question2 = dc.question2.values
		print 'building tfidf'
		tfidf = TFIDF(corpus = question1.tolist() + question2.tolist(), n = 1)
		print 'building POS tagger'
		if not isinstance(tagger, pf.POS_Tagger):
			tagger = pf.POS_Tagger(corpus = question1.tolist() + question2.tolist())

		print 'getting noun features'
		dc.features['nouns_q1'] = np.array([len(pf.unique_nouns(x, tagger = tagger)) if isinstance(x, unicode) else np.NaN for x in question1])
		dc.features['nouns_q2'] = np.array([len(pf.unique_nouns(x, tagger = tagger)) if isinstance(x, unicode) else np.NaN for x in question2])

		dc.features['delta_nouns'] = abs(
				dc.features['nouns_q1'].values  -
				dc.features['nouns_q2'].values
			)
		print 'getting shared noun features'
		dc.features['shared_nouns'] = np.array([pf.num_common_unique_nouns(x,y, tagger = tagger) for x,y in zip(question1, question2)])
		dc.features['shared_nouns_weighted'] = np.array([pf.num_common_unique_nouns_weighted(x,y, tfidf = tfidf, tagger = tagger)
			for x,y in zip(question1, question2)
			])

		print 'getting verb features'

		dc.features['verbs_q1'] = np.array([len(pf.unique_verbs(x, tagger = tagger)) if isinstance(x, unicode) else np.NaN for x in question1])
		dc.features['verbs_q2'] = np.array([len(pf.unique_verbs(x, tagger = tagger)) if isinstance(x, unicode) else np.NaN for x in question2])


		print 'getting shared verb features'

		dc.features['delta_verbs'] = abs(
				dc.features['verbs_q1'].values  -
				dc.features['verbs_q2'].values
			)
		dc.features['shared_verbs'] = np.array([pf.num_common_unique_verbs(x,y, tagger = tagger) for x,y in zip(question1, question2)])
		dc.features['shared_verbs_weighted'] = np.array([pf.num_common_unique_verbs_weighted(x,y, tfidf = tfidf, tagger = tagger)
			for x,y in zip(question1, question2)
			])

		print 'getting interrogative features'
		vectorizer = pf.Interrogatives()
		interQ1 = [pf.interrogatives(x, vectorizer = vectorizer) for x in question1]
		interQ2 = [pf.interrogatives(x, vectorizer = vectorizer) for x in question2]

		dc.features['inter_cosdist'] = np.array([pf.cosine(x,y) for x,y in zip(interQ1, interQ2)])
		dc.features['inter_euclidean'] = np.array([pf.euclidean(x,y) for x,y in zip(interQ1, interQ2)])
		dc.features['inter_cityblock'] = np.array([pf.cityblock(x,y) for x,y in zip(interQ1, interQ2)])
		dc.features['inter_braycurtis'] = np.array([pf.braycurtis(x,y) for x,y in zip(interQ1, interQ2)])
		dc.features['inter_minkowski3'] = np.array([pf.minkowski(x,y, 3) for x,y in zip(interQ1, interQ2)])
		dc.features['inter_minkowski4'] = np.array([pf.minkowski(x,y, 4) for x,y in zip(interQ1, interQ2)])
		dc.features['inter_correlation'] = np.array([pf.correlation(x,y) for x,y in zip(interQ1, interQ2)])
		dc.features['most_common_inter_q1'] = np.array([np.argmax(x) if np.amax(x) > 1 else np.NaN for x in interQ1])
		dc.features['most_common_inter_q2'] = np.array([np.argmax(x) if np.amax(x) > 1 else np.NaN for x in interQ2])

		print 'getting capital letter features'
		dc.features['caps_q1'] = np.array([len(pf.unique_caps(x)) if isinstance(x, unicode) else np.NaN for x in question1])
		dc.features['caps_q2'] = np.array([len(pf.unique_caps(x)) if isinstance(x, unicode) else np.NaN for x in question2])
		dc.features['delta_caps'] = abs(
				dc.features['caps_q1'].values  -
				dc.features['caps_q2'].values
			)
		dc.features['shared_caps'] = np.array([pf.num_common_unique_caps(x,y) for x,y in zip(question1, question2)])
		dc.features['shared_caps_weighted'] = np.array([pf.num_common_unique_caps_weighted(x,y, tfidf = tfidf)
			for x,y in zip(question1, question2)
			])		


		print 'getting numerical features'
		dc.features['numerics_q1'] = np.array([len(pf.unique_nums(x)) if isinstance(x, unicode) else np.NaN for x in question1])
		dc.features['numerics_q2'] = np.array([len(pf.unique_nums(x)) if isinstance(x, unicode) else np.NaN for x in question2])
		dc.features['delta_numerics'] = abs(
				dc.features['numerics_q1'].values  -
				dc.features['numerics_q2'].values
			)
		dc.features['shared_numerics'] = np.array([pf.num_common_unique_nums(x,y) for x,y in zip(question1, question2)])
		dc.features['shared_numerics_weighted'] = np.array([pf.num_common_unique_nums_weighted(x,y, tfidf = tfidf)
			for x,y in zip(question1, question2)
			])		



	else:
		raise InputError('dc', 'input object is not a Data Container')







def get_element(x, index = 0):
	try:
		return x[index]
	except:
		return np.NaN
