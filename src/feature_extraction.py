
from exceptions import *
import numpy as np
import pandas as pd


def char_features(dc, remove_stopwords = False, suffix = ''):
	from data import DataContainer
	import charfeatures as cf
	from ngramfeatures import removestopwords

	if isinstance(dc, DataContainer):
		## Check for needed columns

		if remove_stopwords:
			#print 'removing stopwords'
			question1 = dc.question1.apply(removestopwords)
			question2 = dc.question2.apply(removestopwords)
		else:
			question1 = dc.question1
			question2 = dc.question2

		dc.features['len_q1'+suffix] = np.array([cf.qlength(x) for x in question1])
		dc.features['len_q2'+suffix] = np.array([cf.qlength(x) for x in question2])
		dc.features['len_diff'+suffix] = (dc.features['len_q1'+suffix] - dc.features['len_q2'+suffix]).apply(abs)
		dc.features['are_equiv'+suffix] = question1.replace(' ', '') == question2.replace(' ', '')

		dc.features['z_match_ratio'+suffix] = np.array([cf.diff_ratios(x,y) for (x,y) in zip(question1, question2)])
def ngram_features(dc, n = 1, remove_stopwords = True, suffix = ''):
	from data import DataContainer
	import ngramfeatures as nf

	if isinstance(dc, DataContainer):
		## Check for needed columns

		if remove_stopwords:
			#print 'removing stopwords'
			question1 = dc.question1.apply(nf.removestopwords)
			question2 = dc.question2.apply(nf.removestopwords)
		else:
			question1 = dc.question1
			question2 = dc.question2
		tfidf = nf.TFIDF(question1.tolist() + question2.tolist(), n = n)
		#print 'preparing ngrams'
		token1 = question1.apply(nf.tokenize)
		token2 = question2.apply(nf.tokenize)
		if n > 1:
			ngrams1 = token1.apply(nf.get_ngrams, n = n)
			ngrams2 = token2.apply(nf.get_ngrams, n = n)
		if n < 1:
			raise InputError('n', 'n must be int >= 1')
		if n == 1:
			ngrams1 = token1
			ngrams2 = token2



		#print 'saving features'
		dc.features['num_{0}grams_Q1'.format(n)+suffix] = ngrams1.apply(nf.questionlength, unique = False)
		dc.features['num_{0}grams_Q2'.format(n)+suffix] = ngrams2.apply(nf.questionlength, unique = False)

		dc.features['delta_num_{0}grams'.format(n)+suffix] = abs(
				dc.features['num_{0}grams_Q1'.format(n)+suffix].values -
				dc.features['num_{0}grams_Q2'.format(n)+suffix].values
			)
		dc.features['num_unique_{0}grams_Q1'.format(n)+suffix] = ngrams1.apply(nf.questionlength, unique = True)
		dc.features['num_unique_{0}grams_Q2'.format(n)+suffix] = ngrams2.apply(nf.questionlength, unique = True)

		dc.features['delta_unique_{0}grams'.format(n)+suffix] = abs(
				dc.features['num_unique_{0}grams_Q1'.format(n)+suffix].values -
				dc.features['num_{0}grams_Q2'.format(n)+suffix].values
			)

		dc.features['num_common_{0}grams'.format(n)+suffix] = np.array([len(nf.common_ngrams(x,y)) for x,y in zip(ngrams1, ngrams2)])

		dc.features['frac_common_{0}grams'.format(n)+suffix] = (
			2 * dc.features['num_common_{0}grams'.format(n)+suffix]/(
				dc.features['num_unique_{0}grams_Q1'.format(n)+suffix] + 
				dc.features['num_unique_{0}grams_Q2'.format(n)+suffix]
				)
			)
		dc.features['weighted_frac_common_{0}grams'.format(n) + suffix] = np.array(
				[nf.common_weighted_fraction(x,y, tfidf = tfidf) for x,y in zip(ngrams1, ngrams2)]
			)

		return dc
	else:
		raise InputError('dc', 'input object is not a Data Container')




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
