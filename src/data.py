from __future__ import unicode_literals
import pandas as pd
import numpy as np
import feature_extraction
from exceptions import *
from cleaning import striphtml, remove_ascii, depunctualize, replace_char, to_unicode

class DataContainer(object):
	''' 
	Container Class for competition data, inherets from pd.DataFrame
	'''
	def __init__(self, filename, size = 0):
		if size == 0:
			df = pd.read_csv(filename)
		else:
			df = pd.read_csv(filename).sample(n = size)

		self.question1 = df.question1
		self.question2 = df.question2
		try:
			self.target = df.is_duplicate.apply(lambda x: bool(x))
		except:
			pass
		try:
			self.test_id = df.test_id.apply(lambda x: int(x))
		except:
			pass
		self.features = df[['question1', 'question2']].drop(['question1' , 'question2'], axis = 1, errors = 'ignore')


	def clean_questions(self):
		self.question1 = self._cleanstrarray(self.question1)
		self.question2 = self._cleanstrarray(self.question2)


	def _cleanstrarray(self,series):
		return series.apply(striphtml).apply(remove_ascii).apply(depunctualize).apply(replace_char).apply(to_unicode)

	def get_char_features(self, remove_stopwords = False, suffix = ''):
		return feature_extraction.char_features(self, remove_stopwords = remove_stopwords, suffix = suffix)

	def get_ngram_features(self, n = 1, remove_stopwords = True, suffix = ''):
		return feature_extraction.ngram_features(self, n = n, remove_stopwords = remove_stopwords, suffix = suffix)

	def get_fuzzy_features(self,remove_stopwords = True, suffix = ''):
		return feature_extraction.fuzzy_features(self, remove_stopwords = remove_stopwords, suffix = suffix)

	def get_vector_features(self, vectorizer = None, n_principle_directions = 10, suffix = ''):
		n = n_principle_directions
		return feature_extraction.vector_features(self, vectorizer = vectorizer, n_principle_directions = n, suffix = suffix)

	def get_pos_features(self, tagger = None):
		return feature_extraction.pos_features(self, tagger = tagger)


def clean(dataframe):
	''' 
	Cleans raw data. 
	'''
	if isinstance(dataframe, pd.DataFrame):
		# modify content here
		if ('question1' in dataframe.columns):
			dataframe['question1_raw'] = dataframe.question1
			dataframe['question1'] = clean_question(dataframe.question1)


		if ('question2' in dataframe.columns):
			dataframe['question2_raw'] = dataframe.question2
			dataframe['question2'] = clean_question(dataframe.question2)


		if ('is_duplicate' in dataframe.columns):
			series = dataframe.is_duplicate
			series = series.apply(pd.to_numeric)
			dataframe['is_duplicate'] = series

		if ('id' in dataframe.columns):
			series = dataframe.id
			series = series.apply(pd.to_numeric)
			dataframe['id'] = series

		if ('qid1' in dataframe.columns):
			series = dataframe.qid1
			series = series.apply(pd.to_numeric)
			dataframe['qid1'] = series

		if ('qid2' in dataframe.columns):
			series = dataframe.qid2
			series = series.apply(pd.to_numeric)
			dataframe['qid2'] = series
			return dataframe
	else:
		raise InputError('dataframe', 'input object is not a pd.DataFrame')
		return pd.dataframe()




def clean_question(x):
	return x.apply(striphtml).apply(remove_ascii).apply(lower).apply(depunctualize)

