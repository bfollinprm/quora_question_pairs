import pandas as pd 
import numpy as np
import src.data as data
import src.model as model
import os
import os.path as osp




import src.data as data 

train = DataContainer(osp.join(parentdir, 'data/train.csv'))

def get_data(files = None):
	if files is None:
		parentdir = osp.join(osp.abspath(osp.join(os.getcwd(), os.pardir)), 'QuoraQuestionPairs')

		print 'loading unprocessed data'
		train = data.DataContainer(osp.join(parentdir,'data/train.csv'))
		test = data.DataContainer(osp.join(parentdir, 'data/test.csv'))
		out = [train, test]
	else:
		parentdir = osp.join(osp.abspath(osp.join(os.getcwd(), os.pardir)), 'QuoraQuestionPairs')

		print 'attempting to open pickled data'
		import cPickle
		out = []
		for file in files:
			with open(osp.join(parentdir, file), 'rb') as f:
				out += [cPickle.load(f)]

	return out





def get_features(dc, pos_tagger = None, quoraword2vec = None):
	from feature_extration import *
	print 'getting character features'
	dc.features = CharFeatures(dc, remove_stopwords = False, prefix = 'stops_')
	dc.features = CharFeatures(dc, remove_stopwords = True)

	print 'getting fuzzy comparison features'
	dc.features = NGramFeatures(dc, n = n, remove_stopwords = False, prefix = stops_)
	dc.features = NGramFeatures(dc, n = n, remove_stopwords = True)
	dc.features = NGramFeatures(dc, n = n, remove_stopwords = False, )
	dc.get_char_features(remove_stopwords = False, suffix = '_stops')
	dc.get_char_features(remove_stopwords = True)

	print 'getting part of speech features'
	dc.get_pos_features(tagger = pos_tagger)

	print 'getting fuzzy comparison features'
	dc.get_fuzzy_features(remove_stopwords = False, suffix = '_stops')
	dc.get_fuzzy_features(remove_stopwords = True)


	print 'getting ngram features'
	for n in range(1,3):
		dc.get_ngram_features(n=n, remove_stopwords = False, suffix = 'w_stops')
		dc.get_ngram_features(n=n, remove_stopwords = True, suffix = '')

	print 'getting vector features'
	for vectorizer in ['GoogleNews', 'QuoraQuestions']:
		if vectorizer is 'GoogleNews':
			from src.vectorfeatures import GoogleWord2Vec
			vectorizer = GoogleWord2Vec()
			suffix = 'GoogleNews'
		elif vectorizer is 'QuoraQuestions':
			vectorizer = quoraword2vec
			suffix = 'QuoraQuestions'
		else: 
			vectorizer = None
			suffix = ''
		dc.get_vector_features(n_principle_directions = 10, vectorizer = vectorizer, suffix = suffix)

	return dc


def grid_search():
	train, test = get_data(files = ['data/train.pkl', 'data/test.pkl'])
	from sklearn.model_selection import GridSearchCV
	from xgboost.sklearn import XGBClassifier
	estimator = XGBClassifier()

	parameters = {'nthread':[-1], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': np.logspace(-5,0, 3), #so called `eta` value
              'max_depth': np.arange(2,3),
              'min_child_weight': np.linspace(0,20,3),
              'silent': [1],
              'subsample': np.linspace(0.5,1, 3),
              'colsample_bytree': np.linspace(0.5,1,3),
              'n_estimators': [500], #number of trees, change it to 1000 for better results
              'missing':[np.NaN],
    }
	gridsearch = GridSearchCV(estimator, parameters, cv = 3, verbose = True, refit = True)
	gridsearch.fit(train.features.values, train.target.values)
	best_parameters, score, _ = max(gridsearch.grid_scores_, key=lambda x: x[1])
	print('score:', score)
	for param_name in sorted(best_parameters.keys()):
		print("%s: %r" % (param_name, best_parameters[param_name]))



if __name__ == '__main__':
	print 'getting data'
	try:
		train, test = get_data(files = ['data/train.pkl', 'data/test.pkl'])
	except:
		pass
		import cPickle
		train, test = get_data()
		print 'cleaning data'
		train.clean_questions()
		print 'getting training features'
		from src.posfeatures import POS_Tagger 
		from src.vectorfeatures import QuoraWord2Vec
		print 'training vectorizers'
		pos_tagger = POS_Tagger(corpus = train.question1.tolist() + train.question2.tolist()+test.question1.tolist() + test.question2.tolist(), file = 'data/POSTagger.bin')
		quora_word2vec = QuoraWord2Vec(corpus = train.question1.tolist() + train.question2.tolist() + test.question1.tolist() + test.question2.tolist())
		quora_word2vec = None
		train = get_features(train, pos_tagger = pos_tagger, quoraword2vec = quora_word2vec)
		print 'getting testing features'
		test = None
		with open('data/train.pkl', "wb") as f:
			cPickle.dump(train, f,protocol = -1)


	features, target = train.features, train.target
	features, target = model.reweight_training_set(train.features, train.target)
	print 'getting test set features'
	if not isinstance(test.features, pd.DataFrame):
		_, test = get_data()
		test.clean_questions()
		test = get_features(test, pos_tagger = pos_tagger, quoraword2vec = quora_word2vec)
		with open('data/test.pkl', "wb") as f:
			cPickle.dump(test, f,protocol =  -1)
	print 'saving submission'

	#bst = model.train_xgb(features, target, num_rounds = 200, eta = 0.1)
	predictions, yhat = model.xgb_cross_predictions(
		features,
		target, 
		test.features, 
		num_rounds = 500, 
		eta = 0.05, 
		max_depth = 10, 
		gamma = 0, 
		n_kfolds = 10,
		subsample = 0.5)
	#predictions = model.predict_xgb(test.features, bst, test.features.index, submission_name = 'submission.txt')

	sub = pd.DataFrame()
	sub['test_id'] = test.features.index
	sub['is_duplicate'] = predictions
	sub.to_csv('bagged_XGBoost.txt', index = False)
	yhat.to_csv('training_set_predictions', index_label = 'test_id')

