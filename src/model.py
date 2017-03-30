import xgboost as xgb
import numpy as np 
import pandas as pd 


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression


def xgb_cross_predictions(
	X,
	y, 
	X_test, 
	num_rounds = 1000, 
	eta = 0.01, 
	max_depth = 8, 
	gamma = 0, 
	n_kfolds = 10, 
	subsample = 0.5,
	max_features=1.0,
):
	params = {}
	params['objective'] = 'binary:logistic'
	params['eval_metric'] = 'logloss'
	params['silent'] = True

	params['eta'] = eta
	params['max_depth'] = max_depth
	params['gamma'] = gamma
	params['subsample'] = subsample
	params['max_features'] = max_features
	yhat_out = pd.Series(np.zeros(y.size))
	pred = np.zeros(X_test.shape[0])
	for i, (train, val) in enumerate(KFold(n_splits = n_kfolds, shuffle = True).split(X)):
		print 'Doing k-fold {} of {}'.format(i+1, n_kfolds)
		X_tr, X_val = X.loc[train], X.loc[val]
		y_tr, y_val = y.loc[train], y.loc[val]
		try:
			X_tr = xgb.DMatrix(X_tr, label = y_tr, missing = np.NAN)
			X_val = xgb.DMatrix(X_val, label = y_val, missing = np.NAN)
			X_test = xgb.DMatrix(X_test, missing = np.NAN)
		except:
			pass

		evallist = [(X_tr, 'train'),(X_val, 'eval')]
		print 'training model'
		bst = xgb.train(
				params, 
				X_tr, 
				num_boost_round = num_rounds, 
				evals = evallist, 
				early_stopping_rounds = 50,
				verbose_eval = 10,		
			)
		print 'rescaling predictions'
		#isotonic_rescale = IsotonicRegression()
		yhat_val = bst.predict(X_val)
		yhat_out.loc[val] = yhat_val

		#isotonic_rescale.fit(yhat_val, y_val)
		print 'predicting test set'
		yhat = bst.predict(X_test)
		#yhat = isotonic_rescale.transform(yhat)
		pred += np.array(yhat)
	return pred/n_kfolds, yhat_out



def train_xgb(X,y, num_rounds = 10, eta = 0.1, max_depth = 4, gamma = 0):
	if isinstance(X, pd.DataFrame):
		X = X
	if isinstance(y, pd.Series):
		y = y
	if isinstance(y, pd.DataFrame):
		y = y['is_duplicate']
	params = {}
	params['objective'] = 'binary:logistic'
	params['eval_metric'] = 'logloss'
	params['eta'] = eta
	params['max_depth'] = max_depth
	params['silent'] = True
	params['gamma'] = gamma
	X_tr, X_val, y_tr, y_val = train_test_split(X,y, test_size = 0.25)
	print 'fraction of duplicates:'
	print '\t train:\t{}'.format(sum(y_tr)*1.0/len(y_tr))
	print '\t val:\t{}'.format(sum(y_val)*1.0/len(y_val))
	dtrain = xgb.DMatrix(X_tr, label = y_tr, missing = np.NAN)
	dval = xgb.DMatrix(X_val, label = y_val, missing = np.NAN)
	evallist = [(dtrain, 'train'),(dval, 'eval')]
	bst = xgb.train(
		params, 
		dtrain, 
		num_boost_round = num_rounds, 
		evals = evallist, 
		early_stopping_rounds = 50,
		verbose_eval = 10,
	)
	return bst

def reweight_training_set(X,y):
	pos = X.loc[y == 1]
	neg = X.loc[y == 0]
	p = 0.165
	scale = ((len(pos) / float(len(X)) / p)) - 1
	while scale > 1:
		neg = pd.concat([neg, neg])
		scale -= 1
	neg = pd.concat([neg, neg.sample(n = int(scale * len(neg)))])
	X = pd.concat([pos, neg]).reset_index(drop = True)
	y = pd.Series((np.zeros(len(pos)) + 1).tolist() + np.zeros(len(neg)).tolist())
	return X, y


def predict_xgb(X, bst, test_index, submission_name = 'submission.txt'):
	if isinstance(X, pd.DataFrame):
		X = X
	d_test = xgb.DMatrix(X)
	yhat = bst.predict(d_test)
	sub = pd.DataFrame()
	sub['test_id'] = test_index
	sub['is_duplicate'] = yhat
	sub.to_csv(submission_name, index = False)
	return yhat

def plat_scaling(X, A,B, y):
	return x['y'] - 1.0/(1 + exp(A*x['yhat'] + B))

def plat_re_scale(y, yhat):
	from scipy.optimize import curve_fit
	x = {'y':y, 'yhat':yhat}
	popt, pcov = curve_fit(plat_scaling, x)
	return y - plat_scaling(x, popt[0], popt[1])

def isotonic_rescale(y, yhat):
	from sklearn.isotonic import IsotonicRegression
	return IsotonicRegression().fit_transform(yhat, y)


if __name__ == "__main__":
	X = pd.read_csv('../data/features.csv').drop('Unnamed: 0', errors = 'ignore')
	#ss = StandardScaler()
	#X = ss.fit_transform(X)
	y = pd.read_csv('../data/target.csv')['is_duplicate']
	X.y = reweight_training_set(X,y)
	bst = train_xgb(X,y)






	X_test = pd.read_csv('../data/features.csv').drop('Unnamed: 0', errors = 'ignore')
	#X = ss.fit_transform(X)

	yhat = predict_xgb(X_test, bst, X_test.index, submission_name = 'test_submission.txt')
