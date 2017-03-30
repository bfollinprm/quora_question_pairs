#!/Users/follin/Documents/projects/kaggle/StackExchangeTransferLearn/venv/bin/python
from __future__ import unicode_literals
import numpy as np
from repoze.lru import lru_cache
from nltk import word_tokenize, ngrams
from exceptions import *
from collections import Counter
from chardet import detect

class TFIDF(object):
	def __init__(self, corpus, n = 1, eps = 1000, min_count = 2):
		self.eps = eps
		self.min_count = 2
		corpus = [s for s in corpus if isinstance(s, unicode)]
		self.weights = {
		word[0]: self._get_weight(count, eps, min_count) 
		for word, count in Counter(ngrams(word_tokenize(''.join(corpus).lower()), n)).items()
		}
	def __call__(self,word):
		return self.weights.get(word, 0.0)


	def _get_weight(self, count, eps, min_count):
		if count < min_count:
			return 0.0
		else:
			return 1.0/(count + eps)



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


def removestopwords(x):
	'''
	removes the stop words defined by nltk.corpus from a string
	'''
	if isinstance(x, unicode):
		filtered = [word for word in word_tokenize(x.lower()) if not is_stopword(word)]
		if len(filtered) == 0:
			return x
		else:
			return ' '.join(filtered)
	else:
		return x


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
stops = set(
	['a', "a's", 'able', 'about', 'above', 'according', 'accordingly', 'across', 
	'actually', 'after', 'afterwards', 'again', 'against', "ain't", 'all', 'allow', 
	'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 
	'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 
	'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 
	'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 
	'at', 'available', 'away', 'awfully', 'b', 'be', 'became', 'because', 'become', 
	'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'believe', 
	'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 
	'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause', 
	'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 
	'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 
	'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'definitely', 
	'described', 'despite', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', "don't", 
	'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'either', 'else', 
	'elsewhere', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'every', 
	'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 
	'far', 'few', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'former', 
	'formerly', 'forth', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 
	'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 
	"hadn't", 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he's", 
	'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 
	'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit', 'however', 
	'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 
	'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 
	'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 
	'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 
	'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks', 
	'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 
	'moreover', 'most', 'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 
	'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 
	'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 
	'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 
	'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 
	'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'per', 'perhaps', 
	'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provides', 'q', 'que', 'quite', 
	'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 
	'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 
	'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 
	'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'she', 'should', "shouldn't", 'since', 
	'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 
	'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 
	'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that',
	 "that's", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there's", 
	 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", 
	 "they'll", "they're", "they've", 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 
	 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 
	 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 
	 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 
	 'uucp', 'v', 'value', 'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 
	 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what's", 
	 'whatever', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 
	 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', "who's", 'whoever', 
	 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 
	 'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 
	 'yours', 'yourself', 'yourselves', 'z', 'zero', ''])
@lru_cache(maxsize=500)
def is_stopword(word):
	'''
	memoized stopword comparison
	'''
	return (len([x for x in stops if x == word]) > 0)
