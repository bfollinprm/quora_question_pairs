import re
import string



def remove_ascii(x):
	if isinstance(x, str):
		x = re.sub(r'[^\x00-\x7F]', r' ', x)
		return x
	else:
		return x
def to_unicode(x):
	if isinstance(x, str):
		x = unicode(x, encoding = 'utf-8')
		return x
	else:
		return x

def depunctualize(x):
	'''
	removes all punctuation, capital letters, and non ASCII characters from a string.
	'''
	if isinstance(x, str):
		x = re.sub('['+string.punctuation+']', r' ', x)
		return x
	else:
		return x

def striphtml(x):
	'''
	removes HTML/url tags from string
	'''
	if isinstance(x, str):
		x =  re.sub('<.*?>', r'', x)
		return x
	else:
		return x

def lower(x):
	if isinstance(x, unicode):
		x = x.lower()
	if isinstance(x, str):
		x = x.lower()
	else:
		return x

def replace_char(text):
	if isinstance(text, str):
		text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
		text = re.sub(r"what's", "what is ", text)
		text = re.sub(r"\'s", " ", text)
		text = re.sub(r"\'ve", " have ", text)
		text = re.sub(r"can't", "cannot ", text)
		text = re.sub(r"n't", " not ", text)
		text = re.sub(r"i'm", "i am ", text)
		text = re.sub(r"\'re", " are ", text)
		text = re.sub(r"\'d", " would ", text)
		text = re.sub(r"\'ll", " will ", text)
		text = re.sub(r",", " ", text)
		text = re.sub(r"\.", " ", text)
		text = re.sub(r"!", " ! ", text)
		text = re.sub(r"\/", " ", text)
		text = re.sub(r"\^", " ^ ", text)
		text = re.sub(r"\+", " + ", text)
		text = re.sub(r"\-", " - ", text)
		text = re.sub(r"\=", " = ", text)
		text = re.sub(r"'", " ", text)
		text = re.sub(r"60k", " 60000 ", text)
		text = re.sub(r":", " : ", text)
		text = re.sub(r" e g ", " eg ", text)
		text = re.sub(r" b g ", " bg ", text)
		text = re.sub(r" u s ", " american ", text)
		text = re.sub(r"\0s", "0", text)
		text = re.sub(r" 9 11 ", "911", text)
		text = re.sub(r"e - mail", "email", text)
		text = re.sub(r"j k", "jk", text)
		text = re.sub(r"\s{2,}", " ", text)
	return text
