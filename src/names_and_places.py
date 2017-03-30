import os
import os.path as osp


def getpeople(x):
	is_person = False

	return is_person

def getplaces(x):
	is_place = False

	return is_place

def getorganizations(x):
	is_org = False

	return is_org

class NERClassifier(object):
	def __init__(self):
		from nltk.tag import StanfordNERTagger
		parentdir = osp.join(osp.abspath(osp.join(os.getcwd(), os.pardir)), 'QuoraQuestionPairs')
		jarfile = osp.join(parentdir, 'data','stanford-ner-2015-04-20', 'stanford-ner-3.5.2.jar')
		modelfile = osp.join(parentdir, 'data','stanford-ner-2015-04-20', 'classifiers', 'english.all.3class.distsim.crf.ser.gz')
		self.tagger = StanfordNERTagger(modelfile, path_to_jar = jarfile)

	def __call__(sentence):
		if isinstance(sentence, list):
			return self.tagger.tag()