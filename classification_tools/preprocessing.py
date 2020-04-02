##########################################################
#                PREPROCESSING FUNCTIONS                 #
##########################################################
# rcATT is a tool to prediction tactics and techniques 
# from the ATT&CK framework, using multilabel text
# classification and post processing.
# Version:    1.00
# Author:     Valentine Legoy
# Date:       2019_10_22
# File dedicated to text pre-processing functions for
# future classification: clean up text and stemming.

import re

from sklearn.base import BaseEstimator, TransformerMixin

from nltk import word_tokenize		  
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer

def clean_text(text):
	"""
	Cleaning up the words contractions, unusual spacing, non-word characters and any computer science
	related terms that hinder the classification.
	"""
	text = str(text)
	text = text.lower()
	text = re.sub("\r\n", "\t", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "can not ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r"\'scuse", " excuse ", text)
	text = re.sub('(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.)\{3\}(?:25[0-5] |2[0-4][0-9]|[01]?[0-9][0-9]?)(/([0-2][0-9]|3[0-2]|[0-9]))?', 'IPv4', text)
	text = re.sub('\b(CVE\-[0-9]{4}\-[0-9]{4,6})\b', 'CVE', text)
	text = re.sub('\b([a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b', 'email', text)
	text = re.sub('\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', 'IP', text)
	text = re.sub('\b([a-f0-9]{32}|[A-F0-9]{32})\b', 'MD5', text)
	text = re.sub('\b((HKLM|HKCU)\\[\\A-Za-z0-9-_]+)\b', 'registry', text)
	text = re.sub('\b([a-f0-9]{40}|[A-F0-9]{40})\b', 'SHA1', text)
	text = re.sub('\b([a-f0-9]{64}|[A-F0-9]{64})\b', 'SHA250', text)
	text = re.sub('http(s)?:\\[0-9a-zA-Z_\.\-\\]+.', 'URL', text)
	text = re.sub('CVE-[0-9]{4}-[0-9]{4,6}', 'vulnerability', text)
	text = re.sub('[a-zA-Z]{1}:\\[0-9a-zA-Z_\.\-\\]+', 'file', text)
	text = re.sub('\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b', 'hash', text)
	text = re.sub('x[A-Fa-f0-9]{2}', ' ', text)
	text = re.sub('\W', ' ', text)
	text = re.sub('\s+', ' ', text)
	text = text.strip(' ')
	return text

def processing(df):
	"""
	Creating a function to encapsulate preprocessing, to make it easy to replicate on submission data
	"""
	df['processed'] = df['Text'].map(lambda com : clean_text(com))
	return(df)

def remove_u(input_string):
	"""
	Convert unicode text
	"""
	words = input_string.split()
	words_u = [(word.encode('unicode-escape')).decode("utf-8", "strict") for word in words]
	words_u = [word_u.split('\\u')[1] if r'\u' in word_u else word_u for word_u in words_u]
	return ' '.join(words_u)

class StemTokenizer(object):
	"""
	Transform each word to its stemmed version
	e.g. studies --> studi
	"""
	def __init__(self):
		self.st = EnglishStemmer()
		
	def __call__(self, doc):
		return [self.st.stem(t) for t in word_tokenize(doc)]

class LemmaTokenizer(object):
	"""
	Transform each word to its lemmatized version
	e.g. studies --> study
	"""
	def __init__(self):
		self.wnl = WordNetLemmatizer()
		
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class TextSelector(BaseEstimator, TransformerMixin):
	"""
	Transformer to select a single column from the data frame to perform additional transformations on
	Use on text columns in the data
	"""
	def __init__(self, key):
		self.key = key

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[self.key]