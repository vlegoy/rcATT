##########################################################
#                POSTPROCESSING FUNCTIONS                #
##########################################################
# rcATT is a tool to prediction tactics and techniques 
# from the ATT&CK framework, using multilabel text
# classification and post processing.
# Version:    1.00
# Author:     Valentine Legoy
# Date:       2019_10_22
# File for post-processing functions. Two types of post-
# processing methods are compared to the non-post-
# processing classification at each training of the model
# with new data: confidence propagation and hanging node.
# The results are saved in the configuration file and these
# functions are reused during prediction with the best
# post-processing method.

import joblib
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold

from nltk.corpus import stopwords

import classification_tools.preprocessing as prp
import classification_tools as clt


def print_progress_bar(iteration):
	"""
	Print a progress bar for command-line interface training
	"""
	percent = ("{0:.1f}").format(100 * (iteration / float(50)))
	filledLength = int(iteration)
	bar = '█' * filledLength + '-' * (50 - filledLength)
	prefix = "Progress:"
	suffix = "Complete"
	printEnd = "\r"
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
	if iteration == 50: 
		print()

def confidence_propagation_single(tactics_confidence_list, technique_name, technique_confidence_score):
	"""
	Modify predictions and confidence scores of one technique  using a boosting method depending on this
	technique's and its related tactics' confidence score.
	"""
	new_confidence_score = technique_confidence_score
	i = 0
	for tactic in clt.CODE_TACTICS:
		if not clt.TACTICS_TECHNIQUES_RELATIONSHIP_DF.loc[clt.TACTICS_TECHNIQUES_RELATIONSHIP_DF[tactic] == technique_name].empty:
			lambdaim = 1/(np.exp(abs(technique_confidence_score-tactics_confidence_list[tactic])))
			new_confidence_score = new_confidence_score + lambdaim * tactics_confidence_list[tactic]
		i = i+1
	return new_confidence_score

def confidence_propagation( predprob_tactics, pred_techniques, predprob_techniques):
	"""
	Modify predictions and confidences scores of all techniques of the whole set using 
	confidence_propagation_single function.
	"""
	pred_techniques_corrected = pred_techniques
	predprob_techniques_corrected = predprob_techniques
	tactics_confidence_df = pd.DataFrame(data = predprob_tactics, columns = clt.CODE_TACTICS)
	for j in range(len(predprob_techniques[0])):
		for i in range(len(predprob_techniques)):
			predprob_techniques_corrected[i][j] = confidence_propagation_single(tactics_confidence_df[i:(i+1)], clt.CODE_TECHNIQUES[j], predprob_techniques[i][j])
			if predprob_techniques_corrected[i][j] >= float(0) :
				pred_techniques_corrected[i][j] = int(1)
			else:
				pred_techniques_corrected[i][j] = int(0)
	return pred_techniques_corrected, predprob_techniques_corrected

def hanging_node(pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, c, d):
	"""
	Modify prediction of techniques depending on techniques and related tactics confidence score on a
	threshold basis.
	"""
	predprob_techniques_corrected = pred_techniques
	for i in range(len(pred_techniques)):
		for j in range(len(pred_techniques[0])):
			for k in range(len(pred_tactics[0])):
				if not clt.TACTICS_TECHNIQUES_RELATIONSHIP_DF.loc[clt.TACTICS_TECHNIQUES_RELATIONSHIP_DF[clt.CODE_TACTICS[k]] == clt.CODE_TECHNIQUES[j]].empty:
					if predprob_techniques[i][j] < c and predprob_techniques[i][j] > 0 and predprob_tactics[i][k] < d:
						predprob_techniques_corrected[i][k] = 0 
	return predprob_techniques_corrected

def combinations(c, d):
	"""
	Compute all combinations possible between c and d and their derived values.
	"""
	c_list = [c-0.1, c, c+0.1]
	d_list = [d-0.1, d, d+0.1]
	possibilities = []
	for cl in c_list:
		for dl in d_list:
			possibilities.append([cl, dl])
	return possibilities

def hanging_node_threshold_comparison(pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, known_pred_techniques, permutations):
	"""
	Using different combinations of thresholds retrieve all the F0.5 score macro-averaged between the
	post-processed predictions and the true labels.
	"""
	f05list = []
	for pl in permutations:
		f05list_temp = [pl]
		new_pred_techniques = hanging_node(pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, pl[0], pl[1])
		f05list_temp.append(fbeta_score(known_pred_techniques, new_pred_techniques, beta=0.5, average='macro'))
		f05list.append(f05list_temp)
	return f05list

def find_best_post_processing(cmd):
	"""
	Find best postprocessing approach to use with the new dataset based on the f0.5 score macro-averaged.
	"""
	# add stop words to the list found during the development of rcATT
	stop_words = stopwords.words('english')
	new_stop_words = ["'ll", "'re", "'ve", 'ha', 'wa',"'d", "'s", 'abov', 'ani', 'becaus', 'befor', 'could', 'doe', 'dure', 'might', 'must', "n't", 'need', 'onc', 'onli', 'ourselv', 'sha', 'themselv', 'veri', 'whi', 'wo', 'would', 'yourselv']
	stop_words.extend(new_stop_words)
	
	# download both dataset: original to the tool and added by the user
	train_data_df = pd.read_csv('classification_tools/data/training_data_original.csv', encoding = "ISO-8859-1")
	train_data_added = pd.read_csv('classification_tools/data/training_data_added.csv', encoding = "ISO-8859-1")
	train_data_df.append(train_data_added, ignore_index = True)
	
	# preprocess the report
	train_data_df = prp.processing(train_data_df)
	
	# split the dataset in 5 fold to be able to give a more accurate F0.5 score
	kf = KFold(n_splits=5, shuffle = True, random_state=42)
	reports = train_data_df[clt.TEXT_FEATURES]
	overall_ttps = train_data_df[clt.ALL_TTPS]
	
	# get current configuration parameters for post-processing method hanging-node to define new thresholds
	parameters = joblib.load("classification_tools/data/configuration.joblib")
	c = parameters[1][0]
	d = parameters[1][1]
	permutations = combinations(c, d)
	
	f05_NO = [] #list of f0.5 score for all techniques predictions sets without post-processing
	f05_HN = [] #list of f0.5 score for all techniques predictions sets with hanging node post-processing
	f05_CP = [] #list of f0.5 score for all techniques predictions sets with confidence propagation post-processing
	
	# retrieve minimum and maximum probabilities to use in MinMaxScaler
	min_prob_tactics = 0.0
	max_prob_tactics = 0.0
	min_prob_techniques = 0.0
	max_prob_techniques = 0.0
	
	i = 6 # print progress bar counter
	
	for index1, index2 in kf.split(reports, overall_ttps):
		# splits the dataset according to the kfold split into training and testing sets, and data and labels
		reports_train, reports_test = reports.iloc[index1], reports.iloc[index2]
		overall_ttps_train, overall_ttps_test = overall_ttps.iloc[index1], overall_ttps.iloc[index2]

		train_reports = reports_train[clt.TEXT_FEATURES]
		test_reports = reports_test[clt.TEXT_FEATURES]

		train_tactics = overall_ttps_train[clt.CODE_TACTICS]
		train_techniques = overall_ttps_train[clt.CODE_TECHNIQUES]
		test_tactics = overall_ttps_test[clt.CODE_TACTICS]
		test_techniques = overall_ttps_test[clt.CODE_TECHNIQUES]
		
		# Define a pipeline combining a text feature extractor with multi label classifier for the tactics predictions
		pipeline_tactics = Pipeline([
				('columnselector', prp.TextSelector(key = 'processed')),
				('tfidf', TfidfVectorizer(tokenizer = prp.LemmaTokenizer(), stop_words = stop_words, max_df = 0.90)),
				('selection', SelectPercentile(chi2, percentile = 50)),
				('classifier', OneVsRestClassifier(LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = True, class_weight = 'balanced'), n_jobs = 1))
			])
		# train the model and predict the tactics
		pipeline_tactics.fit(train_reports, train_tactics)
		pred_tactics = pipeline_tactics.predict(test_reports)
		predprob_tactics = pipeline_tactics.decision_function(test_reports)
		
		if np.amin(predprob_tactics) < min_prob_tactics:
			min_prob_tactics = np.amin(predprob_tactics)
		if np.amax(predprob_tactics) > max_prob_tactics:
			max_prob_tactics = np.amax(predprob_tactics)
		
		if cmd:
			print_progress_bar(i)
		
		# Define a pipeline combining a text feature extractor with multi label classifier for the techniques predictions
		pipeline_techniques = Pipeline([
				('columnselector', prp.TextSelector(key = 'processed')),
				('tfidf', TfidfVectorizer(tokenizer = prp.StemTokenizer(), stop_words = stop_words, min_df = 2, max_df = 0.99)),
				('selection', SelectPercentile(chi2, percentile = 50)),
				('classifier', OneVsRestClassifier(LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = False, max_iter = 1000, class_weight = 'balanced'), n_jobs = 1))
			])
		# train the model and predict the techniques
		pipeline_techniques.fit(train_reports, train_techniques)
		pred_techniques = pipeline_techniques.predict(test_reports)
		predprob_techniques = pipeline_techniques.decision_function(test_reports)
		
		if np.amin(predprob_techniques) < min_prob_techniques:
			min_prob_techniques = np.amin(predprob_techniques)
		if np.amax(predprob_techniques) > max_prob_techniques:
			max_prob_techniques = np.amax(predprob_techniques)
		
		i+=2
		if cmd:
			print_progress_bar(i)
		
		# calculate the F0.5 score for each type of post processing and append to the list to keep track over the different folds
		f05_NO.append(fbeta_score(test_techniques, pred_techniques, beta = 0.5, average = 'macro'))
		f05_HN.extend(hanging_node_threshold_comparison(pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, test_techniques, permutations))
		
		i+=2
		if cmd:
			print_progress_bar(i)
		
		CPres, _ = confidence_propagation(predprob_tactics, pred_techniques, predprob_techniques)
		
		i+=2
		if cmd:
			print_progress_bar(i)
		
		f05_CP.append(fbeta_score(test_techniques, CPres, beta = 0.5, average = 'macro'))
		
		i+=2
	
	save_post_processing_comparison=[]
	# find the F0.5 average for each post-processing
	fb05_NO_avg = np.mean(f05_NO)
	fb05_CP_avg = np.mean(f05_CP)
	best_HN=[]
	fb05_Max_HN_avg = 0
	
	if cmd:
		print_progress_bar(48)

	for ps in permutations:
		sum = []
		for prhn in f05_HN:
			if ps == prhn[0]:
				sum.append(prhn[1])
		avg_temp = np.mean(sum)
		if avg_temp >= fb05_Max_HN_avg:
			fb05_Max_HN_avg = avg_temp
			best_HN = ps

	# define the best post-processing based on the F0.5 score average
	if fb05_NO_avg >= fb05_CP_avg and fb05_NO_avg >= fb05_Max_HN_avg:
		save_post_processing_comparison = ["N"]
	elif fb05_CP_avg >= fb05_Max_HN_avg and fb05_CP_avg >= fb05_NO_avg:
		save_post_processing_comparison = ["CP"]
	else:
		save_post_processing_comparison = ["HN"]
	save_post_processing_comparison.extend([best_HN, [min_prob_tactics, max_prob_tactics], [min_prob_techniques, max_prob_techniques]])
	
	# save the results
	joblib.dump(save_post_processing_comparison, "classification_tools/data/configuration.joblib")
	
	if cmd:
		print_progress_bar(50)
		print()
