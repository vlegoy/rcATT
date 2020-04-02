##########################################################
#                      INTRODUCTION                      #
##########################################################
# rcATT is a tool to prediction tactics and techniques 
# from the ATT&CK framework, using multilabel text
# classification and post processing.
# Interface:  graphical
# Version:    1.00
# Author:     Valentine Legoy
# Date:       2019_10_22

from flask import Flask,render_template,url_for,request, send_file
import joblib
import re

import classification_tools.preprocessing as prp
import classification_tools.postprocessing as pop
import classification_tools.save_results as sr
import classification_tools as clt

from operator import itemgetter


#Starts the GUI tool on Flask
app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')
	
@app.route('/save',methods=['POST'])
def save():
	"""
	Save predictions either in the training set or in a JSON file under STIX format.
	"""
	if request.method == 'POST':
		formdict = request.form.to_dict()
		save_type1 = "filesave"
		save_type2 = "trainsave"
		#save to a JSON file in STIX format
		if save_type1 in formdict:
			references = []
			for key, value in formdict.items():
				if key in clt.ALL_TTPS:
					references.append(clt.STIX_IDENTIFIERS[clt.ALL_TTPS.index(key)])
			file_to_save = sr.save_results_in_file(re.sub("\r\n", " ", request.form['hidereport']), request.form['name'], request.form['date'], references)
			return send_file(file_to_save, as_attachment=True)
		#save in the custom training set
		if save_type2 in formdict:
			references = []
			for key, value in formdict.items():
				if key in clt.ALL_TTPS:
					references.append(key)
			sr.save_to_train_set(re.sub("\r\n", "\t", prp.remove_u(request.form['hidereport'].encode('utf8').decode('ISO-8859-1'))), references)
	return ('', 204)

@app.route('/',methods=['POST'])
def retrain():
	"""
	Train the classifier again based on the new data added by the user.
	"""
	if request.method == 'POST':
		clt.train(False)
	return ('', 204)


@app.route('/predict',methods=['POST'])
def predict():
	"""
	Predict the techniques and tactics for the report entered by the user.
	"""

	report_to_predict = ""
	
	if request.method == 'POST':
		report_to_predict = prp.remove_u(request.form['message'].encode('utf8').decode('ISO-8859-1'))
		
		# load postprocessing and min-max confidence score for both tactics and techniques predictions
		parameters = joblib.load("classification_tools/data/configuration.joblib")
		min_prob_tactics = parameters[2][0]	
		max_prob_tactics = parameters[2][1]
		min_prob_techniques = parameters[3][0]
		max_prob_techniques = parameters[3][1]
	
		pred_tactics, predprob_tactics, pred_techniques, predprob_techniques = clt.predict(report_to_predict, parameters)
	
		# change decision value into confidence score to display and prepare results to display
		pred_to_display_tactics = []
		for i in range(len(predprob_tactics[0])):
			conf = (predprob_tactics[0][i] - min_prob_tactics) / (max_prob_tactics - min_prob_tactics)
			if conf < 0:
				conf = 0.0
			elif conf > 1:
				conf = 1.0
			pred_to_display_tactics.append([clt.CODE_TACTICS[i], clt.NAME_TACTICS[i], pred_tactics[0][i], conf*100])
		pred_to_display_techniques = []
		for j in range(len(predprob_techniques[0])):
			conf = (predprob_techniques[0][j] - min_prob_techniques) / (max_prob_techniques - min_prob_techniques)
			if conf < 0:
				conf = 0.0
			elif conf > 1:
				conf = 1.0
			pred_to_display_techniques.append([clt.CODE_TECHNIQUES[j], clt.NAME_TECHNIQUES[j], pred_techniques[0][j], conf*100])
		pred_to_display_tactics = sorted(pred_to_display_tactics, key = itemgetter(3), reverse = True)
		pred_to_display_techniques = sorted(pred_to_display_techniques, key = itemgetter(3), reverse = True)
	return render_template('result.html', report = request.form['message'], predictiontact = pred_to_display_tactics, predictiontech = pred_to_display_techniques)



if __name__ == '__main__':
	app.run(debug = True)