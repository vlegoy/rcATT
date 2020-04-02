#!/usr/bin/python

##########################################################
#                      INTRODUCTION                      #
##########################################################
# rcATT is a tool to prediction tactics and techniques 
# from the ATT&CK framework, using multilabel text
# classification and post processing.
# Interface:  command-line
# Version:    1.00
# Author:     Valentine Legoy
# Date:       2019_10_22

import sys
import getopt
import joblib
import json
from shutil import copyfile
from colorama import init, Fore, Back, Style
from operator import itemgetter

import classification_tools.preprocessing as prp
import classification_tools.postprocessing as pop
import classification_tools.save_results as sr
import classification_tools as clt

#ignore Warnings, does not prevent the display of error
import warnings
warnings.simplefilter('ignore')
init(convert=True)


def correct_file(input_file, feedbacks, output_file):
	"""
	Correct results output by rcATT and save it in a file.
	"""
	with open(input_file) as f:
		data = json.load(f)
	report = data["description"]
	title = data["name"]
	date = data["published"][0:10]
	ttps = feedbacks.split(",") 
	if output_file!='':
		save_stix_file(report, title, date, ttps, output_file)
	else:
		save_stix_file(report, title, date, ttps, input_file)

def save_stix_file(report, title, date, ttps, output_file):
	"""
	Save prediction in a JSON file under STIX format
	"""
	if(date == ''):
		date = "1970-01-01"
	references = []
	for key in ttps:
		if key in clt.ALL_TTPS:
			references.append(clt.STIX_IDENTIFIERS[clt.ALL_TTPS.index(key)])
	file_to_save = sr.save_results_in_file(report, title, date, references)
	copyfile(file_to_save, output_file)

def save_train_set(input_file):
	"""
	Save JSON file output by rcATT to training set.
	"""
	with open(input_file) as f:
		data = json.load(f)
	refs = data["object_refs"]
	references = []
	for refid in range(len(clt.STIX_IDENTIFIERS)):
		if clt.STIX_IDENTIFIERS[refid] in refs:
			references.append(clt.ALL_TTPS[refid])
	sr.save_to_train_set(data["description"], references)

def predict(report_to_predict_file, output_file, title, date):
	"""
	Predict tactics and techniques from a report in a txt file.
	"""
	# parse text from file
	report_to_predict = ""
	with open(report_to_predict_file, 'r', newline = '', encoding = 'ISO-8859-1') as filetoread:
		data = filetoread.read()
		report_to_predict = prp.remove_u(data)
	
	# load postprocessingand min-max confidence score for both tactics and techniques predictions
	parameters = joblib.load("classification_tools/data/configuration.joblib")
	min_prob_tactics = parameters[2][0]	
	max_prob_tactics = parameters[2][1]
	min_prob_techniques = parameters[3][0]
	max_prob_techniques = parameters[3][1]
	
	pred_tactics, predprob_tactics, pred_techniques, predprob_techniques = clt.predict(report_to_predict, parameters)
	
	# change decision value into confidence score to display
	for i in range(len(predprob_tactics[0])):
		conf = (predprob_tactics[0][i] - min_prob_tactics) / (max_prob_tactics - min_prob_tactics)
		if conf < 0:
			conf = 0.0
		elif conf > 1:
			conf = 1.0
		predprob_tactics[0][i] = conf*100
	for j in range(len(predprob_techniques[0])):
		conf = (predprob_techniques[0][j] - min_prob_techniques) / (max_prob_techniques - min_prob_techniques)
		if conf < 0:
			conf = 0.0
		elif conf > 1:
			conf = 1.0
		predprob_techniques[0][j] = conf*100
	
	#prepare results to display
	ttps = []
	to_print_tactics = []
	to_print_techniques = []
	for ta in range(len(pred_tactics[0])):
		if pred_tactics[0][ta] == 1:
			ttps.append(clt.CODE_TACTICS[ta])
			to_print_tactics.append([1, clt.NAME_TACTICS[ta], predprob_tactics[0][ta]])
		else:
			to_print_tactics.append([0, clt.NAME_TACTICS[ta], predprob_tactics[0][ta]])
	for te in range(len(pred_techniques[0])):
		if pred_techniques[0][te] == 1:
			ttps.append(clt.CODE_TECHNIQUES[te])
			to_print_techniques.append([1, clt.NAME_TECHNIQUES[te], predprob_techniques[0][te]])
		else:
			to_print_techniques.append([0, clt.NAME_TECHNIQUES[te], predprob_techniques[0][te]])
	to_print_tactics = sorted(to_print_tactics, key = itemgetter(2), reverse = True)
	to_print_techniques = sorted(to_print_techniques, key = itemgetter(2), reverse = True)
	print("Predictions for the given report are : ")
	print("Tactics :")
	for tpta in to_print_tactics:
		if tpta[0] == 1:
			print(Fore.YELLOW + '' + tpta[1] + " : " + str(tpta[2]) + "% confidence")
		else:
			print(Fore.CYAN + '' + tpta[1] + " : " + str(tpta[2]) + "% confidence")
	print(Style.RESET_ALL)
	print("Techniques :")
	for tpte in to_print_techniques:
		if tpte[0] == 1:
			print(Fore.YELLOW + '' + tpte[1] + " : "+str(tpte[2])+"% confidence")
		else:
			print(Fore.CYAN + '' + tpte[1] + " : "+str(tpte[2])+"% confidence")
	print(Style.RESET_ALL)
	if output_file != '':
		save_stix_file(report_to_predict, title, date, ttps, output_file)
		print("Results saved in " + output_file)

def main(argv):
	input_file = ''
	output_file = ''
	added_file = 0
	added_feedback = ''
	title = ''
	date = ''
	pred = 0
	try:
		opts, args = getopt.getopt(argv,"htapf:i:o:n:d:",["help","train","add-to-training","predict","feedback=","input-file=","output-file=","report-title=","publishing-date="])
	except getopt.GetoptError:
		print('Python app to extract Att&ck tactics and techniques from cyber threat reports')
		print('type: <app.py -h> or <app.py --help> to see how to use this tool')
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("                                                      ")
			print(Fore.RED+"                        d8888 88888888888 88888888888 ")
			print("                       d88888     888         888     ")
			print("                      d88P888     888         888     ")
			print(" 888d888 .d8888b     d88P 888     888         888     ")
			print(" 888P\"  d88P\"       d88P  888     888         888     ")
			print(" 888    888        d88P   888     888         888     ")
			print(" 888    Y88b.     d8888888888     888         888     ")
			print(" 888     \"Y8888P d88P     888     888         888     ")
			print(Style.RESET_ALL)
			print(' rcATT is a python tool to predict ATT&CK tactics and techniques from cyber threat reports. Tactics and techniques displayed in yellow are predicted as included in the report. The percentage displayed next to the name of the tactic/technique is the likelihood of this tactic/technique of being in the report. If the tactic/technique is indicated as not being in the report, despite the displayed likelihood, it is due to the post-processing in our model. If you disagree with the prediction, you can correct these results and save them to the training set to improve it.')
			print(' ')
			print(' Commands : ')
			print(' \t-t   --train\t : petrain the tool with the newly added reports')
			print(' \t-p   --predict\t : predict TTPs for report in the input file')
			print(' \t-f   --feedback\t : change the results given by the tool in a previously output json file by a list of given TTPs')
			print(' \t-a   --add-to-training\t : add a json file output by the tool to the training set')
			print(' \t-i   --input-file\t : input file: .txt for --predict, .json for --feedback and --add-to-training (required)')
			print(' \t-o   --output-file\t : output file: json for --predict (if not given no results will be saved) and --feedback (if not given, changes will be saved in the input file)')
			print(' \t-n   --report-title\t : title of the report to add to the json file')
			print(' \t-d   --publishing-date\t : publishing date of the report to add to the json file (use the YYYY-MM-DD format)')
			print(' ')
			print(' Examples:')
			print(' \trcATT_cmd.py --train')
			print(' \trcATT_cmd.py -p -i input.txt -o input.json -n title -d 1970-01-01')
			print(' \trcATT_cmd.py -f TA0005,TA0003 -i input.json -o output.json')
			print(' \trcATT_cmd.py -a -i output.json')
			sys.exit()
		elif opt in ("-t", "--train"):
			print('Retraining the tool. This will take some time...')
			clt.train(True)
			print('Training finished!')
			sys.exit()
		elif opt in ("-f", "--feedback"):
			added_feedback = arg
		elif opt in ("-a", "--add-to-training"):
			added_file = 1
		elif opt in ("-p", "--predict"):
			pred = 1
		elif opt in ("-i", "--input-file"):
			input_file = arg
		elif opt in ("-o", "--output-file"):
			output_file = arg
		elif opt in ("-n", "--report-title"):
			title = arg
		elif opt in ("-d", "--publishing-date"):
			date = arg
	if input_file != '' and pred != 0:
		predict(input_file, output_file, title, date)
	if added_feedback != '' and input_file != '':
		print("Adding changes to selected results...")
		correct_file(input_file, added_feedback, output_file)
		if output_file != '':
			print("Change added and saved in " + output_file)
		else:
			print("Change added and saved in " + input_file)
	if added_file == 1 and input_file != '':
		print('Adding the file to the training set...')
		save_train_set(input_file)
		print(input_file + ' added to the training set!')

if __name__ == "__main__":
	main(sys.argv[1:])