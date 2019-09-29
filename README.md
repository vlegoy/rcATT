
<img src="img/rcattlogo.png" alt="rcATT logo">

<p align="center">Reports Classification by Adversarial Tactics and Techniques</p>

<p align="center"><img src="https://img.shields.io/badge/made%20with-python-blue.svg" alt="made with python">   <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license MIT">    <!--<img src="https://img.shields.io/github/release-date/ValLGY/rcATT" alt="release date">--></p>

A python app to predict Att&ck tactics and techniques from cyber threat reports

## Usage

This tool is designed to predict tactics and techniques from the ATT&CK framework (https://attack.mitre.org/) in cyber threat reports, such as the ones that can be linked in https://otx.alienvault.com/ or https://exchange.xforce.ibmcloud.com/.

rcATT is useable either by a command-line interface or a graphical interface. Both versions have the same functionalities:
<ul>
  <li>predict tactics and techniques from a given cyber threat reports in a text format</li>
  <li>order and visualize the confidence of the classifier for each techniques and tactics, even the one predicted as non-included in the report</li>
  <li>save results in a json file in a STIX format</li>
  <li>give feedbacks to the tool by modifying the prediction to positive or negative</li>
  <li>save the feedbacks and/or the results to the training set</li>
  <li>retrain the classifier with new data</li>
</ul>

## Installation
This tool requires:
<ul>
  <li>python >= 3.5</li>
  <li>joblib</li>
  <li>flask</li>
  <li>pandas</li>
  <li>pickle</li>
  <li>numpy</li>
  <li>stix2</li>
  <li>sklearn</li>
  <li>nltk<ul><li>punkt</li><li>stopwords</li><li>wordnet</li></ul></li>
  <li>colorama</li>
</ul>
Then simply download the tool and run that app file with python.

## How it works
### Predict tactics and techniques from a given cyber threat reports in a text format
<img src="img/rcATTcmd.jpg" alt="rcATT command-line help">
<img src="img/rcattcmdres.gif" alt="rcATT command-line results">
<img src="img/rcATTgui.jpg" alt="rcATT GUI">

### Give feedbacks to the tool by modifying the prediction to positive or negative
<img src="img/rcattguichange.gif" alt="rcATT change results">

### Save the feedbacks and/or the results to the training set

### Retrain the classifier with new data

### Save results in a json file in a STIX format

<img src="img/rcATTgui2.jpg" alt="rcATT save in stix">
<img src="img/ExampleStix.jpg" alt="rcATT stix ouput">

## More details

This tool is the result of a Mater thesis on the prediction of tactics and techniques in cyber threat reports. You can find more details on this work in the following paper: [link to the paper]
