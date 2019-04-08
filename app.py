from flask import Flask,render_template,url_for,request
import pandas as pd 
import os, sys, getopt, csv, pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle as cPickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score


STOP = set(stopwords.words('english'))



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	"""Predictions"""
	print('Message Received....')
	if request.method == 'POST':
		#cv = CountVectorizer()
		#cv._validate_vocabulary()
		message = request.form['message']
		data = [message]
		
		my_prediction = train_multinomial_nb(data)

	return render_template('result.html',prediction = my_prediction) 

def train_multinomial_nb(data):

	print('Training the Model...')
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']

	cv = CountVectorizer()
	X = cv.fit_transform(X)

	msg_train, msg_test, label_train, label_test = train_test_split(X, y, test_size=0.2)

	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()

	nb_detector = clf.fit(msg_train, label_train)
	predictions = clf.predict(msg_test)

	vect = cv.transform(data).toarray()

	my_prediction = nb_detector.predict(vect)
	
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
	
	print('Accuracy score: ', format(accuracy_score(label_test, predictions)))
	print('Precision score: ', format(precision_score(label_test, predictions)))
	print('Recall score: ', format(recall_score(label_test, predictions)))
	print('F1 score: ', format(f1_score(label_test, predictions)))
	
	return my_prediction


if __name__ == '__main__':
	app.run(debug=True)