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
		
		

		
		if(os.path.isfile('models/sms_spam_nb_model.pkl') == False):
			print()
			print("Creating Naive Bayes Model.....")
			train_multinomial_nb()

		print('Loading Pickle File...')

		
		nb_detector = cPickle.load(open('models/sms_spam_nb_model.pkl','rb'))
		nb = list(map(float,nb_detector)),10
		my_prediction = nb.predict([data])

	print('Results are available...')
	return render_template('result.html',prediction = my_prediction)


def train_multinomial_nb():
	print('Training the Model...')
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']

	

	msg_train, msg_test, label_train, label_test = train_test_split(X, y, test_size=0.2)

	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()

	nb_detector = clf.fit(msg_train, label_train)

	predictions = nb_detector.predict(msg_test)

	print(":: Confusion Matrix")
	print()
	print(confusion_matrix(label_test, predictions))
	print()
	print(":: Classification Report")
	print()
	print(classification_report(label_test, predictions))
	
	# save model to pickle file
	print('Creating Pickle file')
	file_name = 'models/sms_spam_nb_model.pkl'
	with open(file_name, 'wb') as fout:
		cPickle.dump(nb_detector, fout)
	print('model written to: ' + file_name)
	


if __name__ == '__main__':
	app.run(debug=True)