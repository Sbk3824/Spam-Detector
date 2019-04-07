# Imports
import pandas as pd
import os, sys, getopt, csv, sklearn
import pickle as cPickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score

from textblob import TextBlob


# Dataset
STOP = set(stopwords.words('english'))
MESSAGES = pd.read_csv('./Data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])

# Preprocessing
def split_tokens(message):
    message = str(message, 'utf8')
    return TextBlob(message).words

def split_lemmas(message):
    message = str(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

def split_stopwords(message):
    message = str(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words if word not in STOP]

# Training
def train_multinomial_nb(messages):
    # split dataset for cross validation

    msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)
    # create pipeline
    pipeline = Pipeline([('bow', CountVectorizer(analyzer=split_stopwords)),('tfidf', TfidfTransformer()),('classifier', MultinomialNB())])
    # pipeline parameters to automatically explore and tune
    params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_lemmas, split_tokens, split_stopwords),
    }
    grid = GridSearchCV(
        pipeline,
        params, # parameters to tune via cross validation
        refit=True, # fit using all data, on the best detected classifier
        n_jobs=-1,
        scoring='accuracy',
        cv=StratifiedKFold(label_train, n_folds=5),
    )
    # train
    nb_detector = grid.fit(msg_train, label_train)
    print()
    predictions = nb_detector.predict(msg_test)
    print(":: Confusion Matrix")
    print()
    print(confusion_matrix(label_test, predictions))
    print( "")
    print(":: Classification Report")
    print()
    print(classification_report(label_test, predictions))
    # save model to pickle file
    file_name = 'models/sms_spam_nb_model.pkl'
    with open(file_name, 'wb') as fout:
        cPickle.dump(nb_detector, fout)
    print('model written to: ' + file_name)

def train_svm(messages):
    # split dataset for cross validation
    msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)
    # create pipeline
    pipeline = Pipeline([('bow', CountVectorizer(analyzer=split_stopwords)),('tfidf', TfidfTransformer()),('classifier', SVC())])
    # pipeline parameters to automatically explore and tune
    params = [
        {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
        {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
    ]
    grid = GridSearchCV(
        pipeline,
        param_grid=params, # parameters to tune via cross validation
        refit=True, # fit using all data, on the best detected classifier
        n_jobs=-1,
        scoring='accuracy',
        cv=StratifiedKFold(label_train, n_folds=5),
    )
    # train
    svm_detector = grid.fit(msg_train, label_train)
    print( "")
    print(":: Confusion Matrix")
    print()
    print(confusion_matrix(label_test, svm_detector.predict(msg_test)))
    print()
    print()
    print()
    print(classification_report(label_test, svm_detector.predict(msg_test)))
    # save model to pickle file
    file_name = 'models/sms_spam_svm_model.pkl'
    with open(file_name, 'wb') as fout:
        cPickle.dump(svm_detector, fout)
    print('model written to: ' + file_name)


def main(argv):
  # check if models exist, if not run training
   if(os.path.isfile('models/sms_spam_nb_model.pkl') == False):
     print()
     print("Creating Naive Bayes Model.....")
     train_multinomial_nb(MESSAGES)

   if(os.path.isfile('models/sms_spam_svm_model.pkl') == False):
     print()
     print("Creating SVM Model.....")
     train_svm(MESSAGES)

   inputmessage = ''
   try:
      opts, args = getopt.getopt(argv,"hm:",["message="])
   except getopt.GetoptError:
      print('sms-spam-detector.py -m <message string>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('sms-spam-detector.py -m <message string>')
         sys.exit()
      elif opt in ("-m", "--message"):
         prediction = predict(arg)
   print('This message is predicted by', prediction)

def predict(message):
  nb_detector = cPickle.load(open('models/sms_spam_nb_model.pkl'))
  svm_detector = cPickle.load(open('models/sms_spam_svm_model.pkl'))

  nb_predict = nb_detector.predict([message])[0]
  svm_predict = svm_detector.predict([message])[0]

  return 'SVM as ' + svm_predict + ' and Naive Bayes as ' + nb_predict

if __name__ == "__main__":
   main(sys.argv[1:])

