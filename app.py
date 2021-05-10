
#  write anaconda coce her
#  FLASK RUN

import pandas as pd #for reading csv data and importing as data frame
import numpy as np
import matplotlib.pyplot as plt
import nltk 
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import os.path
import stopword    
filename = 'finalized_model.h5'
loaded_model = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))


#  if not rum
# PY -M VENV ENV
# env/Scripts/activate
# set FLASK_APP=app.PY
# flask run
#  for debug mode on
# export:FLASK_ENV="development"
def RegExpTokenizer(Sent):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(Sent)
def Lemmatizing_Words(Sent):
    Lm = WordNetLemmatizer()
    Lemmatized_Words = []
    for word in Sent:
        Lemmatized_Words.append(Lm.lemmatize(word))
    return Lemmatized_Words
def UnTokenizer(Sent):
    rejoin = TreebankWordDetokenizer()
    rejoined_sent = rejoin.detokenize(Sent)
    return rejoined_sent

from flask import Flask, render_template, url_for, request, jsonify
app = Flask(__name__)
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'sentiment_analysis'

# mysql = MySQL(app)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/analyse", methods=['POST'])
def analyse():
    # if request.method == "POST":
    # this below gives code  request.form['review']
    details = str(request.form['review'])
    review_sent = RegExpTokenizer(str(details))
    review_sent = stopword.Eliminate_Stop_Words(review_sent)
    review_sent = Lemmatizing_Words(review_sent)
    review_sent = UnTokenizer(review_sent)
    testingwords_vectors = vectorizer.transform([details])
    output =loaded_model.predict(testingwords_vectors)
    if output == '__label__2' :
       data = 'positive'
    if output == '__label__1' :
        data = 'negative'
    #return review
    # cur = mysql.connection.cursor()
    # cur.execute(
    #     "INSERT INTO sentiment_analysis review VALUES (%s)", review)
    # mysql.connection.commit()
    # cur.close()
    # return 'success'
#  REPACE BEOW DETAIL BY OUTPUT DATA
    return render_template('reviewed.html', review=data)


if __name__ == "__main__":
    app.run(debug=True)
