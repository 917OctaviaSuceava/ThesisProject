import pickle

import pandas as pd
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import text_preprocessing
nltk.download('punkt')
nltk.download('stopwords')

# methods = [text_preprocessing.text_only(), text_preprocessing.text_author(), text_preprocessing.text_title_author()]

def logreg_tfidf():
    X, Y = text_preprocessing.text_only()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    train_prediction = model.predict(X_train)
    accuracy = accuracy_score(train_prediction, Y_train)
    precision = precision_score(Y_train, train_prediction)
    recall = recall_score(Y_train, train_prediction)
    f1 = f1_score(Y_train, train_prediction)
    print("-------TRAINING DATA RESULTS-------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    test_prediction = model.predict(X_test)
    accuracy = accuracy_score(test_prediction, Y_test)
    precision = precision_score(Y_test, test_prediction)
    recall = recall_score(Y_test, test_prediction)
    f1 = f1_score(Y_test, test_prediction)
    print("-------TEST DATA RESULTS-------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


def logreg_tfidf1():
    pass


def logreg_tfidf2():
    pass

logreg_tfidf()
