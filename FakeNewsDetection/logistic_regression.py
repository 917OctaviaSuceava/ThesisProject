import pickle
from datetime import time

import pandas as pd
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import text_preprocessing
import time

nltk.download('punkt')
nltk.download('stopwords')

X_pkl = pickle.load(open("tfidf.pickle", "rb"))
Y_pkl = pickle.load(open("Y1.pickle", "rb"))

X1_pkl = pickle.load(open("tfidf1.pickle", "rb"))
Y1_pkl = pickle.load(open("Y2.pickle", "rb"))

X2_pkl = pickle.load(open("tfidf2.pickle", "rb"))
Y2_pkl = pickle.load(open("Y3.pickle", "rb"))

X, Y = X_pkl, Y_pkl
X1, Y1 = X1_pkl, Y1_pkl
X2, Y2 = X2_pkl, Y2_pkl
modelLR = LogisticRegression()


def logreg_tfidf(x, y):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=Y, random_state=2)
    modelLR.fit(X_train, Y_train)

    test_prediction = modelLR.predict(X_test)
    accuracy = accuracy_score(test_prediction, Y_test)
    precision = precision_score(Y_test, test_prediction)
    recall = recall_score(Y_test, test_prediction)
    f1 = f1_score(Y_test, test_prediction)

    return accuracy, precision, recall, f1


def train_first():
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    print("-----ONLY TEXT-----")
    for i in range(100):
        start_time = time.time()
        accuracy, precision, recall, f1 = logreg_tfidf(X, Y)
        end_time = time.time()
        print("finished iteration " + str(i) + "; elapsed time: " + str(end_time - start_time) + "s")
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    pickle.dump(modelLR, open("LR1.pickle", "wb"))
    print("Average Accuracy:", sum(accuracies) / len(accuracies))
    print("Average Precision:", sum(precisions) / len(precisions))
    print("Average Recall:", sum(recalls) / len(recalls))
    print("Average F1-score:", sum(f1_scores) / len(f1_scores))


def train_second():
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    print("-----TEXT AND AUTHOR-----")
    for i in range(100):
        start_time = time.time()
        accuracy, precision, recall, f1 = logreg_tfidf(X1, Y1)
        end_time = time.time()
        print("finished iteration " + str(i) + "; elapsed time: " + str(end_time - start_time) + "s")
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    pickle.dump(modelLR, open("LR2.pickle", "wb"))
    print("Average Accuracy:", sum(accuracies) / len(accuracies))
    print("Average Precision:", sum(precisions) / len(precisions))
    print("Average Recall:", sum(recalls) / len(recalls))
    print("Average F1-score:", sum(f1_scores) / len(f1_scores))


def train_third():
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    print("-----TEXT, TITLE AND AUTHOR-----")
    for i in range(100):
        start_time = time.time()
        accuracy, precision, recall, f1 = logreg_tfidf(X2, Y2)
        end_time = time.time()
        print("finished iteration " + str(i) + "; elapsed time: " + str(end_time - start_time) + "s")
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    pickle.dump(modelLR, open("LR2.pickle", "wb"))
    print("Average Accuracy:", sum(accuracies) / len(accuracies))
    print("Average Precision:", sum(precisions) / len(precisions))
    print("Average Recall:", sum(recalls) / len(recalls))
    print("Average F1-score:", sum(f1_scores) / len(f1_scores))
