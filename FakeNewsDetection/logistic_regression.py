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
modelLR1 = LogisticRegression()
modelLR2 = LogisticRegression()


def logreg_tfidf(x, y, model):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=Y, random_state=2)
    model.fit(X_train, Y_train)

    test_prediction = model.predict(X_test)
    accuracy = accuracy_score(test_prediction, Y_test)
    precision = precision_score(Y_test, test_prediction)
    recall = recall_score(Y_test, test_prediction)
    f1 = f1_score(Y_test, test_prediction)

    return accuracy, precision, recall, f1


def train(val):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    print("-----LOGISTIC REGRESSION-----")
    for i in range(100):
        start_time = time.time()
        if val == 1:
            accuracy, precision, recall, f1 = logreg_tfidf(X, Y, modelLR)
        elif val == 2:
            accuracy, precision, recall, f1 = logreg_tfidf(X1, Y1, modelLR1)
        else:
            accuracy, precision, recall, f1 = logreg_tfidf(X2, Y2, modelLR2)
        end_time = time.time()
        print("finished iteration " + str(i) + "; elapsed time: " + str(end_time - start_time) + "s")
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    if val == 1:
        pickle.dump(modelLR, open("LR1.pickle", "wb"))
    elif val == 2:
        pickle.dump(modelLR1, open("LR2.pickle", "wb"))
    else:
        pickle.dump(modelLR2, open("LR3.pickle", "wb"))

    print("Average Accuracy:", sum(accuracies) / len(accuracies))
    print("Average Precision:", sum(precisions) / len(precisions))
    print("Average Recall:", sum(recalls) / len(recalls))
    print("Average F1-score:", sum(f1_scores) / len(f1_scores))
