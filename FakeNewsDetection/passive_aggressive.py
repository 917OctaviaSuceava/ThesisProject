from sklearn import tree
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
import pydotplus

X_pkl = pickle.load(open("tfidf.pickle", "rb"))
Y_pkl = pickle.load(open("Y1.pickle", "rb"))

X1_pkl = pickle.load(open("tfidf1.pickle", "rb"))
Y1_pkl = pickle.load(open("Y2.pickle", "rb"))

X2_pkl = pickle.load(open("tfidf2.pickle", "rb"))
Y2_pkl = pickle.load(open("Y3.pickle", "rb"))

X, Y = X_pkl, Y_pkl
X1, Y1 = X1_pkl, Y1_pkl
X2, Y2 = X2_pkl, Y2_pkl

modelPA = PassiveAggressiveClassifier()
modelPA1 = PassiveAggressiveClassifier()
modelPA2 = PassiveAggressiveClassifier()


def passive_aggressive_tfidf(x, y, model):
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
    print("-----PASSIVE AGGRESSIVE CLASSIFIER-----")
    for i in range(100):
        start_time = time.time()
        if val == 1:
            accuracy, precision, recall, f1 = passive_aggressive_tfidf(X, Y, modelPA)
        elif val == 2:
            accuracy, precision, recall, f1 = passive_aggressive_tfidf(X1, Y1, modelPA1)
        else:
            accuracy, precision, recall, f1 = passive_aggressive_tfidf(X2, Y2, modelPA2)
        end_time = time.time()
        print("finished iteration " + str(i) + "; elapsed time: " + str(end_time - start_time) + "s")
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    if val == 1:
        pickle.dump(modelPA, open("PA1.pickle", "wb"))
    elif val == 2:
        pickle.dump(modelPA1, open("PA2.pickle", "wb"))
    else:
        pickle.dump(modelPA2, open("PA3.pickle", "wb"))

    print("Average Accuracy:", sum(accuracies) / len(accuracies))
    print("Average Precision:", sum(precisions) / len(precisions))
    print("Average Recall:", sum(recalls) / len(recalls))
    print("Average F1-score:", sum(f1_scores) / len(f1_scores))

