from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
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

modelDT = DecisionTreeClassifier()
modelDT1 = DecisionTreeClassifier()
modelDT2 = DecisionTreeClassifier()


def decision_tree_tfidf(x, y, model):
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
    print("-----DECISION TREES-----")
    for i in range(100):
        start_time = time.time()
        if val == 1:
            accuracy, precision, recall, f1 = decision_tree_tfidf(X, Y, modelDT)
        elif val == 2:
            accuracy, precision, recall, f1 = decision_tree_tfidf(X1, Y1, modelDT1)
        else:
            accuracy, precision, recall, f1 = decision_tree_tfidf(X2, Y2, modelDT2)
        end_time = time.time()
        print("finished iteration " + str(i) + "; elapsed time: " + str(end_time - start_time) + "s")
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    if val == 1:
        pickle.dump(modelDT, open("DT1.pickle", "wb"))
        # build_tree(modelDT, 1)
    elif val == 2:
        pickle.dump(modelDT1, open("DT2.pickle", "wb"))
        # build_tree(modelDT1, 2)
    else:
        pickle.dump(modelDT2, open("DT3.pickle", "wb"))
        # build_tree(modelDT2, 3)

    print("Average Accuracy:", sum(accuracies) / len(accuracies))
    print("Average Precision:", sum(precisions) / len(precisions))
    print("Average Recall:", sum(recalls) / len(recalls))
    print("Average F1-score:", sum(f1_scores) / len(f1_scores))


def build_tree(model, x):
    columns = ['id', 'title', 'author', 'text']
    labels = ['0', '1']
    file_to_save = "decision_tree_" + str(x) + ".png"
    dot_data = export_graphviz(model, out_file=None, feature_names=columns, class_names=labels, filled=True,
                               rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(file_to_save)



def get_x_y():
    return X, Y
