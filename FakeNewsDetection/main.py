import pickle

from matplotlib import pyplot as plt
from sklearn import tree
import logistic_regression
import decision_trees
import passive_aggressive

if __name__ == "__main__":
    # decision_trees.train(2)
    # model_DT = pickle.load(open("DT1.pickle", "rb"))
    # tfidf = pickle.load(open('tfidf.pickle', 'rb'))
    # print(tfidf.shape)
    passive_aggressive.train(3)
    # decision_trees.build_tree(model_DT, 1)
    # logistic_regression.train(3)
