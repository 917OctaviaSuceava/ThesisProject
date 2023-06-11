import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import text_preprocessing

logistic_regression_model_1 = pickle.load(open("D:\\ThesisProject\\FakeNewsDetection\\LR1.pickle", "rb"))

decision_trees_model_1 = pickle.load(open("D:\\ThesisProject\\FakeNewsDetection\\DT1.pickle", "rb"))

passive_aggressive_model_1 = pickle.load(open("D:\\ThesisProject\\FakeNewsDetection\\PA1.pickle", "rb"))

tfidf_vectorizer_1 = pickle.load(open("D:\\ThesisProject\\FakeNewsDetection\\tfidf_vectorizer.pickle", "rb"))


test_data = pd.read_csv('D:\\ThesisProject\\FakeNewsDetection\\data\\test.csv', na_filter=False)
test_data['preprocessed_text'] = test_data['text'].apply(text_preprocessing.preprocess_text)

test_features = test_data['preprocessed_text']

test_matrix = tfidf_vectorizer_1.transform(test_features)

predictions_1 = logistic_regression_model_1.predict(test_matrix)
predictions_2 = decision_trees_model_1.predict(test_matrix)
predictions_3 = passive_aggressive_model_1.predict(test_matrix)

# compare the predicted labels with the actual labels from submit.csv
submit_data = pd.read_csv('D:\\ThesisProject\\FakeNewsDetection\\data\\submit.csv')
actual_labels = submit_data['label']

# calculate evaluation metrics (accuracy, precision, recall, F1-score)
accuracy_1 = accuracy_score(actual_labels, predictions_1)
precision_1 = precision_score(actual_labels, predictions_1)
recall_1 = recall_score(actual_labels, predictions_1)
f1_1 = f1_score(actual_labels, predictions_1)
print("======= LOGISTIC REGRESSION =======")
print("Accuracy:", accuracy_1)
print("Precision:", precision_1)
print("Recall:", recall_1)
print("F1-score:", f1_1)


accuracy_2 = accuracy_score(actual_labels, predictions_2)
precision_2 = precision_score(actual_labels, predictions_2)
recall_2 = recall_score(actual_labels, predictions_2)
f1_2 = f1_score(actual_labels, predictions_2)
print("======= DECISION TREES =======")
print("Accuracy:", accuracy_2)
print("Precision:", precision_2)
print("Recall:", recall_2)
print("F1-score:", f1_2)


accuracy_3 = accuracy_score(actual_labels, predictions_3)
precision_3 = precision_score(actual_labels, predictions_3)
recall_3 = recall_score(actual_labels, predictions_3)
f1_3 = f1_score(actual_labels, predictions_3)
print("======= PASSIVE AGGRESSIVE CLASSIFIER =======")
print("Accuracy:", accuracy_3)
print("Precision:", precision_3)
print("Recall:", recall_3)
print("F1-score:", f1_3)
