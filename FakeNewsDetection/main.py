import pickle

import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    if pd.isna(text):
        return None
    # only take into consideration non-null data entries
    if isinstance(text, str):
        # print('Before lowercase:', text)
        # convert text to lowercase
        text = text.lower()
        # remove punctuation => no characters should be replaced, the punctuation should be removed
        text = text.translate(str.maketrans('', '', string.punctuation))
        # remove numbers
        text = re.sub(r'\d+', '', text)
        # tokenize
        tokens = word_tokenize(text)
        # remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # create new string from tokens
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('data/train.csv', na_filter=False)
    print(data.head())
    labels = data.label
    print(labels)

    # preprocess the data
    preprocessed_data = data.text.apply(preprocess_text)

    tfidf_vectorizer = TfidfVectorizer()
    # fit the vectorizer to the preprocessed data
    tfidf_vectorizer.fit(preprocessed_data)
    pickle.dump(tfidf_vectorizer, open("file.pickle", "wb"))
    # transform the preprocessed data into a TF-IDF matrix, only taking non-null entries
    tf_idf_matrix = tfidf_vectorizer.transform(preprocessed_data.dropna())

    # transform labels into numerical values
    label_encoder = preprocessing.LabelEncoder()
    # fit the encoder
    encoded_labels = label_encoder.fit_transform(data['label'])
    # replace the labels in the dataframe with the encoded ones
    data['label'] = encoded_labels

    print(data.head())

    # y_df = data['label']  # targets

    # print(tf_idf_matrix)
    # split the data into training-testing sets
    x_train, x_test, y_train, y_test = train_test_split(tf_idf_matrix, encoded_labels, random_state=0)

    # train a logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # make predictions on the testing set
    y_pred = model.predict(x_test)

    # convert numerical labels back to original
    predicted_labels = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
