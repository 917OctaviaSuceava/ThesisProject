import pickle
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    if pd.isna(text):
        return None
    # only take into consideration non-null data entries
    if isinstance(text, str):
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


def text_only():
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('data/train.csv', na_filter=False)
    print(data.head())

    # preprocess the data
    preprocessed_data = data.text.apply(preprocess_text)

    tfidf_vectorizer = TfidfVectorizer()

    # fit the vectorizer to the preprocessed data
    tfidf_vectorizer.fit(preprocessed_data)

    # transform the preprocessed data into a TF-IDF matrix, only taking non-null entries
    tf_idf_matrix = tfidf_vectorizer.transform(preprocessed_data.dropna())

    pickle.dump(tf_idf_matrix, open("tfidf.pickle", "wb"))
    Y = data.label.values
    print(Y)
    return tf_idf_matrix, Y


def text_author():
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('data/train.csv', na_filter=False)
    print(data.head())

    # preprocess the data
    new_text = data.text.apply(preprocess_text)
    preprocessed_data = data.author + new_text

    tfidf_vectorizer = TfidfVectorizer()

    # fit the vectorizer to the preprocessed data
    tfidf_vectorizer.fit(preprocessed_data)

    # transform the preprocessed data into a TF-IDF matrix, only taking non-null entries
    tf_idf_matrix = tfidf_vectorizer.transform(preprocessed_data.dropna())

    pickle.dump(tf_idf_matrix, open("tfidf1.pickle", "wb"))


def text_title_author():
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('data/train.csv', na_filter=False)
    print(data.head())
    labels = data.label
    print(labels)

    # preprocess the data
    new_text = data.text.apply(preprocess_text)
    new_title = data.title.apply(preprocess_text)
    preprocessed_data = data.author + new_title + new_text

    tfidf_vectorizer = TfidfVectorizer()

    # fit the vectorizer to the preprocessed data
    tfidf_vectorizer.fit(preprocessed_data)

    # transform the preprocessed data into a TF-IDF matrix, only taking non-null entries
    tf_idf_matrix = tfidf_vectorizer.transform(preprocessed_data.dropna())

    pickle.dump(tf_idf_matrix, open("tfidf2.pickle", "wb"))


text_only()