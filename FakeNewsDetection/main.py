import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from FakeNewsDetection.text_preprocessing import preprocess_text

app = Flask(__name__)
cors = CORS(app)

logistic_regression_model_1 = pickle.load(open("LR1.pickle", "rb"))
logistic_regression_model_2 = pickle.load(open("LR2.pickle", "rb"))
logistic_regression_model_3 = pickle.load(open("LR3.pickle", "rb"))

decision_trees_model_1 = pickle.load(open("DT1.pickle", "rb"))
decision_trees_model_2 = pickle.load(open("DT2.pickle", "rb"))
decision_trees_model_3 = pickle.load(open("DT3.pickle", "rb"))

passive_aggressive_model_1 = pickle.load(open("PA1.pickle", "rb"))
passive_aggressive_model_2 = pickle.load(open("PA2.pickle", "rb"))
passive_aggressive_model_3 = pickle.load(open("PA3.pickle", "rb"))


@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    title = data.get('title')
    author = data.get('author')
    text = data.get('text')

    print("TITLE: " + title)
    print("AUTHOR: " + author)
    print("TEXT: " + text)

    lr_accuracy_1, dt_accuracy_1, pa_accuracy_1 = preprocess_predict_text(text)
    lr_accuracy_3 = ''
    dt_accuracy_3 = ''
    pa_accuracy_3 = ''
    lr_accuracy_2 = ''
    dt_accuracy_2 = ''
    pa_accuracy_2 = ''
    if title == '' and author != '':
        lr_accuracy_2, dt_accuracy_2, pa_accuracy_2 = preprocess_predict_text_author(text)
    elif title != '' and author != '':
        lr_accuracy_2, dt_accuracy_2, pa_accuracy_2 = preprocess_predict_text_author(text)
        lr_accuracy_3, dt_accuracy_3, pa_accuracy_3 = preprocess_predict_text_author_title(text, title)

    response = {
        'lr_accuracy_1': lr_accuracy_1.item(),
        'dt_accuracy_1': dt_accuracy_1.item(),
        'pa_accuracy_1': pa_accuracy_1.item(),
    }

    if lr_accuracy_2 != '':
        response['lr_accuracy_2'] = lr_accuracy_2.item()
    if dt_accuracy_2 != '':
        response['dt_accuracy_2'] = dt_accuracy_2.item()
    if pa_accuracy_2 != '':
        response['pa_accuracy_2'] = pa_accuracy_2.item()

    if lr_accuracy_3 != '':
        response['lr_accuracy_3'] = lr_accuracy_3.item()
    if dt_accuracy_3 != '':
        response['dt_accuracy_3'] = dt_accuracy_3.item()
    if pa_accuracy_3 != '':
        response['pa_accuracy_3'] = pa_accuracy_3.item()
    print(lr_accuracy_1, lr_accuracy_2, lr_accuracy_3, dt_accuracy_1, dt_accuracy_2, dt_accuracy_3,
          pa_accuracy_1, pa_accuracy_2, pa_accuracy_3)

    return jsonify(response)


def preprocess_predict_text(input_text):
    preprocessed_text = preprocess_text(input_text)
    if not preprocessed_text:
        return '', '', ''

    # load the TF-IDF vectorizer
    tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pickle", "rb"))

    # transform the preprocessed text into a TF-IDF matrix
    tf_idf_matrix = tfidf_vectorizer.transform([preprocessed_text])

    predictionLR = logistic_regression_model_1.predict(tf_idf_matrix)
    predictionDT = decision_trees_model_1.predict(tf_idf_matrix)
    predictionPA = passive_aggressive_model_1.predict(tf_idf_matrix)
    return predictionLR[0], predictionDT[0], predictionPA[0]


def preprocess_predict_text_author(input_text):
    preprocessed_text = preprocess_text(input_text)
    if not preprocessed_text:
        return '', '', ''

    # load the TF-IDF vectorizer
    tfidf_vectorizer = pickle.load(open("tfidf_vectorizer1.pickle", "rb"))

    # transform the preprocessed text into a TF-IDF matrix
    tf_idf_matrix = tfidf_vectorizer.transform([preprocessed_text])
    predictionLR = logistic_regression_model_2.predict(tf_idf_matrix)
    predictionDT = decision_trees_model_2.predict(tf_idf_matrix)
    predictionPA = passive_aggressive_model_2.predict(tf_idf_matrix)
    return predictionLR[0], predictionDT[0], predictionPA[0]


def preprocess_predict_text_author_title(input_text, input_title):
    preprocessed_text = preprocess_text(input_text + input_title)
    if not preprocessed_text:
        return '', '', ''

    # load the TF-IDF vectorizer
    tfidf_vectorizer = pickle.load(open("tfidf_vectorizer2.pickle", "rb"))

    # transform the preprocessed text into a TF-IDF matrix
    tf_idf_matrix = tfidf_vectorizer.transform([preprocessed_text])
    predictionLR = logistic_regression_model_3.predict(tf_idf_matrix)
    predictionDT = decision_trees_model_3.predict(tf_idf_matrix)
    predictionPA = passive_aggressive_model_3.predict(tf_idf_matrix)
    return predictionLR[0], predictionDT[0], predictionPA[0]


if __name__ == "__main__":
    app.run(debug=True)
    # decision_trees.train(3)
    # model_DT = pickle.load(open("DT1.pickle", "rb"))
    # tfidf = pickle.load(open('tfidf.pickle', 'rb'))
    # print(tfidf.shape)
    # passive_aggressive.train(3)
    # decision_trees.build_tree(model_DT, 1)
    # logistic_regression.train(3)
