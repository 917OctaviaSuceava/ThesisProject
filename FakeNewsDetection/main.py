import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)


@app.route("/members")
def members():
    return {"members": ["m1", "m2", "m3"]}


@app.route('/process_text', methods=['POST'])
def process_text():
    text = request.get_json().get('text')  # Extract the text from the request

    # Process the text and generate a response
    response = 'Message received and processed on the server'
    print(text)

    return jsonify({'response': response})


if __name__ == "__main__":
    app.run(debug=True)
    # decision_trees.train(3)
    # model_DT = pickle.load(open("DT1.pickle", "rb"))
    # tfidf = pickle.load(open('tfidf.pickle', 'rb'))
    # print(tfidf.shape)
    # passive_aggressive.train(3)
    # decision_trees.build_tree(model_DT, 1)
    # logistic_regression.train(3)
