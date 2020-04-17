from app import app
from flask import render_template, jsonify, abort, request
from app.suggest import SearchIndexUnweighted, SearchIndex
import os

@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Miguel'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]

    with open(os.environ.get("WORD_LIST", "wordlist.txt"), "r") as f:
        words = []
        for line in f:
            words.append(line.strip())

    return render_template('index.html', title='Home', user=user, posts=posts, wordlist=words)


@app.route("/suggest", methods=["GET", "POST"])
def suggest():
    if not request.form or "text" not in request.form:
        print(request.form)
        print(request.json)
        return abort(400)
    index = SearchIndex.get_instance()
    if len(request.form["text"]) <= 1:
        return jsonify({"text": ""})

    print(index.search(request.form["text"]))
    print(request.form)
    return jsonify({"text": index.search(request.form["text"])})
