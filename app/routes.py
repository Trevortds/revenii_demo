from app import app
from flask import render_template, jsonify, abort, request
from app.suggest import SearchIndexUnweighted, SearchIndex
from operator import itemgetter
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
    # print(request)
    # print(request.args)
    # print(request.form)
    # print(request.json)
    if not request.form or "text" not in request.form:
        print("error!")
        return abort(400)
    index = SearchIndex.get_instance()
    query = request.form["text"]
    if len(query) <= 1:
        return jsonify({"suggestions": ""})

    results = index.search(query)
    results = reweight_results(query, results)
    # print(results)
    return jsonify({"suggestions": [result[0] for result in results]})


def reweight_results(query, results):
    for pair in results:
        pair[1] += 10 * lenCommonPrefix([query, pair[0]])
    return sorted(results, key=itemgetter(1), reverse=True)


def lenCommonPrefix(strs):
    if not strs: 
        return 0
    shortest_str = min(strs, key=len)
    for i in range(len(shortest_str)):
        if all([x.startswith(shortest_str[:i+1]) for x in strs]):
            continue
        else:
            return i
    return len(shortest_str)
