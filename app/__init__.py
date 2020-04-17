from flask import Flask
from flask_bootstrap import Bootstrap
import threading
from app import suggest
import os
import random

app = Flask(__name__)

bootstrap = Bootstrap(app)


def index_builder():
    print("building index")
    with open(os.environ.get("WORD_LIST", "wordlist.txt"), "r") as f:
        words = []
        weights = []
        for line in f:
            words.append(line.strip())
            weights.append(random.randint(0, 10))

    index = suggest.SearchIndex.get_instance(words, weights)
    print("done building index")


thread = threading.Thread(index_builder())
thread.start()

from app import routes
