FROM python:3.8-alpine


RUN apk update
RUN apk add build-base libzmq musl-dev
RUN apk add make automake gcc g++ subversion

RUN adduser -D flaskuser

WORKDIR /home/flaskuser

COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt
RUN venv/bin/pip install gunicorn

COPY app app
#COPY migrations migrations
COPY flask_app.py boot.sh wordlist.txt ./
#COPY flask_app.py config.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP flask_app.py
ENV WORD_LIST /home/flaskuser/wordlist.txt

RUN chown -R flaskuser:flaskuser ./
USER flaskuser

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]