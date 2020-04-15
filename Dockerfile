FROM python:3.8-alpine

RUN adduser -D flaskuser

WORKDIR /home/flaskuser

COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt
RUN venv/bin/pip install gunicorn

COPY app app
#COPY migrations migrations
COPY flask_app.py boot.sh ./
#COPY flask_app.py config.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP flask_app.py

RUN chown -R flaskuser:flaskuser ./
USER flaskuser

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]