FROM python:3.8
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install fastapi==0.78.0 uvicorn==0.17.6 elasticsearch==8.2.0 slackweb==1.0.5 mecab-python3==0.996.5 numpy==1.22.4 gensim==3.8.3
ENTRYPOINT [ "python3", "main.py" ]