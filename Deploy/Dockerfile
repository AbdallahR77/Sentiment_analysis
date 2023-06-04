FROM python:3.9.15

RUN pip install --upgrade pip

ADD App.py .
ADD model_weight ./model_weight
ADD tokinizer2 .
ADD requirements.txt .
ADD Load_Model.py .
ADD Preprocessing.py .


RUN pip install -r  requirements.txt
EXPOSE 8000

CMD ["python", "./App.py"]
