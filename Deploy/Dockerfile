FROM python:3.9.15

RUN pip install --upgrade pip


# ADD App.py .
ADD dumy_App.py .
COPY model_weight .
COPY tokinizer2 .
COPY Distiltokinizer .
ADD requirements.txt .
ADD Load_Model.py .
ADD Preprocessing.py .


RUN pip install -r  requirements.txt
ARG FLASK_DEBUG="false"

EXPOSE 8000
# CMD ["python", "./dumy_App.py"]
CMD ["gunicorn", "dumy_App:app"]

