import json
from flask import Flask, request, jsonify
from flask_cors import CORS




## Test Preprocessing
def Prediction(Model_input):
    return json.dumps({"Negative" : 0.5, "Positive" : 0.5})



## Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

cors = CORS(app, resources={r"*": {"origins": "*"}})




@app.route('/',  methods = ['POST'])
def sentiment_anaylsis():
    sentence = request.get_json()['sentence']
    response_body = Prediction(sentence)

   
    
    return response_body


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000, debug=True)

