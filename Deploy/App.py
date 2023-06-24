import torch
from torch import cuda
from transformers import RobertaTokenizer, DistilBertTokenizer
from Load_Model import *
from Preprocessing import Senti_Preproc
import torch.nn.functional as F
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gdown

## Check Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DistilBertClass() #RobertaClass()
model_name = "model_DiziRoBERTa.pt"

# Download model weight if it dosn't exist
if os.path.exists(f"model_weight/{model_name}"):
    print("model weight is exists.")
else:

    print("Model Dowenloading")
    # file_url = "https://drive.google.com/uc?id=1-2uh9i8sIxwW95KhCff3Bve8Ayz2ifeF" ROBERTA Model
    file_url = "https://drive.google.com/uc?id=1pibxdKOg4c4U24kyuTXIsOlvIGrLW6lp"
    destination_file_path = f"model_weight/{model_name}"  # Replace with the desired destination file path
    gdown.download(file_url, destination_file_path, quiet=False)


## Load the model and tokinzer
model.load_state_dict(torch.load(f'model_weight/{model_name}',  map_location=torch.device(device)))
Ro_tokenizer = DistilBertTokenizer.from_pretrained('Distiltokinizer')

## Make Model in Evalution states
model.eval()

## Test Preprocessing
Pre_porecessing = Senti_Preproc(Ro_tokenizer)

def Prediction(Model_input):
    with torch.no_grad():
        ids = Model_input['ids'].to(device, dtype = torch.long).unsqueeze(0)
        mask = Model_input['mask'].to(device, dtype = torch.long).unsqueeze(0)
        token_type_ids = Model_input['token_type_ids'].to(device, dtype=torch.long).unsqueeze(0)  
        outputs = model(ids, mask, token_type_ids)
        
        softmax_output = F.softmax(outputs, dim=1)
        N_propa = torch.sum(softmax_output[0,0:3]) .numpy().tolist()
        P_propa = torch.sum(softmax_output[0,3:]).numpy().tolist()

        return json.dumps({"Negative" : N_propa, "Positive" : P_propa})



## Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

cors = CORS(app, resources={r"*": {"origins": "*"}})




@app.route('/',  methods = ['POST'])
def sentiment_anaylsis():
    sentence = request.get_json()['sentence']
    Model_input = Pre_porecessing.toknize(sentence)
    response_body = Prediction(Model_input)

   
    
    return response_body


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

