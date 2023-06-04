import torch
from torch import cuda
from transformers import RobertaTokenizer
from Load_Model import RobertaClass
from preprocessing import Senti_Preproc
import torch.nn.functional as F
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gdown

## Check Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RobertaClass()

# Download model weight if it dosn't exist
if os.path.exists("model_weight/model_RoBERTa2.pt"):
    print("model weight is exists.")
else:
    print("File does not exist.")

## Load the Toknizer and the Model weights 
if device =="cpu": # map the weights to CPU if there is no GPU
    model.load_state_dict(torch.load('model_weight/model_RoBERTa2.pt',  map_location=torch.device('cpu')))
else :
    print("Model Dowenloading")
    file_url = "https://drive.google.com/uc?id=1-2uh9i8sIxwW95KhCff3Bve8Ayz2ifeF"
    destination_file_path = "model_weight/model_RoBERTa2.pt"  # Replace with the desired destination file path
    gdown.download(file_url, destination_file_path, quiet=False)
    

Ro_tokenizer = RobertaTokenizer.from_pretrained('tokinizer2')

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

