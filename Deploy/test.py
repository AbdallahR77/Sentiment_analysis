import torch
from torch import cuda
from transformers import RobertaTokenizer
from Load_Model import RobertaClass
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
model = RobertaClass()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)


torch.save(quantized_model, 'Model_qun_model_RoBERTa3.pt')