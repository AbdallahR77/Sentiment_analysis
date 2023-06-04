import requests
file_url = "https://drive.google.com/uc?id=1-2uh9i8sIxwW95KhCff3Bve8Ayz2ifeF"
destination_file_path = "model_weight/model_RoBERTa2sdsd.pt"  # Replace with the desired destination file path

import gdown

gdown.download(file_url, destination_file_path, quiet=False)