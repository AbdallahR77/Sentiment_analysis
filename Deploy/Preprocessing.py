import nltk
import re
import string
import torch

nltk.download('stopwords')
from nltk.corpus import stopwords


class Senti_Preproc():
    def __init__(self, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        stop_words = set(stopwords.words('english'))
        self.negation_words = ["didn", "weren", "doesn't"," hasn't", "couldn",
                            "mustn't", "isn", "hadn't", "isn't", "wasn't", "mightn't",
                            "couldn't", "needn't", "haven't", "shan't", "wouldn't",
                            "not", "no", "never", "none", "nobody", "nowhere", "nothing","isn'", "'t",
                            "shouldn't", "aren't", "didn't", "didn'", "don't", "hadn't", "won't", "wasn", "hadn"]
        self.stop_wrods = [ word for word in stop_words if word not in   self.negation_words]



        
    def preprocess_text(self, text):
        # Remove unnecessary white spaces
        text = " ".join(text.split())

        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', text)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
         
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        splited_text = text.split()
        cleaned_text = []
        for word in splited_text:
            if word not in self.stop_wrods:
                cleaned_text.append(word)
            elif word in self.negation_words:
                cleaned_text.append("not")
        text  = " ".join(cleaned_text)
        return text

    def toknize(self, text):
        text = self.preprocess_text(text)
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
        return {
        'ids': torch.tensor(ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)}
