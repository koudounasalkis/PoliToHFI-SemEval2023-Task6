import os
import json
import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, RobertaTokenizerFast
from transformers import AutoModelForTokenClassification


############################################################
#                                                          #
#                        NER EXTRACTOR                     #
#                                                          #
############################################################
class NERExtractor:
    def __init__(self, ner_model_path, tokenizer, original_label_list):
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_model_path
        )
        self.ner_model
        self.ner_model.eval()
        self.tokenizer = tokenizer

        labels_list = ["B-" + l for l in original_label_list]
        labels_list += ["I-" + l for l in original_label_list]
        labels_list = sorted(labels_list + ["O"])[::-1]
        self.labels_to_idx = dict(
            zip(sorted(labels_list)[::-1], range(len(labels_list)))
        )
        print(self.labels_to_idx)
        self.idx_to_labels = {v[1]: v[0] for v in self.labels_to_idx.items()}

    ## Extract NER from text
    def extract_ner(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            verbose=False, 
            return_offsets_mapping=True
        )
        offset_mapping = inputs['offset_mapping'].squeeze(0).tolist()[1:-1]
        
        del inputs['offset_mapping']

        with torch.no_grad():
            logits = self.ner_model(**inputs).logits

        predicted_token_class_ids = logits.argmax(-1).squeeze(0).cpu().numpy().tolist()[1:-1]
        
        predictions = []
        for i, (offset, prediction) in enumerate(zip(offset_mapping, predicted_token_class_ids)):

            prediction = self.idx_to_labels[prediction].split('-')[-1]

            if prediction != "O":

                if i > 0:
                  prec_prediction = self.idx_to_labels[predicted_token_class_ids[i-1]].split('-')[-1]

                  if prediction == prec_prediction:
                      predictions[-1]['end'] = offset[1]
                  else:
                      predictions.append(
                          {
                              'label': prediction,
                              'start': offset[0],
                              'end': offset[1],
                          }
                      )
                else:
                  predictions.append(
                    {
                        'label': prediction,
                        'start': offset[0],
                        'end': offset[1],
                      }
                  )
        
        return predictions


############################################################
#                                                          #
#                        INFERENCE                         #
#                                                          #
############################################################                    

## Define the models to use with the corresponding checkpoint and tokenizer
base_dir = "results"
all_model_path = [
    (f'{base_dir}/bert-large-NER/checkpoint-65970',
    'dslim/bert-large-NER'),                    # ft on NER
    (f'{base_dir}/roberta-large-ner-english/checkpoint-65970',
    'Jean-Baptiste/roberta-large-ner-english'), # ft on NER
    (f'{base_dir}/nlpaueb/legal-bert-base-uncased/checkpoint-65970',
    'nlpaueb/legal-bert-base-uncased'),         # ft on Legal Domain
    (f'{base_dir}/saibo/legal-roberta-base/checkpoint-65970',
    'saibo/legal-roberta-base'),                # ft on Legal Domain
    (f'{base_dir}/nlpaueb/bert-base-uncased-eurlex/checkpoint-65970',
    'nlpaueb/bert-base-uncased-eurlex'),        # ft on Eurlex
    (f'{base_dir}/nlpaueb/bert-base-uncased-echr/checkpoint-65970',
    'nlpaueb/bert-base-uncased-echr'),          # ft on ECHR
    (f'{base_dir}/studio-ousia/luke-base/checkpoint-65970',
    'studio-ousia/luke-base'),                  # LUKE base
    (f'{base_dir}/studio-ousia/luke-large/checkpoint-65970',
    'studio-ousia/luke-large'),                 # LUKE large
]

## Loop over the models
for model_path in sorted(all_model_path):

    ## Load the test data
    test_data = 'data/NER_TEST/NER_TEST_DATA_FS.json'
    data = json.load(open(test_data)) 

    ## Load the tokenizer
    tokenizer_path = model_path[1]
    if 'luke' in model_path[0]: 
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) 
    
    ## Define the labels list
    ll = [
            "COURT",
            "PETITIONER",
            "RESPONDENT",
            "JUDGE",
            "DATE",
            "ORG",
            "GPE",
            "STATUTE",
            "PROVISION",
            "PRECEDENT",
            "CASE_NUMBER",
            "WITNESS",
            "OTHER_PERSON",
            "LAWYER"
    ]

    ## Initialize the NER extractor
    ner_extr = NERExtractor(
        ner_model_path = model_path[0], 
        tokenizer = tokenizer, 
        original_label_list=ll)
    
    print(model_path)
    print(tokenizer)

    ## Extract NER from the test data
    for i in tqdm(range(len(data))):

        text = data[i]['data']['text']
        source = data[i]['meta']['source']
        
        results = ner_extr.extract_ner(text)
        
        results_output = []
        for j, r in enumerate(results):
            o = {
                "value": {
                    "start": r['start'],
                    "end": r['end'],
                    "text": text[r['start']:r['end']],
                    "labels": [r['label']]
                },
                "id": f"{i}-{j}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels"
            }
            results_output.append(o)
        data[i]['annotations'][0]['result'] = results_output
    
    ## Save the results
    json.dump(data, open(f'{base_dir}/all/{model_path[0].split("/")[-2]}_predictions.json', 'w'))