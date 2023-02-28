'''
python second_level_train.py --data_type single --strategy last --max_sentences 256 --attention_layers 3 --mlp_layers 3
'''

import json
from code.second_level_dataset import LJPESecondLevelClassificationDataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from architecture.second_level_model import SecondLevelModel
import torch
from tqdm import tqdm
import numpy as np
import os
import sklearn.metrics as metrics 
import pandas as pd

if __name__ == "__main__":

    parser = ArgumentParser(description='Training of second level (document level) model')
    parser.add_argument('--data_type', 
        help='type of train data', 
        default="single", 
        choices=["single", "multi"],
        required=False, 
        type=str)
    parser.add_argument('--checkpoint_folder', 
        help='Model folder', 
        default="second_level_results/", 
        required=False, 
        type=str)
    parser.add_argument('--config', 
        help='Parameter config', 
        default="config/second_level_config.json", 
        required=False, 
        type=str)
    parser.add_argument('--strategy',
        help='Strategy for training',
        default="last",
        choices=["first", "last"],
        required=False,
        type=str)
    parser.add_argument('--max_sentences',
        help='max number of sentences to consider at second level of hierarchy',
        default=256,
        required=False,
        type=int)
    parser.add_argument('--attention_layers',
        help='number of attention layers',
        default=3,
        required=False,
        type=int)
    parser.add_argument('--mlp_layers',
        help='number of attention layers',
        default=3,
        required=False,
        type=int)
  
    args = parser.parse_args()

    data_type = args.data_type          # e.g., 'single'
    checkpoint_folder = args.checkpoint_folder  # e.g., './results'
    strategy = args.strategy            # e.g., 'first'
    max_sentences = args.max_sentences  # e.g., 256
    attention_layers = args.attention_layers  # e.g., 3
    mlp_layers = args.mlp_layers  # e.g., 3
    config = json.load(open(args.config))

    # Load data
    test_embeddings = np.load(os.path.join("testData", "predict", data_type+"_test_embeddings.npy"), allow_pickle=True)
    with open(os.path.join("testData", "predict", data_type+"_test_doc_ids.txt"), "r") as f:
        test_ids = f.readlines()
    test_ids = [line.split(".")[0] for line in test_ids]

    # Create dataloaders
    test_ds = LJPESecondLevelClassificationDataset(test_embeddings, np.array([1]*len(test_embeddings)), max_sentences=max_sentences, strategy=strategy)
    test_dataloader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=16, pin_memory=True)
    #test_dataloader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    embedding_size = test_embeddings[0].shape[1]
    checkpoint = torch.load(os.path.join(checkpoint_folder, "model.pt"))
    model = SecondLevelModel(d_model=embedding_size, d_hid=embedding_size, nhead=4, nlayers=attention_layers, dropout=0, mlp_layers=mlp_layers)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    total_loss = 0
    predicted = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            embeddings, attention_masks, labels = batch

            #################

            e = embeddings[0, :, :]
            e = e.unsqueeze(0).repeat(64, 1, 1)
            print(e)
            print(e.shape)

            a = attention_masks[0, :]
            a = a.unsqueeze(0).repeat(64, 1)
            a = a.transpose(1, 0)
            print(a.shape)

            output = model(e, a)
            print(output)

            exit()

            #################


            embeddings = embeddings.to(device)
            attention_masks = attention_masks.to(device)
            attention_masks = attention_masks.transpose(1, 0)
            output = model(embeddings, attention_masks)
            predicted.extend(output.cpu().detach().numpy())

    predicted = np.array(predicted)
    predicted = predicted > 0.5
    predicted = predicted.astype(int)

    df = pd.DataFrame()
    df["uid"] = test_ids
    df["prediction"] = list(np.squeeze(predicted, 1))
    df.to_csv(os.path.join(checkpoint_folder, "predictions.csv"), index=False)


