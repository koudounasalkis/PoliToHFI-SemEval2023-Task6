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
    parser.add_argument('--output_filename', 
        help='Output filename',
        default="ensamble_predictions.csv",
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
  
    args = parser.parse_args()
    output_filename = args.output_filename
    strategy = args.strategy            # e.g., 'first'
    max_sentences = args.max_sentences  # e.g., 256
    config = json.load(open(args.config))

    # Load data
    single_test_embeddings = np.load(os.path.join("testData", "predict", "single_test_embeddings.npy"), allow_pickle=True)
    with open(os.path.join("testData", "predict", "single_test_doc_ids.txt"), "r") as f:
        test_ids = f.readlines()
    single_test_ids = [line.split(".")[0] for line in test_ids]
    multi_test_embeddings = np.load(os.path.join("testData", "predict", "multi_test_embeddings.npy"), allow_pickle=True)
    with open(os.path.join("testData", "predict", "multi_test_doc_ids.txt"), "r") as f:
        test_ids = f.readlines()
    multi_test_ids = [line.split(".")[0] for line in test_ids]

    # Create dataloaders
    single_test_ds = LJPESecondLevelClassificationDataset(single_test_embeddings, np.array([1]*len(single_test_embeddings)), max_sentences=max_sentences, strategy=strategy)
    single_test_dataloader = DataLoader(single_test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=16, pin_memory=True)
    multi_test_ds = LJPESecondLevelClassificationDataset(multi_test_embeddings, np.array([1]*len(multi_test_embeddings)), max_sentences=max_sentences, strategy=strategy)
    multi_test_dataloader = DataLoader(multi_test_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=16, pin_memory=True)

    # Create model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    single_embedding_size = single_test_embeddings[0].shape[1]
    multi_embedding_size = multi_test_embeddings[0].shape[1]
    checkpoint_folder_list = [
        #"second_level_results/second_level_train_single_last_2_3_5e-05",
        #"second_level_results/NEW_second_level_train_single_last_2_5_5e-05",
        #"second_level_results/NEW_second_level_train_single_last_2_7_5e-05",
        "second_level_results/second_level_train_multi_last_2_3_5e-05",
        "second_level_results/NEW_second_level_train_multi_last_2_5_5e-05",
        "second_level_results/NEW_second_level_train_multi_last_2_7_5e-05"
    ]

    all_predictions = []
    for checkpoint_folder in checkpoint_folder_list:
        checkpoint = torch.load(os.path.join(checkpoint_folder, "model.pt"))
        nlayers = int(checkpoint_folder.split("_")[-3])
        mlp_layers = int(checkpoint_folder.split("_")[-2])
        tp = "single" if "single" in checkpoint_folder else "multi"
        if tp == "single":
            embedding_size = single_embedding_size
            test_dataloader = single_test_dataloader
            test_ids = single_test_ids
        else:
            embedding_size = multi_embedding_size
            test_dataloader = multi_test_dataloader
            test_ids = multi_test_ids
        model = SecondLevelModel(d_model=embedding_size, d_hid=embedding_size, nhead=4, nlayers=nlayers, dropout=0, mlp_layers=mlp_layers)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        predicted = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                embeddings, attention_masks, labels = batch
                embeddings = embeddings.to(device)
                attention_masks = attention_masks.to(device)
                attention_masks = attention_masks.transpose(1, 0)
                output = model(embeddings, attention_masks)
                predicted.extend(output.cpu().detach().numpy())
        predicted = np.array(predicted)
        all_predictions.append(predicted)

    all_predictions = np.mean(all_predictions, axis=0) / len(checkpoint_folder_list)
    predicted = predicted > 0.5
    predicted = predicted.astype(int)

    df = pd.DataFrame()
    df["uid"] = test_ids
    df["prediction"] = list(np.squeeze(predicted, 1))
    df.to_csv(os.path.join("second_level_results", output_filename), index=False)


