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

if __name__ == "__main__":

    parser = ArgumentParser(description='Training of second level (document level) model')
    parser.add_argument('--data_type', 
        help='type of train data', 
        default="single", 
        choices=["single", "multi"],
        required=False, 
        type=str)
    parser.add_argument('--output_folder', 
        help='Output folder', 
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
    output_folder = args.output_folder  # e.g., './results'
    strategy = args.strategy            # e.g., 'first'
    max_sentences = args.max_sentences  # e.g., 256
    attention_layers = args.attention_layers  # e.g., 3
    mlp_layers = args.mlp_layers  # e.g., 3
    config = json.load(open(args.config))

    # Load data
    train_embeddings = np.load(os.path.join("trainData", data_type + "_embeddings_train.npy"), allow_pickle=True)
    train_labels = np.load(os.path.join("trainData", data_type + "_labels_train.npy"), allow_pickle=True)
    val_embeddings = np.load(os.path.join("trainData", data_type + "_embeddings_val.npy"), allow_pickle=True)
    val_labels = np.load(os.path.join("trainData", data_type + "_labels_val.npy"), allow_pickle=True)
    
    # Create dataloaders
    train_ds = LJPESecondLevelClassificationDataset(train_embeddings, train_labels, strategy=strategy, max_sentences=max_sentences)
    train_dataloader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=16, pin_memory=True) 
    valid_ds = LJPESecondLevelClassificationDataset(val_embeddings, val_labels)
    valid_dataloader = DataLoader(valid_ds, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=16, pin_memory=True)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = torch.from_numpy(train_embeddings[0]).shape[1]
    model = SecondLevelModel(d_model=embedding_size, d_hid=embedding_size, nhead=4, nlayers=attention_layers, dropout=0.25, mlp_layers=mlp_layers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'])
    
    # create log file
    output_folder = os.path.join(output_folder, f"NEW_second_level_train_{data_type}_{strategy}_{attention_layers}_{mlp_layers}_{config['LR']}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log_file = os.path.join(output_folder, "log.txt")
    with open(log_file, "w") as f:
        f.write("Start training..\n")


    best_f1 = 0
    best_epoch = 0

    for epoch in range(config['EPOCHS']):
        print(f"Epoch {epoch}")

        # train model
        model.train()
        total_loss = 0
        predicted = []
        gt = []
        for batch in tqdm(train_dataloader):
            embeddings, attention_masks, labels = batch
            embeddings = embeddings.to(device)
            attention_masks = attention_masks.to(device)
            attention_masks = attention_masks.transpose(1, 0)
            output = model(embeddings, attention_masks)
            predicted.extend(output.cpu().detach().numpy())
            gt.extend(labels.cpu().detach().numpy())
            labels = labels.to(device).float()
            loss = torch.nn.functional.binary_cross_entropy(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        predicted = np.array(predicted)
        gt = np.array(gt)
        f1_macro = metrics.f1_score(gt, predicted > 0.5, average='macro')
        f1 = metrics.f1_score(gt, predicted > 0.5, average=None)
        print(f"Train F1 score: {f1_macro} - {f1}")

        total_loss /= len(train_dataloader)
        print(f"Total train loss: {total_loss}")

        # write on log file
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}\n\tTrain F1 score: {f1_macro} - {f1} - Total train loss: {total_loss}\n")

        # evaluate model
        model.eval()
        total_loss = 0
        predicted = []
        gt = []
        with torch.no_grad():
            for batch in tqdm(valid_dataloader):
                embeddings, attention_masks, labels = batch
                embeddings = embeddings.to(device)
                attention_masks = attention_masks.to(device)
                attention_masks = attention_masks.transpose(1, 0)
                output = model(embeddings, attention_masks)
                predicted.extend(output.cpu().detach().numpy())
                gt.extend(labels.cpu().detach().numpy())
                labels = labels.to(device).float()
                loss = torch.nn.functional.binary_cross_entropy(output, labels)
                total_loss += loss.item()
            
        predicted = np.array(predicted)
        gt = np.array(gt)
        
        accuracy = metrics.accuracy_score(gt, predicted > 0.5)
        precision = metrics.precision_score(gt, predicted > 0.5, average="macro")
        recall = metrics.recall_score(gt, predicted > 0.5, average="macro")
        f1_macro = metrics.f1_score(gt, predicted > 0.5, average='macro')
        f1 = metrics.f1_score(gt, predicted > 0.5, average=None)
        print(f"Validation accuracy: {accuracy}")
        print(f"Validation precision: {precision}")
        print(f"Validation recall: {recall}")
        print(f"Validation F1 score: {f1_macro} - {f1}")

        total_loss /= len(valid_dataloader)
        print(f"Total val loss: {total_loss}")

        # write on log file
        with open(log_file, "a") as f:
            f.write(f"\tValidation accuracy: {accuracy} - Validation precision: {precision} - Validation recall: {recall} - Validation F1 score: {f1_macro} - {f1} - Total val loss: {total_loss}\n")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_folder, "model.pt"))

    print(f"Best F1 score: {best_f1} - Best epoch: {best_epoch}")
    with open(log_file, "a") as f:
        f.write("\nTRAIN FINISHED!\n")
        f.write(f"Best F1 score: {best_f1} - Best epoch: {best_epoch}\n")
