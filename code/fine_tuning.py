import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.utils.rnn as rnn
import csv
import pandas as pd
import os
import random
import numpy as np
from datetime import datetime
import argparse

from LSTM_kuroda import LSTMClassification
from get_dataset import googledrive_download, init_dataset

# GPUチェック
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model')
    parser.add_argument('--fine_tuned_model')
    return parser.parse_args()


def main():

    args = parse_arguments()

    pretrained_model_path = args.pretrained_model
    fine_tuned_model_path = args.fine_tuned_model

    # the number of players
    num_player = 22

    batch_size = 2048
    input_dim = (num_player + 1) * 2
    hidden_dim = 20
    target_size = 18
    num_epochs = 1000
    lr = 0.01

    # 各戦術的行動の名前 
    tactical_action_name_list = ['Build up 1', 'Progression 1', 'Final third 1', 'Counter-attack 1', 'High press 1', 'Mid block 1', 'Low block 1', 'Counter-press 1', 'Recovery 1', 'Build up 2', 'Progression 2', 'Final third 2', 'Counter-attack 2', 'High press 2', 'Mid block 2', 'Low block 2', 'Counter-press 2', 'Recovery 2']

    # numpy load
    sequence_np, label_np = googledrive_download(bepro=True) # _0_or_1, 
    print(sequence_np)
    print(sequence_np.shape, label_np.shape)

    label_np = label_np # [:, 1:]

    train_loader, val_loader, test_loader = init_dataset(sequence_np, label_np, batch_size)

    # モデルを定義
    model = LSTMClassification(input_dim=input_dim, hidden_dim=hidden_dim, target_size=target_size)
    
    # 学習済みモデルのロード
    model.load_state_dict(torch.load(pretrained_model_path), strict=False)

    model = fine_tuning(train_loader, val_loader, test_loader, model, num_epochs, lr)
    torch.save(model.state_dict(), fine_tuned_model_path)


def fine_tuning(train_loader, val_loader, test_loader, model, num_epochs, lr):
    
    # 最適化手法と損失関数の定義
    model = model.to(device)
    loss_function = nn.HuberLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # ファインチューニング
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()
            
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_function(predictions, labels.float())
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

        # 検証
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()
                predictions = model(inputs)
                loss = loss_function(predictions, labels.float())
                val_losses.append(loss.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")

    return model


if __name__ == "__main__":
    main()