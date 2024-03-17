import os
import math
import time
import torch
import numpy as np
import pandas as pd
from torch import nn, Tensor
from torch.utils.data import DataLoader

from model import TransformerModel

LR = 0.1
N_HEAD = 2
D_HID = 300
N_LAYERS = 2
D_MODEL = 300
DROPOUT = 0.5
BATCH_SIZE = 32
MAX_SEQ_LEN = 256
SCHED_GAMMA = 0.9
SCHED_STEP_SIZE = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def split_dataframe_into_batches(df, batch_size):
    num_batches = (len(df) + batch_size - 1) // batch_size
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        batches.append(batch_df)
    return batches


def train(data_file):
    data = pd.read_json(data_file, lines=True)
    batch_list = split_dataframe_into_batches(data, BATCH_SIZE)
    
    model = TransformerModel(max_seq_len=MAX_SEQ_LEN, d_model=D_MODEL, nhead=N_HEAD, d_hid=D_HID, nlayers=N_LAYERS, dropout=DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHED_STEP_SIZE, gamma=SCHED_GAMMA)
    
    model.train()
    total_loss = 0.0
    start = time.time()
    
    for i, batch in enumerate(batch_list):
        output, target = model(batch, return_target=True)
        print(output.shape())
        print(target.shape())
        break



if __name__=='__main__':
    # train("data/A2_train.jsonl")
    train("data/A2_val.jsonl")