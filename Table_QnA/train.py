import math
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from model import TransformerModel

LR = 0.1
N_HEAD = 2
D_HID = 300
N_LAYERS = 2
D_MODEL = 300
DROPOUT = 0.5
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_SEQ_LEN = 256
LOG_INTERVAL = 10
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
    num_batches = len(batch_list)
    
    model = TransformerModel(max_seq_len=MAX_SEQ_LEN, d_model=D_MODEL, nhead=N_HEAD, d_hid=D_HID, nlayers=N_LAYERS, dropout=DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHED_STEP_SIZE, gamma=SCHED_GAMMA)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for epoch in NUM_EPOCHS:
        for i in range(len(batch_list)):
            output, target = model.forward(batch_list[i], return_target=True)
            # print(output.shape)
            # print(target.shape)
            output = output.squeeze(1)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if i % LOG_INTERVAL == 0 and i > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / LOG_INTERVAL
                cur_loss = total_loss / LOG_INTERVAL
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()


if __name__=='__main__':
    # train("data/A2_train.jsonl")
    train("data/A2_temp.jsonl")