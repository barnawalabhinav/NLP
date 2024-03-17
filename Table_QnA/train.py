import math
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from model import TransformerModel

LR = 1.0
N_HEAD = 2
D_HID = 300
N_LAYERS = 2
D_MODEL = 300
DROPOUT = 0.1
BATCH_SIZE = 125
NUM_EPOCHS = 10
MAX_SEQ_LEN = 256
LOG_INTERVAL = 10
SCHED_GAMMA = 0.95
SCHED_STEP_SIZE = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def split_dataframe_into_batches(data_file, batch_size):
    df = pd.read_json(data_file, lines=True)
    num_batches = (len(df) + batch_size - 1) // batch_size
    batches = []
    for batch_id in range(num_batches):
        start_idx = batch_id * batch_size
        end_idx = min((batch_id + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        batches.append(batch_df)
    return batches


def train(train_data_file, val_data_file, save_path="model.pth", show_val_loss=False):
    train_batch_list = split_dataframe_into_batches(train_data_file, BATCH_SIZE)
    num_batches = len(train_batch_list)

    val_batch_list = split_dataframe_into_batches(val_data_file, BATCH_SIZE)
    
    model = TransformerModel(max_seq_len=MAX_SEQ_LEN, d_model=D_MODEL, nhead=N_HEAD, d_hid=D_HID, nlayers=N_LAYERS, dropout=DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHED_STEP_SIZE, gamma=SCHED_GAMMA)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        for batch_id in range(len(train_batch_list)):
            output, target = model.forward(train_batch_list[batch_id], return_target=True)
            # output = output.squeeze(2)
            # print(output)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch_id % LOG_INTERVAL == 0 and batch_id > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / LOG_INTERVAL
                train_loss = total_loss / LOG_INTERVAL
                train_ppl = math.exp(train_loss)
                print(f'| epoch {epoch:3d} | {batch_id:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'Train loss {train_loss:5.2f} | train ppl {train_ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

                torch.save(model.state_dict(), save_path)

                if show_val_loss:
                    val_loss = 0.0
                    num_data_points = 0
                    with torch.no_grad():
                        for batch_id in range(len(val_batch_list)):
                            output, target = model.forward(val_batch_list[batch_id], return_target=True)
                            # output = output.squeeze(2)
                            loss = criterion(output, target)
                            val_loss += loss.item()
                            num_data_points += 1

                    print(num_data_points)
                    val_loss /= num_data_points
                    lr = scheduler.get_last_lr()[0]
                    val_ppl = math.exp(val_loss)
                    print(f'| Val loss {val_loss:5.2f} | Val ppl {val_ppl:8.2f}')

        scheduler.step()

        val_loss = 0.0
        num_data_points = 0
        with torch.no_grad():
            for batch_id in range(len(val_batch_list)):
                output, target = model.forward(val_batch_list[batch_id], return_target=True)
                # output = output.squeeze(2)
                loss = criterion(output, target)
                val_loss += loss.item()
                num_data_points += 1

        val_loss /= num_data_points
        lr = scheduler.get_last_lr()[0]
        ppl = math.exp(val_loss)
        print(f'| epoch {epoch:3d} | lr {lr:02.2f} | '
            f'Val loss {val_loss:5.2f} | Val ppl {ppl:8.2f}')


if __name__=='__main__':
    train("data/A2_train.jsonl", "data/A2_val.jsonl", save_path="model_1.pth", show_val_loss=True)