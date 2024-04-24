import sys
import math
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from model import *


torch.manual_seed(367546)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(LOG_FILE, 'w') as f:
    f.write(str(device) + '\n')


def split_dataframe_into_batches(data_file, batch_size, target=False):
    df = pd.read_json(data_file, lines=True)

    # ####################################
    # TODO: Write code to predict a column first and then predict rows in that column.
    # ####################################

    # for data_point, line in df.iterrows():
    #     question = line['question']
    #     qid = line['qid']
    #     col_header = line['table']['cols']
    #     rows = line['table']['rows']
    #     types = line['table']['types']
    #     caption = line['table']['caption']
    #     id = line['table']['id']
    #     if target:
    #         label_col = line['label_col']
    #         label_cell = np.array(line['label_cell'])
    #         label_row = np.sort(np.array(line['label_row']))

    #         first_column = label_cell[:, 0].astype(int)
    #         sorted_index = np.argsort(first_column)
    #         label_cell = label_cell[sorted_index]

    #         correct_rows = []
    #         incorrect_rows = []
    #         for row_id in range(len(rows)):
    #             if row_id in label_row:
    #                 correct_rows.append(row_id)
    #             else:
    #                 incorrect_rows.append(row_id)

    num_batches = (len(df) + batch_size - 1) // batch_size
    batches = []
    for batch_id in range(num_batches):
        start_idx = batch_id * batch_size
        end_idx = min((batch_id + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        batches.append(batch_df)
    return batches


def train(train_data_file, val_data_file, save_path="model.pth", show_val_loss=False):

    LR = 1.0
    N_HEAD = 2
    D_HID = 512
    N_LAYERS = 1
    D_MODEL = 300
    DROPOUT = 0.1
    BATCH_SIZE = 100
    NUM_EPOCHS = 50
    MAX_SEQ_LEN = 256
    LOG_INTERVAL = 10
    SCHED_GAMMA = 0.95
    SCHED_STEP_SIZE = 1

    train_batch_list = split_dataframe_into_batches(train_data_file, BATCH_SIZE, target=True)
    num_batches = len(train_batch_list)

    val_batch_list = split_dataframe_into_batches(val_data_file, BATCH_SIZE, target=True)
    
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
            # with open(LOG_FILE, 'a') as f:
            #     f.write(str(output) + '\n')
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            total_norm = 0
            for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                total_norm += p.grad.data.norm(2).item() ** 2
            total_norm **= 0.5
            with open(LOG_FILE, 'a') as f:
                f.write(str(('Grad Norm:', total_norm)) + '\n')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch_id % LOG_INTERVAL == 0 and batch_id > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / LOG_INTERVAL
                train_loss = total_loss / LOG_INTERVAL
                train_ppl = math.exp(train_loss)
                with open(LOG_FILE, 'a') as f:
                    f.write(str(f'| epoch {epoch:3d} | {batch_id:5d}/{num_batches:5d} batches | '
                        f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                        f'Train loss {train_loss:5.2f} | train ppl {train_ppl:8.2f}\n'))
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

                    val_loss /= num_data_points
                    lr = scheduler.get_last_lr()[0]
                    val_ppl = math.exp(val_loss)
                    with open(LOG_FILE, 'a') as f:
                        f.write(str(f'| Val loss {val_loss:5.2f} | Val ppl {val_ppl:8.2f}\n'))

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
        with open(LOG_FILE, 'a') as f:
            f.write(str(f'| epoch {epoch:3d} | lr {lr:02.2f} | '
                f'Val loss {val_loss:5.2f} | Val ppl {ppl:8.2f}\n'))


def train_col(train_data_file, val_data_file, save_path="model_col.pth", show_val_loss=False):

    LR = 1.0
    BATCH_SIZE = 250
    NUM_EPOCHS = 100
    LOG_INTERVAL = 10
    SCHED_GAMMA = 0.95
    SCHED_STEP_SIZE = 1

    train_batch_list = split_dataframe_into_batches(train_data_file, BATCH_SIZE, target=True)
    num_batches = len(train_batch_list)

    val_batch_list = split_dataframe_into_batches(val_data_file, BATCH_SIZE, target=True)
    
    model = TransformerModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHED_STEP_SIZE, gamma=SCHED_GAMMA)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        for batch_id in range(len(train_batch_list)):
            output, embed_src = model.forward(train_batch_list[batch_id], return_target=True)
            # output = output.squeeze(2)
            # with open(LOG_FILE, 'a') as f:
            #     f.write(str(output) + '\n')
            target = torch.stack([*embed_src['target'].values]).to(device)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            total_norm = 0
            for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                total_norm += p.grad.data.norm(2).item() ** 2
            total_norm **= 0.5
            with open(LOG_FILE, 'a') as f:
                f.write(str(('Grad Norm:', total_norm)) + '\n')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch_id % LOG_INTERVAL == 0 and batch_id > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / LOG_INTERVAL
                train_loss = total_loss / LOG_INTERVAL
                train_ppl = math.exp(train_loss)
                with open(LOG_FILE, 'a') as f:
                    f.write(str(f'| epoch {epoch:3d} | {batch_id:5d}/{num_batches:5d} batches | '
                        f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                        f'Train loss {train_loss:5.9f} | train ppl {train_ppl:8.2f}\n'))
                total_loss = 0
                start_time = time.time()

                torch.save(model.state_dict(), save_path)

                if show_val_loss:
                    val_loss = 0.0
                    num_data_points = 0
                    with torch.no_grad():
                        for batch_id in range(len(val_batch_list)):
                            output, embed_src = model.forward(val_batch_list[batch_id], return_target=True)
                            # output = output.squeeze(2)
                            target = torch.stack([*embed_src['target'].values]).to(device)
                            loss = criterion(output, target)
                            # index_to_col = embed_src['index_to_col']
                            col_to_indices = embed_src['col_to_indices']
                            col_header = embed_src['col_header']
                            prob = 0
                            col = 0
                            for col_id in len(col_header):
                                col_prob = 0
                                for ind in col_to_indices[col_id]:
                                    col_prob = max(col_prob, target[ind])
                                if col_prob > prob:
                                    prob = col_prob
                                    col = col_id

                            val_loss += loss.item()
                            num_data_points += 1
                            pred = {
                                'label_col': [col_header[col]],
                                'label_cell': [[]],
                                'label_row': [],
                                'qid': embed_src['qid']
                            }

                    val_loss /= num_data_points
                    lr = scheduler.get_last_lr()[0]
                    val_ppl = math.exp(val_loss)
                    with open(LOG_FILE, 'a') as f:
                        f.write(str(f'| Val loss {val_loss:5.2f} | Val ppl {val_ppl:8.2f}\n'))

        scheduler.step()

        val_loss = 0.0
        num_data_points = 0
        with torch.no_grad():
            for batch_id in range(len(val_batch_list)):
                output, embed_src = model.forward(val_batch_list[batch_id], return_target=True)
                # output = output.squeeze(2)
                target = torch.stack([*embed_src['target'].values]).to(device)
                loss = criterion(output, target)
                val_loss += loss.item()
                num_data_points += 1

        val_loss /= num_data_points
        lr = scheduler.get_last_lr()[0]
        ppl = math.exp(val_loss)
        with open(LOG_FILE, 'a') as f:
            f.write(str(f'| epoch {epoch:3d} | lr {lr:02.2f} | '
                f'Val loss {val_loss:5.2f} | Val ppl {ppl:8.2f}\n'))



if __name__=='__main__':
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    # train("data/A2_train.jsonl", "data/A2_val.jsonl", save_path="model_3.pth", show_val_loss=False)
    train_col(train_data_file=train_file, val_data_file=val_file, save_path="model_col_1.pth", show_val_loss=False)