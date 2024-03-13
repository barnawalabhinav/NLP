import nltk
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import gensim.downloader as api
from nltk.tokenize import word_tokenize


NUM_NEG_ROWS = 100
WORD2VEC = api.load("word2vec-google-news-300")

nltk.download('punkt')


class Flatten(nn.Module):
    def __init__(self, max_seq_len, token_size=300):
        super(Flatten, self).__init__()
        self.max_seq_len = max_seq_len
        self.cls = nn.Parameter(torch.randn(token_size, dtype=float))
        self.sep = nn.Parameter(torch.randn(token_size, dtype=float))
        self.seg_id = nn.Parameter(torch.randn(2, token_size, dtype=float))
        self.col_id = nn.Parameter(torch.randn(max_seq_len, token_size, dtype=float))
        self.rank_id = nn.Parameter(torch.randn(max_seq_len, token_size, dtype=float))
        self.pos = nn.Parameter(torch.randn(max_seq_len, token_size, dtype=float))
        self.row_id = nn.Parameter(torch.randn(max_seq_len, token_size, dtype=float))
        self.pad = nn.Parameter(torch.randn(token_size, dtype=float))

    def forward(self, data_file):
        sampled_data = []
        data = pd.read_json(data_file, lines=True)
        for line in data:
            question = word_tokenize(line['question'].lower())
            qid = line['qid']
            cols = line['table']['cols']
            rows = line['table']['rows']
            types = line['table']['types']
            caption = line['table']['caption']
            id = line['table']['id']
            label_col = line['label_col']
            label_cell = np.array(line['label_cell'])
            label_row = np.sort(np.array(line['label_row']))

            first_column = label_cell[:, 0].astype(int)
            sorted_index = np.argsort(first_column)
            label_cell = label_cell[sorted_index]

            # Class Token
            cls_token = self.cls + self.pos[0] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
            all_tokens = [cls_token]

            # Question Token
            index = 1
            for ques_tok in question:
                ques_tok = WORD2VEC.wv[ques_tok] + self.pos[index] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
                index += 1
                all_tokens.append(ques_tok)

            # Separator Token
            sep_token = self.sep + self.pos[index] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
            all_tokens.append(sep_token)

            # Table's column header Token
            col_cnt = 1
            for column in cols:
                col_tok = word_tokenize(column.lower())
                for tok in col_tok:
                    tok = WORD2VEC.wv[tok] + self.pos[index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[0] + self.rank_id[0]
                    index += 1
                    all_tokens.append(tok)
                col_cnt += 1

            # Sampling rows
            correct_rows = []
            incorrect_rows = []
            for row_id in range(len(rows)):
                if row_id in label_row:
                    correct_rows.append(row_id)
                else:
                    incorrect_rows.append(row_id)
            negative_rows = random.sample(incorrect_rows, min(NUM_NEG_ROWS, len(incorrect_rows)))

            # Table's correct row Token
            row_cnt = 1
            for row_id in correct_rows:
                label_row[row_cnt - 1] = row_id
                label_cell[row_cnt - 1][0] = row_id
                row = rows[row_id]
                col_cnt = 1
                for cell in row:
                    cell_tok = word_tokenize(cell.lower())
                    for tok in cell_tok:
                        tok = WORD2VEC.wv[tok] + self.pos[index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[row_cnt] + self.rank_id[0]
                        index += 1
                        all_tokens.append(tok)
                    col_cnt += 1
                row_cnt += 1

            # Table's incorrect row Token
            for row_id in negative_rows:
                row = rows[row_id]
                col_cnt = 0
                temp_tokens = []
                for cell in row:
                    cell_tok = word_tokenize(cell.lower())
                    for tok in cell_tok:
                        tok = WORD2VEC.wv[tok] + self.pos[index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[row_cnt] + self.rank_id[0]
                        index += 1
                        temp_tokens.append(tok)
                    col_cnt += 1
                if index > self.max_seq_len:
                    index -= len(temp_tokens)
                    break
                all_tokens.extend(temp_tokens)
                row_cnt += 1

            flat_data = {
                'qid': qid,
                'caption': caption,
                'id': id,
                'label_col': label_col,
                'label_cell': label_cell,
                'label_row': label_row,
                'index': index
            }

            # Padding Token
            pad_token = self.pad + self.pos[index] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
            while index < self.max_seq_len:
                all_tokens.append(pad_token)
                index += 1

            flat_data['tokens'] = torch.tensor(all_tokens)
            sampled_data.append(flat_data)

        df = pd.DataFrame(sampled_data)
        print(df)


flattener = Flatten(128)
flattener.forward("../data/A2_val.jsonl")