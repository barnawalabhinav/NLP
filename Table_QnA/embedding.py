import nltk
import torch
import random
import hashlib
import numpy as np
import pandas as pd
from torch import nn, tensor
import gensim.downloader as api
from nltk.tokenize import word_tokenize


NUM_NEG_ROWS = 4
WORD2VEC = api.load("word2vec-google-news-300")

nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Custom_Embedding(nn.Module):
    def __init__(self, max_seq_len, token_size=300):
        super(Custom_Embedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.cls = nn.Parameter(torch.randn(token_size, dtype=float))
        self.sep = nn.Parameter(torch.randn(token_size, dtype=float))
        self.seg_id = nn.Parameter(torch.randn(2, token_size, dtype=float))
        self.col_id = nn.Parameter(torch.randn(max_seq_len, token_size, dtype=float))
        self.rank_id = nn.Parameter(torch.randn(max_seq_len, token_size, dtype=float))
        self.pos = nn.Parameter(torch.randn(max_seq_len, token_size, dtype=float))
        self.row_id = nn.Parameter(torch.randn(max_seq_len, token_size, dtype=float))
        self.pad = nn.Parameter(torch.randn(token_size, dtype=float))

    def str_to_vec(self, str, dim=300):
        byte_repr = str.encode()
        hash_value = hashlib.sha256(byte_repr).digest()
        hash_array = np.frombuffer(hash_value, dtype=np.uint8)[:dim]
        if len(hash_array) < dim:
            hash_array = np.pad(hash_array, (0, dim - hash_array.shape[0]), 'wrap')
        hash_array = hash_array / 255.0
        return hash_array

    def embed(self, token):
        if token in WORD2VEC.key_to_index:
            return WORD2VEC[token]
        else:
            return self.str_to_vec(token)

    def select_cols(self, tokenized_ques, col_header):
        needed_cols = []
        col_id = 0
        for header in col_header:
            tokens = word_tokenize(header.lower())
            for token in tokens:
                if token in tokenized_ques:
                    needed_cols.append(col_id)
                    break
            col_id += 1

    def flatten_1(self, data_file):
        sampled_data = []
        data = pd.read_json(data_file, lines=True)
        for data_point, line in data.iterrows():
            question = word_tokenize(line['question'].lower())
            qid = line['qid']
            col_header = line['table']['cols']
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
            # all_tokens = torch.cat(all_tokens, torch.stack([tensor(self.embed(ques_tok)) + self.pos[i+1] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0] for i, ques_tok in enumerate(question)]))
            # index = all_tokens.shape[0]
            index = 1
            for ques_tok in question:
                ques_tok = tensor(self.embed(ques_tok)) + self.pos[index] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
                index += 1
                all_tokens.append(ques_tok)

            # Separator Token
            sep_token = self.sep + self.pos[index] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
            all_tokens.append(sep_token)

            # Table's column header Token
            col_cnt = 1
            for column in col_header:
                col_tok = word_tokenize(column.lower())
                for tok in col_tok:
                    tok = tensor(self.embed(tok)) + self.pos[index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[0] + self.rank_id[0]
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

            print(data_point, ": Necessary token count:", index, end=', ')

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
                        tok = tensor(self.embed(tok)) + self.pos[index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[row_cnt] + self.rank_id[0]
                        index += 1
                        all_tokens.append(tok)
                    col_cnt += 1
                row_cnt += 1

            print(data_point, index)

            # Table's incorrect row Token
            for row_id in negative_rows:
                row = rows[row_id]
                col_cnt = 0
                temp_tokens = []
                for cell in row:
                    cell_tok = word_tokenize(cell.lower())
                    for tok in cell_tok:
                        tok = tensor(self.embed(tok)) + self.pos[index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[row_cnt] + self.rank_id[0]
                        index += 1
                        if index >= self.max_seq_len:
                            break
                        temp_tokens.append(tok)
                    if index >= self.max_seq_len:
                        break
                    col_cnt += 1
                if index >= self.max_seq_len:
                    index -= len(temp_tokens)
                    break
                all_tokens.extend(temp_tokens)
                row_cnt += 1

            flat_data = {
                'qid': qid,
                'caption': caption,
                'id': id,
                'types': types,
                'label_col': label_col,
                'label_cell': label_cell,
                'label_row': label_row,
                'index': index
            }

            # Padding Token
            if index < self.max_seq_len:
                pad_token = self.pad + self.pos[index] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
                while index < self.max_seq_len:
                    all_tokens.append(pad_token)
                    index += 1

            flat_data['tokens'] = torch.stack(all_tokens)
            sampled_data.append(flat_data)

        df = pd.DataFrame(sampled_data)
        print(len(df.loc))
        print(df.loc[0])

    def flatten_2(self, data, tar):
        sampled_data = []
        for data_point, line in data.iterrows():
            question = word_tokenize(line['question'].lower())
            qid = line['qid']
            col_header = line['table']['cols']
            rows = line['table']['rows']
            types = line['table']['types']
            caption = line['table']['caption']
            id = line['table']['id']
            if tar:
                label_col = line['label_col']
                label_cell = np.array(line['label_cell'])
                label_row = np.sort(np.array(line['label_row']))

                first_column = label_cell[:, 0].astype(int)
                sorted_index = np.argsort(first_column)
                label_cell = label_cell[sorted_index]

            # Class Token
            cls_token = self.cls + self.pos[0] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
            # all_tokens = [cls_token]
            all_tokens = cls_token.unsqueeze(0)

            # Question Token
            all_tokens = torch.cat((all_tokens, torch.stack([tensor(self.embed(ques_tok)) + self.pos[i+1] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0] for i, ques_tok in enumerate(question)])), dim=0)
            index = all_tokens.shape[0]
            
            # index = 1
            # for ques_tok in question:
            #     ques_tok = tensor(self.embed(ques_tok)) + self.pos[index] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
            #     index += 1
            #     all_tokens.append(ques_tok)

            # Separator Token
            sep_token = self.sep + self.pos[index] + self.seg_id[0] + self.col_id[0] + self.row_id[0] + self.rank_id[0]
            # all_tokens.append(sep_token)
            all_tokens = torch.cat((all_tokens, sep_token.unsqueeze(0)), dim=0)
            index += 1

            # # Adding column header tokens
            # col_tokens = [word_tokenize(col.lower()) for col in col_header]
            # col_tokens = [[tensor(self.embed(tok)) for tok in tokens] for tokens in col_tokens]
            # all_tokens = torch.cat((all_tokens, torch.stack([tok + self.pos[index + i] + self.seg_id[1] + self.col_id[i+1] + self.row_id[0] + self.rank_id[0] for i, tokens in enumerate(col_tokens) for j, tok in enumerate(tokens)])), dim=0)

            # Table's column header Token

            # selected_cols = self.select_cols(question, col_header)
            # col_header = [col_header[i] for i in selected_cols]

            if tar:
                correct_col_ind = col_header.index(label_col[0])
                correct_head_ind = []
            col_cnt = 1
            temp_tokens = []
            for column in col_header:
                col_tok = word_tokenize(column.lower())
                for tok in col_tok:
                    tok = tensor(self.embed(tok)) + self.pos[index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[0] + self.rank_id[0]
                    if tar and col_cnt == correct_col_ind + 1:
                        correct_head_ind.append(index)
                    index += 1
                    temp_tokens.append(tok)
                col_cnt += 1
            all_tokens = torch.cat((all_tokens, torch.stack(temp_tokens)), dim=0)

            if tar:
                # Sampling rows
                correct_rows = []
                incorrect_rows = []
                for row_id in range(len(rows)):
                    if row_id in label_row:
                        correct_rows.append(row_id)
                    else:
                        incorrect_rows.append(row_id)

                # print(data_point, ": Necessary token count:", index, end=', ')

                # Table's correct row Token
                for correct_row_id in correct_rows:
                    loc_tokens = all_tokens
                    loc_index = index
                    row_cnt = 1
                    correct_row_pos = random.randint(0, min(NUM_NEG_ROWS, len(incorrect_rows)))
                    negative_rows = random.sample(incorrect_rows, min(NUM_NEG_ROWS, len(incorrect_rows)))

                    # Table's incorrect rows before correct row
                    temp_tokens = []
                    while row_cnt <= correct_row_pos:
                        row_id = negative_rows[row_cnt-1]
                        row = rows[row_id]
                        col_cnt = 0
                        for cell in row:
                            cell_tok = word_tokenize(cell.lower())
                            embedding = np.zeros(300)
                            for tok in cell_tok:
                                embedding += self.embed(tok)
                            tok = tensor(embedding) + self.pos[loc_index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[row_cnt] + self.rank_id[0]
                            loc_index += 1
                            temp_tokens.append(tok)
                            col_cnt += 1
                        row_cnt += 1
                    if len(temp_tokens) > 0:
                        loc_tokens = torch.cat((loc_tokens, torch.stack(temp_tokens)), dim=0)

                    # correct_cell = [correct_row_pos, label_cell[cnt][1]]
                    # correct_col_ind = col_header.index(label_cell[cnt][1])
                    correct_token_ind = 0
                    row = rows[correct_row_id]
                    col_cnt = 1
                    temp_tokens = []
                    for cell in row:
                        cell_tok = word_tokenize(cell.lower())
                        embedding = np.zeros(300)
                        for tok in cell_tok:
                            embedding += self.embed(tok)
                        tok = tensor(embedding) + self.pos[loc_index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[row_cnt] + self.rank_id[0]
                        if col_cnt == correct_col_ind + 1:
                            correct_token_ind = loc_index
                        loc_index += 1
                        temp_tokens.append(tok)
                        col_cnt += 1
                    row_cnt += 1
                    loc_tokens = torch.cat((loc_tokens, torch.stack(temp_tokens)), dim=0)
                    
                    # Table's incorrect rows after correct row
                    temp_tokens = []
                    while row_cnt <= min(NUM_NEG_ROWS, len(incorrect_rows)):
                        row_id = negative_rows[row_cnt-1]
                        row = rows[row_id]
                        col_cnt = 0
                        for cell in row:
                            cell_tok = word_tokenize(cell.lower())
                            embedding = np.zeros(300)
                            for tok in cell_tok:
                                embedding += self.embed(tok)
                            tok = tensor(embedding) + self.pos[loc_index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[row_cnt] + self.rank_id[0]
                            loc_index += 1
                            temp_tokens.append(tok)
                            col_cnt += 1
                        row_cnt += 1
                    if len(temp_tokens) > 0:
                        loc_tokens = torch.cat((loc_tokens, torch.stack(temp_tokens)), dim=0)

                    target = torch.zeros(self.max_seq_len)
                    target[correct_token_ind] = 1
                    for ind in correct_head_ind:
                        target[ind] = 1

                    # print(loc_index)

                    # Padding Token
                    if loc_index < self.max_seq_len:
                        num_pad_tokens = self.max_seq_len - loc_index
                        pad_tensor = self.pad.unsqueeze(0).repeat((num_pad_tokens, 1))
                        loc_tokens = torch.cat((loc_tokens, pad_tensor), dim=0)
                        # while loc_index < self.max_seq_len:
                        #     loc_tokens.append(pad_token)
                        #     loc_index += 1

                    # mask = torch.tensor((loc_tokens != self.pad).all(dim=1).unsqueeze(1).unsqueeze(2))
                    mask = (loc_tokens != self.pad).all(dim=1)
                    # print(mask)

                    flat_data = {
                        'qid': qid,
                        'caption': caption,
                        'id': id,
                        'types': types,
                        'label_tok_ind': correct_token_ind,
                        'index': loc_index,
                        'target': target,
                        'mask': mask,
                        'tokens': loc_tokens
                    }

                    sampled_data.append(flat_data)
            else:
                row_cnt = 1
                temp_tokens = []
                sampled_rows = random.sample(rows, min(NUM_NEG_ROWS+1, len(rows)))
                for row in sampled_rows:
                    col_cnt = 0
                    for cell in row:
                        cell_tok = word_tokenize(cell.lower())
                        embedding = np.zeros(300)
                        for tok in cell_tok:
                            embedding += self.embed(tok)
                        tok = tensor(embedding) + self.pos[index] + self.seg_id[1] + self.col_id[col_cnt] + self.row_id[row_cnt] + self.rank_id[0]
                        index += 1
                        temp_tokens.append(tok)
                        col_cnt += 1
                    row_cnt += 1
                if len(temp_tokens) > 0:
                    all_tokens = torch.cat((all_tokens, torch.stack(temp_tokens)), dim=0)

                # Padding Token
                if index < self.max_seq_len:
                    num_pad_tokens = self.max_seq_len - index
                    pad_tensor = self.pad.unsqueeze(0).repeat((num_pad_tokens, 1))
                    all_tokens = torch.cat((all_tokens, pad_tensor), dim=0)
                    # while index < self.max_seq_len:
                    #     all_tokens.append(pad_token)
                    #     index += 1

                mask = (all_tokens != self.pad).all(dim=1)
                # print(mask)

                flat_data = {
                    'qid': qid,
                    'caption': caption,
                    'id': id,
                    'types': types,
                    'index': index,
                    'mask': mask,
                    'tokens': all_tokens
                }

                sampled_data.append(flat_data)

        df = pd.DataFrame(sampled_data)
        # print(data_point, df.shape[0])
        # print(df['label_tok_ind'])
        # print(df['tokens'])
        print(df)
        print(df['mask'])
        return df
    
    def forward(self, data, target=False):
        return self.flatten_2(data, target)


if __name__=='__main__':
    flattener = Custom_Embedding(256)
    data = pd.read_json("data/A2_val.jsonl", lines=True)
    flattener.forward(data, target=False)