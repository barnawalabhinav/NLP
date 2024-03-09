import math
import torch
import gensim
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import gensim.downloader as api
from gensim.utils import tokenize

def custom_tokenize(text):
    text = text.lower()
    tokens = text.split(' ')
    return tokens

def load_data(data_file):
    data = pd.read_json(data_file, lines=True)
    # df = pd.DataFrame.from_dict(data)

    flattened_data = []

    for line in data:
        question = tokenize(line['question'], lowercase=True, deacc=True)
        qid = tokenize(line['qid'])
        cols = tokenize(line['table']['cols'])
        rows = line['table']['rows']
        types = line['table']['types']
        caption = line['table']['caption']
        id = line['table']['id']
        label_col = line['label_col']
        label_cell = line['label_cell']
        label_row = line['label_row']
        
    #     record = {
    #         "question": question,
    #         "qid": qid,
    #         "cols": cols,
    #         "rows": rows,
    #         "types": types,
    #         "caption": caption,
    #         "id": id,
    #         "label_col": label_col,
    #         "label_cell": label_cell,
    #         "label_row": label_row
    #     }
    #     arr.append(record)
    # df = pd.DataFrame(arr)

    # Display the DataFrame
    # print(df.iloc[0].table['cols'])

    return df


def embed(dataframe):
    tables = []
    for table in dataframe.table.unique():
        tables.append(table)

    table_df = pd.DataFrame(tables)
    
    print(table_df[0])


class TableTranformer(nn.Module):
    def __init__(self):
        super(TableTranformer, self).__init__()

        self.embedding = api.load("word2vec-google-news-300")

    def predict(self, table, question):
        result = self.nlp(table=table, query=question)
        return result


if __name__ == "__main__":
    df = load_data("../data/A2_val.jsonl")
    print(df)
    embed(df)