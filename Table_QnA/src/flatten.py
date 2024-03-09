import torch
import torch.nn as nn


import re
import pandas as pd
from gensim.utils import to_unicode, deaccent


PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
PAT_ALPHANUMERIC = re.compile(r'\w+|\$[\d.]+|\S', re.UNICODE)
_SPECIAL_CHAR = re.compile(r"\W")


class Flatten(nn.Module):
    def __init__(self, token_size):
        super(Flatten, self).__init__()
        self.cls = nn.Parameter(torch.randn(token_size))
        self.sep = nn.Parameter(torch.randn(token_size))
        self.seg_id = nn.Parameter(torch.randn(2, token_size))
        self.col_id = nn.Parameter(torch.randn(100, token_size))
        self.row_id = nn.Parameter(torch.randn(100, token_size))
        self.rank_id = nn.Parameter(torch.randn(100, token_size))
        
    def _get_table_text(self, table):
        yield table['cols']
        for row in table['rows']:
            yield row

    def _tokenize_text(self, text):
        text = _SPECIAL_CHAR.sub(" ", text)
        return text.lower().split()

    def _tokenize_table(self, table,
        # max_ngram_length
    ):
        """Serializes table and returns token ngrams up to a maximum length."""
        for row in self._get_table_text(table):
            tokens = []
            for text in row:
                tokens.extend(self._tokenize_text(text))
            print(list(tokens), '\n---')
            yield " ".join(tokens)

            # for index in range(len(tokens)):
            # 	length = min(max_ngram_length, len(tokens) - index)
            # 	ngram = tokens[index:index + length]
            # 	yield " ".join(ngram)

            # Generates different subset of final ngrams
            # for index in range(len(tokens)):
            # 	for length in range(1, min(max_ngram_length, len(tokens) - index) + 1):
            # 		ngram = tokens[index:index + length]
            # 		yield " ".join(ngram)


    def forward(self, X):
        return X + self.pos[:, :X.size(1)]