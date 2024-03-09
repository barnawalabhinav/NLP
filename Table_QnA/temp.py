import re
import pandas as pd
from gensim.utils import to_unicode, deaccent


PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
PAT_ALPHANUMERIC = re.compile(r'\w+|\$[\d.]+|\S', re.UNICODE)
_SPECIAL_CHAR = re.compile(r"\W")


def _get_table_text(table):
    yield table['cols']
    for row in table['rows']:
        yield row


def _tokenize_text(text):
    text = _SPECIAL_CHAR.sub(" ", text)
    return text.lower().split()


def _get_ngrams(
    table,
    # max_ngram_length,
):
    """Serializes table and returns token ngrams up to a maximum length."""
    for row in _get_table_text(table):
        tokens = []
        for text in row:
            tokens.extend(_tokenize_text(text))
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


data = pd.read_json('data/A2_val.jsonl', lines=True)
df = pd.DataFrame.from_dict(data)
print(list(_get_ngrams(df.table[0])))


def tokenize(text, lowercase=False, encoding='utf8', errors="strict", deacc=False):
    lowercase = lowercase
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)

    tokens = []
    # for elem in text:
    #     if isinstance(elem, (int, float)):
    #         tokens.append(f'NUM_{elem}')
    #     elif re.match(r'\d{4}-\d{2}-\d{2}')


# var = ["The price is $199.99.", '2', '3']
# tok = list(tokenize(var[0], lowercase=True, deacc=True))
# print(tok)

# tok = tokenize(preprocess(var[1]))
# print(list(tok))
