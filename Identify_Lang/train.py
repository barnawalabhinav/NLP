import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from evaluation import compute_macro_f1_score, compute_micro_f1_score


def load_data(data_path):
    df = pd.read_json(data_path)
    return df


def svm(dataframe, model_path):
    pipeline = Pipeline([
        # ('tfidf', TfidfVectorizer()),
        ('cntvec', CountVectorizer()),
        ('svc', SVC())
    ])
    pipeline.fit(dataframe['text'].values, dataframe['langid'].values)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def naive_bayes_baseline(dataframe, model_path):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(2, 2))),
        # ('tfidf', TfidfVectorizer()),
        # ('cntvec', CountVectorizer()),
        ('mnb', MultinomialNB(alpha=0.01))
    ])
    pipeline.fit(dataframe['text'].values, dataframe['langid'].values)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def logistic_regr_baseline(dataframe, model_path):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        # ('cntvec', CountVectorizer()),
        ('log_regr', LogisticRegression(solver='saga', warm_start=True))
    ])
    pipeline.fit(dataframe['text'].values, dataframe['langid'].values)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def train(data_path, model_path, model_type='naive_bayes'):
    df = load_data(data_path)

    # # Downsample the data
    # for lang in ['bn', 'de', 'en', 'es', 'fr', 'hi', 'it', 'kn', 'ml', 'mr', 'pt', 'sv', 'ta']:
    #     lang_indices = df[df.langid == lang].index
    #     if len(lang_indices) > 160000:
    #         random_indices = np.random.choice(lang_indices, 160000, replace=False)
    #         drop_indices = lang_indices.difference(random_indices)
    #         df = df.drop(index=drop_indices)


    if model_type == 'naive_bayes':
        naive_bayes_baseline(df, model_path)
    elif model_type == 'svm':
        svm(df, model_path)
    elif model_type == 'logistic_regr':
        logistic_regr_baseline(df, model_path)

    print('------------------ Training Done! ------------------')


if __name__ == '__main__':
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    train(data_path, model_path)
    # train(data_path, model_path, 'logistic_regr')
