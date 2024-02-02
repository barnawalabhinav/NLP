import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


# TODO: Grid search for hyperparameters (alpha)
def naive_bayes_baseline(dataframe, model_path):
    pipeline = Pipeline([
        # ('tfidf', TfidfVectorizer()),
        ('cntvec', CountVectorizer()),
        ('mnb', MultinomialNB(alpha=0.5))
    ])
    pipeline.fit(dataframe['text'].values, dataframe['langid'].values)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def train(data_path, model_path, model_type='naive_bayes'):
    df = load_data(data_path)

    if model_type == 'naive_bayes':
        naive_bayes_baseline(df, model_path)
    elif model_type == 'svm':
        svm(df, model_path)

    print('------------------ Training Done! ------------------')


def predict(data_path, model_path):
    df = load_data(data_path)

    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    y_pred = pipeline.predict(df['text'].values)

    y_true = df['langid'].values
    micro_f1 = compute_micro_f1_score(y_pred, y_true)
    macro_f1 = compute_macro_f1_score(y_pred, y_true)
    accuracy = np.mean(y_pred == y_true)
    print(f'Micro F1: {micro_f1}')
    print(f'Macro F1: {macro_f1}')
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    train('data/train.json', 'models/model.pkl', 'naive_bayes')
    predict('data/valid.json', 'models/model.pkl')
