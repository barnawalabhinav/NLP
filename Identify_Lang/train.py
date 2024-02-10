import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# from evaluation import compute_macro_f1_score, compute_micro_f1_score


def load_data(data_path):
    df = pd.read_json(data_path)
    return df


def svm(dataframe, model_path):
    print("------- Training SVM --------")
    pipeline = Pipeline([
        # ('tfidf', TfidfVectorizer()),
        ('cntvec', CountVectorizer()),
        ('svc', SVC())
    ])
    pipeline.fit(dataframe['text'].values, dataframe['langid'].values)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def naive_bayes(dataframe, model_path):
    print("------- Training Naive Bayes --------")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
        # ('tfidf', TfidfVectorizer()),
        # ('tfidf', CountVectorizer(ngram_range=(1, 2))),
        # ('cntvec', CountVectorizer()),
        ('mnb', MultinomialNB(alpha=0.005))
    ])
    pipeline.fit(dataframe['text'].values, dataframe['langid'].values)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def logistic_regr(dataframe, model_path):
    print("------- Training Logistic Regression --------")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        # ('cntvec', CountVectorizer()),
        ('log_regr', LogisticRegression(penalty='elasticnet', l1_ratio=0.5,
         solver='saga', warm_start=True, multi_class='multinomial'))
    ])
    pipeline.fit(dataframe['text'].values, dataframe['langid'].values)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)


def random_forest(dataframe, model_path):
    print("------- Training Random Forest --------")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        # ('cntvec', CountVectorizer()),
        ('random_forest', RandomForestClassifier(
            n_estimators=100, max_depth=2, random_state=0))
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
        naive_bayes(df, model_path)
    elif model_type == 'svm':
        svm(df, model_path)
    elif model_type == 'logistic_regr':
        logistic_regr(df, model_path)
    elif model_type == 'random_forest':
        random_forest(df, model_path)

    print('------------------ Training Done! ------------------')

    # ---------------------------------------------- #

    # with open(model_path, 'rb') as f:
    #     pipeline = pickle.load(f)
    # y_pred = pipeline.predict(df['text'].values)

    # y_true = df['langid'].values
    # micro_f1 = compute_micro_f1_score(y_pred, y_true)
    # macro_f1 = compute_macro_f1_score(y_pred, y_true)
    # accuracy = np.mean(y_pred == y_true)
    # print(f'TRAIN Micro F1: {micro_f1}')
    # print(f'TRAIN Macro F1: {macro_f1}')
    # print(f'TRAIN Accuracy: {accuracy}')

    # # ---------------------------------------------- #

    # df = load_data("data/valid.json")
    # y_pred = pipeline.predict(df['text'].values)

    # y_true = df['langid'].values
    # micro_f1 = compute_micro_f1_score(y_pred, y_true)
    # macro_f1 = compute_macro_f1_score(y_pred, y_true)
    # accuracy = np.mean(y_pred == y_true)
    # print(f'VALID Micro F1: {micro_f1}')
    # print(f'VALID Macro F1: {macro_f1}')
    # print(f'VALID Accuracy: {accuracy}')

    # # ---------------------------------------------- #

    # df = load_data("data/valid_new.json")
    # y_pred = pipeline.predict(df['text'].values)

    # y_true = df['langid'].values
    # micro_f1 = compute_micro_f1_score(y_pred, y_true)
    # macro_f1 = compute_macro_f1_score(y_pred, y_true)
    # accuracy = np.mean(y_pred == y_true)
    # print(f'VALID_NEW Micro F1: {micro_f1}')
    # print(f'VALID_NEW Macro F1: {macro_f1}')
    # print(f'VALID_NEW Accuracy: {accuracy}')


if __name__ == '__main__':
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        train(data_path, model_path, model_type)
    else:
        train(data_path, model_path)
