import sys
import pickle
import numpy as np

from evaluation import compute_macro_f1_score, compute_micro_f1_score
from train import load_data


def predict(data_path, model_path, out_path):
    df = load_data(data_path)

    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    y_pred = pipeline.predict(df['text'].values)
    with open(out_path, 'w') as f:
        for pred in y_pred:
            f.write(pred + '\n')

    # y_true = df['langid'].values
    # micro_f1 = compute_micro_f1_score(y_pred, y_true)
    # macro_f1 = compute_macro_f1_score(y_pred, y_true)
    # accuracy = np.mean(y_pred == y_true)
    # print(f'Micro F1: {micro_f1}')
    # print(f'Macro F1: {macro_f1}')
    # print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    out_path = sys.argv[3]
    predict(data_path, model_path, out_path)
