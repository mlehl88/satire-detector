import logging
from pathlib import Path

import pandas as pd
from scipy.sparse import issparse
from sklearn.base import BaseEstimator

POS_LABEL = 'true'


def get_labels(label_path):
    labels = {}
    with open(label_path, 'r') as label_file:
        for row in label_file.readlines():
            id_, string_label = row.strip().split(' ')
            labels[id_] = 1 if string_label == POS_LABEL else 0
    return labels


def read_text(fp):
    with open(fp, 'rb') as infile:
        text = infile.read()

    # Guess encoding if necessary
    try:
        return str(text, 'utf-8').strip()
    except UnicodeDecodeError:
        return str(text, 'latin1').strip()


def load_data(data_path, label_path):
    labels = get_labels(label_path)

    data = []
    for fp in data_path.iterdir():
        try:
            data.append({
                'id': fp.name,
                'text': read_text(fp),
                'label': labels[fp.name]})
        except KeyError:
            logging.error('Could not find label for file %s' % fp.name)
            continue
    return data
