'''
    python3 utils/chalearn_isogd_json.py annotation_ChaLearn_IsoGD
'''

from __future__ import print_function, division
import os
import sys
import json
import pandas as pd
from tqdm import tqdm


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    rows_iter = tqdm(range(data.shape[0]))
    for i in rows_iter:
        row = data.iloc[i, :]
        basename = row[0].split('.')[0]
        label = int(row[2])

        keys.append(basename)
        key_labels.append(label)
        
        rows_iter.set_description('{} set conversion'.format(subset))

    database = {}
    keys_iter = tqdm(range(len(keys)))
    for i in keys_iter:
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}
        
        keys_iter.set_description('{} set filling dictionary'.format(subset))

    return database


def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(int(data.iloc[i, 1]))
    return labels


def convert_chalearn_isogd_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                           val_csv_path, test_csv_path, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')
    test_database = convert_csv_to_dict(test_csv_path, 'testing')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    dst_data['database'].update(test_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    csv_dir_path = sys.argv[1]

    label_csv_path = os.path.join(csv_dir_path, 'classInd.txt')
    train_csv_path = os.path.join(csv_dir_path, 'trainlist.txt')
    val_csv_path = os.path.join(csv_dir_path, 'vallist.txt')
    test_csv_path = os.path.join(csv_dir_path, 'testlist.txt')
    dst_json_path = os.path.join(csv_dir_path, 'chalearn_isogd.json')
    convert_chalearn_isogd_csv_to_activitynet_json(label_csv_path, train_csv_path,
                                           val_csv_path, test_csv_path, dst_json_path)

