'''
python3 utils/nvgesture_json.py annotation_NVGesture
'''

from __future__ import print_function, division
import os
import sys
import json
import pandas as pd
import random
from tqdm import tqdm


def load_labels(label_path):
    data = pd.read_csv(label_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(int(data.iloc[i, 0]))
    return labels


def load_split_nvgesture(file_with_split = './nvgesture_train_correct.lst', list_split = list()):
    '''
    Load data in as a list, each element is a dictionary with this structure: 
    {   
        'dataset': 'nvgesture', 
        'depth': '<path>', 
        'depth_start': <start_frame(int)>, 
        'depth_end': <end_frame(int)>, 
        'color': '<path>', 
        'color_start': <start_frame(int)>, 
        'color_end': <end_frame(int)>, 
        'label': <label (int-1)>
    }
    '''
    video_classes = dict()
    params_dictionary = dict()
    with open(file_with_split,'r') as f:
        dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
        dict_name  = dict_name[:dict_name.find('_')]
        
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            params = line.split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1].replace('Video_data/', '')
            video_class = path.split('/')[1]
            if video_class in video_classes:
                video_classes[video_class]['samples'] += 1
            else:
                video_classes[video_class] = {'samples': 0, 'training': 0}
            for param in params[1:]:
                    parsed = param.split(':')
                    key = parsed[0]
                    if key == 'label':
                        # make label start from 0
                        label = int(parsed[1]) - 1 
                        params_dictionary['label'] = label
                    # elif key in ('depth','color','duo_left'):
                    elif key in ('depth','color'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1]
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])

                        params_dictionary[key+'_end'] = int(parsed[3])
            '''
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']          

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  
            '''
            list_split.append(params_dictionary)
 
    return list_split, video_classes
    

def convert_nvgesture_annotation_to_activitynet_json(label_path, train_path, test_csv_path, dst_json_path):
    labels = load_labels(label_path)
    train_list = list()
    test_list = list()
    _, train_classes = load_split_nvgesture(file_with_split=train_path, list_split=train_list)
    _, test_classes = load_split_nvgesture(file_with_split=test_path, list_split=test_list)
    
    data = dict()
    
    train_perc = 0.67
    list_iter = tqdm(train_list)
    for element in list_iter:
        video_class = element['color'].split('/')[1]
        if train_classes[video_class]['training'] < int(train_classes[video_class]['samples'] * train_perc):
            data[element['color'][2:]]={'subset': 'training', 'annotations': {'label': element['label']+1}, 'start': element['color_start'], 'end': element['color_end']}
            train_classes[video_class]['training'] += 1
        else:
            data[element['color'][2:]]={'subset': 'validation', 'annotations': {'label': element['label']+1}, 'start': element['color_start'], 'end': element['color_end']}
        list_iter.set_description('Training set conversion')
    '''
    random.shuffle(train_list)
    val_perc = 0.33
    train_samples = int(len(train_list)*(1-val_perc))
    
    list_iter = tqdm(range(train_samples))
    for i in list_iter:
        data[train_list[i]['color'][2:]]={'subset': 'training', 'annotations': {'label': train_list[i]['label']+1}, 'start': train_list[i]['color_start'], 'end': train_list[i]['color_end']}
        list_iter.set_description('Training set conversion')
    
    list_iter = tqdm(range(train_samples, len(train_list)))
    for i in list_iter:
        data[train_list[i]['color'][2:]]={'subset': 'validation', 'annotations': {'label': train_list[i]['label']+1}, 'start': train_list[i]['color_start'], 'end': train_list[i]['color_end']}
        list_iter.set_description('Validation set conversion')
    '''
    random.shuffle(test_list)
    list_iter = tqdm(test_list)
    for element in test_list:
        data[element['color'][2:]]={'subset': 'testing', 'annotations': {'label': element['label']+1}, 'start': element['color_start'], 'end': element['color_end']}
        list_iter.set_description('Test set conversion')
        
    dst_data = dict()
    dst_data['labels'] = labels
    dst_data['database'] = data

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    annotation_dir_path = sys.argv[1]

    label_path = os.path.join(annotation_dir_path, 'classInd.txt')
    train_path = os.path.join(annotation_dir_path, 'nvgesture_train_correct_cvpr2016_v2.lst')
    test_path = os.path.join(annotation_dir_path, 'nvgesture_test_correct_cvpr2016_v2.lst')
    dst_json_path = os.path.join(annotation_dir_path, 'nvgesture.json')
    convert_nvgesture_annotation_to_activitynet_json(label_path, train_path, test_path, dst_json_path)