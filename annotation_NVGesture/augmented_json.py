'''
python3 augmented_json.py --src_json_path nvgesture.json --dst_json_path nvgesture_aug.json
'''


import os
import argparse
from tqdm import tqdm
import json
import copy
import random


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_json_path', default=None, type=str, help='')
    parser.add_argument('--dst_json_path', default=None, type=str, help='')

    args = parser.parse_args()

    return args


def load_json(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


if __name__ == "__main__":    
    opt = parse_opts()
    
    ann_json = load_json(opt.src_json_path)
    new_json = copy.deepcopy(ann_json)
    
    data_iter = tqdm(ann_json['database'].items())
    for key, value in data_iter:
        folders = key.split('/')
        
        # Augment only training set
        if value['subset'] == 'training':
            
            # Rotation-dependent label
            if value['annotations']['label'] in [1, 5, 20]:
                new_video_id = os.path.join(folders[0], '{}-flip'.format(folders[1]), folders[2])
                new_json['database'][new_video_id] = {'subset': 'training', 
                                'annotations': {"label": value['annotations']['label'] + 1},
                                'start': value['start'],
                                'end': value['end']}
                for val in ['-10', '-5', '5', '10']:
                    new_video_id = os.path.join(folders[0], '{}-rot_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_rot_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label'] + 1},
                                    'start': value['start'],
                                    'end': value['end']}
                for val in ['0.8', '0.9', '1.1', '1.2']:
                    new_video_id = os.path.join(folders[0], '{}-scale_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_scale_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label'] + 1},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-bright_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_bright_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label'] + 1},
                                    'start': value['start'],
                                    'end': value['end']}
            
            # Rotation-dependent label
            elif value['annotations']['label'] in [2, 6, 21]:
                new_video_id = os.path.join(folders[0], '{}-flip'.format(folders[1]), folders[2])
                new_json['database'][new_video_id] = {'subset': 'training', 
                                'annotations': {"label": value['annotations']['label'] - 1},
                                'start': value['start'],
                                'end': value['end']}
                for val in ['-10', '-5', '5', '10']:
                    new_video_id = os.path.join(folders[0], '{}-rot_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_rot_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label'] - 1},
                                    'start': value['start'],
                                    'end': value['end']}
                for val in ['0.8', '0.9', '1.1', '1.2']:
                    new_video_id = os.path.join(folders[0], '{}-scale_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_scale_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label'] - 1},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-bright_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_bright_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label'] - 1},
                                    'start': value['start'],
                                    'end': value['end']}
            
            # No rotation-dependent label
            else:
                new_video_id = os.path.join(folders[0], '{}-flip'.format(folders[1]), folders[2])
                new_json['database'][new_video_id] = {'subset': 'training', 
                                'annotations': {"label": value['annotations']['label']},
                                'start': value['start'],
                                'end': value['end']}
                for val in ['-10', '-5', '5', '10']:
                    new_video_id = os.path.join(folders[0], '{}-rot_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_rot_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                for val in ['0.8', '0.9', '1.1', '1.2']:
                    new_video_id = os.path.join(folders[0], '{}-scale_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_scale_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-bright_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
                    new_video_id = os.path.join(folders[0], '{}-flip_bright_{}'.format(folders[1], val), folders[2])
                    new_json['database'][new_video_id] = {'subset': 'training', 
                                    'annotations': {"label": value['annotations']['label']},
                                    'start': value['start'],
                                    'end': value['end']}
        
        data_iter.set_description('Dictionary creation.')
    
    keys = list(new_json['database'].keys())
    random.seed()
    random.shuffle(keys)
    shuffle_json = dict()
    shuffle_json['labels'] = ann_json['labels']
    shuffle_json['database'] = dict()
    for key in keys:
        shuffle_json['database'][key] = new_json['database'][key]
    
    with open(opt.dst_json_path, 'w') as dst_file:
        json.dump(shuffle_json, dst_file)