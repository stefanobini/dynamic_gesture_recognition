'''
-nvgesture
|---Video_data
|-------class_01
|-----------subject1_r0
|---------------nucleus.txt
|---------------sk_color.avi
|---------------sk_depth.avi
|-----------...
|-----------subject1_r5
|---------------nucleus.txt
|---------------sk_color.avi
|---------------sk_depth.avi
|-----------...
|-----------subject20_r0
|---------------nucleus.txt
|---------------sk_color.avi
|---------------sk_depth.avi
|-----------...
|-----------subject20_r5
|---------------nucleus.txt
|---------------sk_color.avi
|---------------sk_depth.avi
|------...
|------class_25
|---------------nucleus.txt
|---------------sk_color.avi
|---------------sk_depth.avi
|-----------...
|-----------subject1_r5
|---------------nucleus.txt
|---------------sk_color.avi
|---------------sk_depth.avi
|-----------...
|-----------subject20_r0
|---------------nucleus.txt
|---------------sk_color.avi
|---------------sk_depth.avi
|-----------...
|-----------subject20_r5
|---------------nucleus.txt
|---------------sk_color.avi
|---------------sk_depth.avi
'''

import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import random
from numpy.random import randint
from tqdm import tqdm

from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, sample_duration, image_loader):
    video = list()
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = dict()
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = list()
    annotations = list()
    frames = list()

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            # label = value['annotations']['label']
            # video_names.append('{}/{}'.format(label, key))
            video_names.append(key)
            annotations.append(value['annotations'])
            frames.append({'begin': value['start'], 'end': value['end']})

    return video_names, annotations, frames


def make_dataset(root_path, annotation_path, modalities, subset, n_samples_for_each_video, sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations, frames = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = dict()
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    
    dataset = list()
    data_iter = tqdm(range(len(video_names)), '{} set loading'.format(subset), total=len(video_names))
    for i in data_iter:
        mod_folder = ''
        video_paths = dict()
        video_path = ''
        for modality in modalities:
            if modality == 'RGB' or modality == 'D':
                mod_folder = 'RGB-D_frames'
            elif modality == 'OF' or modality == 'OF_D':
                mod_folder = 'OF_frames'
            elif modality == 'MHI' or modality == 'MHI_D':
                mod_folder = 'MHI_frames'
            
            video_name = video_names[i].replace('color', 'depth') if modality in ['D', 'OF_D', 'MHI_D'] else video_names[i]
            video_path = os.path.join(root_path, mod_folder, video_name)
            if not os.path.exists(video_path):
                print(video_path)
                continue
            video_paths[modality] = video_path

        begin_t = int(frames[i]['begin'])
        end_t = int(frames[i]['end'])
        n_frames = end_t - begin_t
        sample = {
            'videos': video_paths,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            # 'video_id': video_names[i].split('/')[1]
            'video_id': video_names[i]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1
        
        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1, math.ceil((end_t - 1 - sample_duration) / (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(begin_t, end_t, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(end_t, j + sample_duration)))
                dataset.append(sample_j)

    
    return dataset, idx_to_class


class NVGesture(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    #'''
    def __init__(self,
                 root_path,
                 annotation_path,
                 modalities,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 cnn_dim=3):
        self.data, self.class_names = make_dataset(
            root_path,
            annotation_path,
            modalities,
            subset,
            n_samples_for_each_video,
            sample_duration)
        self.modalities = modalities
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.sample_duration = sample_duration
        self.loader = get_loader()
        self.cnn_dim = cnn_dim
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clips_list, target) where clips_list contain the same video in the selected modalities, and target is class_index of the target class.
        """
        
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        
        clips_list = list()
        for modality in self.modalities:
            path = self.data[index]['videos'][modality]
            # print('PATH: {}\tMODALITY: {}\tFRAME INDICES: {}\tSAMPLE DURATION: {}'.format(path, modality, frame_indices, self.sample_duration))
            # clip = self.loader(path, frame_indices)
            clip = self.loader(path, frame_indices, self.sample_duration)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            # im_dim = clip[0].size()[-2:]
            if self.cnn_dim == 3:
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            else:
                clip = torch.stack(clip, 0)
            # print('clip shape: {}'.format(clip.shape))
            
            clips_list.append(clip)
        
        clips = torch.stack(clips_list, 0)  # trasform in a tensor
        
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # print('Video: {}\nLabel: {}\n'.format(path, target))

        return clips, target

    def __len__(self):
        return len(self.data)