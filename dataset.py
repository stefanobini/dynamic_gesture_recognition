from datasets.kinetics import Kinetics
from datasets.ucf101 import UCF101
from datasets.jester import Jester
from datasets.isogd import IsoGD
from datasets.nvgesture import NVGesture

def get_training_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['jester', 'isogd', 'nvgesture']

    if opt.dataset == 'jester':
        training_data = Jester(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    elif opt.dataset == 'isogd':
        training_data = IsoGD(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    elif opt.dataset == 'nvgesture':
        training_data = NVGesture(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['jester', 'isogd', 'nvgesture']

    if opt.dataset == 'jester':
        validation_data = Jester(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    elif opt.dataset == 'isogd':
        validation_data = IsoGD(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    elif opt.dataset == 'nvgesture':
        validation_data = NVGesture(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['jester', 'isogd', 'nvgesture']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'jester':
        test_data = Jester(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            subset,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    elif opt.dataset == 'isogd':
        test_data = IsoGD(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            subset,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    elif opt.dataset == 'nvgesture':
        test_data = NVGesture(
            opt.video_path,
            opt.annotation_path,
            opt.modalities,
            subset,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            cnn_dim=opt.cnn_dim)
    return test_data
