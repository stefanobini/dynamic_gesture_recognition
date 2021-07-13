'''
    Convert video.avi in folder of its streams.
    python3 utils/video_jpg_chalearn_isogd.py datasets/chalearn_isogd datasets/chalearn_isogd_frame_video
'''

from __future__ import print_function, division
import os
import sys
import subprocess
from tqdm import tqdm


def class_process(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)  # arrive in folder with video.avi
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)
    
    for file_name in os.listdir(class_path):  # for each video.avi in video folder
        if '.avi' not in file_name:  # check if the file is a video.avi
            continue
        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_class_path,
                                          name)  # a subfolder is created for each video.avi, which will contain the rgb frames

        video_file_path = os.path.join(class_path, file_name)  # build video path
        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.mkdir(dst_directory_path)
                else:
                    continue
            else:
                os.mkdir(dst_directory_path)
        except:
            print(dst_directory_path)
            continue
        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
        # print(cmd)
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # print('\n')


if __name__ == "__main__":
    dir_path = sys.argv[1]  # avi_video_directory
    dst_dir_path = sys.argv[2]  # jpg_video_directory

    # Training set
    train_dir_path = os.path.join(dir_path, 'train')
    train_dst_dir_path = os.path.join(dst_dir_path, 'train')
    if not os.path.exists(train_dst_dir_path):
        os.mkdir(train_dst_dir_path)
    files_iter = tqdm(os.listdir(train_dir_path))
    for class_name in files_iter:  # for each directory in directory list
        class_process(train_dir_path, train_dst_dir_path, class_name)
        files_iter.set_description('Training set conversion')

    # Validation set
    valid_dir_path = os.path.join(dir_path, 'valid')
    valid_dst_dir_path = os.path.join(dst_dir_path, 'valid')
    if not os.path.exists(valid_dst_dir_path):
        os.mkdir(valid_dst_dir_path)
    files_iter = tqdm(os.listdir(valid_dir_path))
    for class_name in files_iter:  # for each directory in directory list
        class_process(valid_dir_path, valid_dst_dir_path, class_name)
        files_iter.set_description('Validation set conversion')

    # Testing set
    test_dir_path = os.path.join(dir_path, 'test')
    test_dst_dir_path = os.path.join(dst_dir_path, 'test')
    if not os.path.exists(test_dst_dir_path):
        os.mkdir(test_dst_dir_path)
    files_iter = tqdm(os.listdir(test_dir_path))
    for class_name in files_iter:  # for each directory in directory list
        class_process(test_dir_path, test_dst_dir_path, class_name)
        files_iter.set_description('Testing set conversion')
