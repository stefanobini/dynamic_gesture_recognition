'''
In each folder containing video frame, create a file that contain the number of frame contained in this video.
python3 utils/n_frames_chalearn_isogd.py datasets/chalearn_isogd_frame_video
I need to insert a control to check that the length of the video in RGB is equal to that of Depth.
'''
from __future__ import print_function, division
import os
import sys
import subprocess
from tqdm import tqdm

total_frames = 0
n_video = 0

def class_process(dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)     # i am in a sub-folder with one subfolder (containing video frames) for each video
    if not os.path.isdir(class_path):
        return

    for file_name in os.listdir(class_path):
        video_dir_path = os.path.join(class_path, file_name)    # i am in a sub-sub-folder with al the frames related to a video
        image_indices = []
        for image_file_name in os.listdir(video_dir_path):
            if 'image' not in image_file_name:
                continue
            image_indices.append(int(image_file_name[6:11]))

        if len(image_indices) == 0:
            # print('no image files', video_dir_path)
            n_frames = 0
        else:
            image_indices.sort(reverse=True)
            n_frames = image_indices[0]
            # print(video_dir_path, n_frames)
        global total_frames
        global n_video
        total_frames += n_frames
        n_video += 1
        with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
            dst_file.write(str(n_frames))


if __name__ == "__main__":
    dir_path = sys.argv[1]

    # Training set
    train_dir_path = os.path.join(dir_path, 'train')
    files_iter = tqdm(os.listdir(train_dir_path))
    for class_name in files_iter:    # i am in the train folder, so i have different folder with different video inside
        class_process(train_dir_path, class_name)
        files_iter.set_description('Training set count')

    # Validation set
    valid_dir_path = os.path.join(dir_path, 'valid')
    files_iter = tqdm(os.listdir(valid_dir_path))
    for class_name in files_iter:
        class_process(valid_dir_path, class_name)
        files_iter.set_description('Validation set count')

    # Testing set
    test_dir_path = os.path.join(dir_path, 'test')
    files_iter = tqdm(os.listdir(test_dir_path))
    for class_name in files_iter:
        class_process(test_dir_path, class_name)
        files_iter.set_description('Testing set count')
    
    avg_frame_video = total_frames / n_video
    print('Average number of frame per video: {}'.format(avg_frame_video))
    with open('avg_frame_video.txt', 'w') as f:
        f.write('Average number of frame per video: {}'.format(avg_frame_video))
