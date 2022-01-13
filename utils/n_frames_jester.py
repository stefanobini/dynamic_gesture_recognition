'''
python3 utils/n_frames_jester.py ../datasets/jester/RGB_frames
'''

from __future__ import print_function, division
import os
import sys
import subprocess
from tqdm import tqdm


def class_process(dir_path):
    total_frames = 0
    n_videos = 0
    if not os.path.isdir(dir_path):
        return
    
    data_iter = tqdm(os.listdir(dir_path))
    for file_name in data_iter:
        video_dir_path = os.path.join(dir_path, file_name)
        image_indices = []
        
        for image_file_name in os.listdir(video_dir_path):
            if '00' not in image_file_name:
                continue
            image_indices.append(int(image_file_name[0:4]))

        if len(image_indices) == 0:
            print('no image files', video_dir_path)
            n_frames = 0
        else:
            image_indices.sort(reverse=True)
            n_frames = len(image_indices)
            # print(video_dir_path, n_frames)
        total_frames += n_frames
        n_videos += 1
        
        with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
            dst_file.write(str(n_frames))
        
        data_iter.set_description('Dataset count')
    return total_frames, n_videos


if __name__=="__main__":
    dir_path = sys.argv[1]
    total_frames, n_videos = class_process(dir_path)
    
    avg_frame_video = total_frames / n_videos
    print('Average number of frame per video: {}'.format(avg_frame_video))
    with open('avg_frame_video.txt', 'w') as f:
        f.write('Average number of frame per video: {}'.format(avg_frame_video))

