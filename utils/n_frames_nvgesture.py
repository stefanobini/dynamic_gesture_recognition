'''
In each folder containing video frame, create a file that contain the number of frame contained in this video.
python3 utils/n_frames_chalearn_isogd.py datasets/chalearn_isogd_frame_video
I need to insert a control to check that the length of the video in RGB is equal to that of Depth.

python3 utils/n_frames_nvgesture.py ../datasets/nvgesture/RGB-D_frames
'''
from __future__ import print_function, division
import os
import sys
import subprocess
from tqdm import tqdm


def class_process(dir_path, class_name, total_frames, n_videos):
    class_path = os.path.join(dir_path, class_name)     # i am in a class folder
    if not os.path.isdir(class_path):
        return

    for file_name in os.listdir(class_path):
        video_dir_path = os.path.join(class_path, file_name)    # i am in a subject with al the frames related to a video
        image_indices = []
        
        for modality in ['sk_color', 'sk_depth']:
            mod_video_dir_path = os.path.join(video_dir_path, modality)     # i am in a modality folder
            if not os.path.isdir(mod_video_dir_path):
                # print('1: {}'.format(mod_video_dir_path))
                return
            
            for image_file_name in os.listdir(mod_video_dir_path):
                if 'image' not in image_file_name:
                    # print('2: {}'.format(image_file_name))
                    continue
                # print('3: {}'.format(image_file_name))
                image_indices.append(int(image_file_name[6:11]))

            if len(image_indices) == 0:
                # print('no image files', mod_video_dir_path)
                n_frames = 0
            else:
                image_indices.sort(reverse=True)
                n_frames = image_indices[0]
                # print(mod_video_dir_path, n_frames)
            total_frames += n_frames
            n_videos += 1
            with open(os.path.join(mod_video_dir_path, 'n_frames'), 'w') as dst_file:
                dst_file.write(str(n_frames))
    
    return total_frames, n_videos


if __name__ == "__main__":
    dir_path = sys.argv[1]
    total_frames = 0
    n_videos = 0

    files_iter = tqdm(os.listdir(dir_path))
    for class_name in files_iter:    # i am in the modality folder, so i have different folder with different video inside
        frames, videos = class_process(dir_path, class_name, total_frames, n_videos)
        total_frames += frames
        n_videos += videos
        files_iter.set_description('Dataset count')
    
    avg_frame_video = total_frames / n_videos
    print('Average number of frame per video: {}'.format(avg_frame_video))
    with open('avg_frame_video.txt', 'w') as f:
        f.write('Average number of frame per video: {}'.format(avg_frame_video))
