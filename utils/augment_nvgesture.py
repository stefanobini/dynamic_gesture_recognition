'''
python3 aument_nvgesture.py --dataset_path ../../../../../mnt/sdc1/sbini/nvgesture
'''


import cv2
import numpy
import os
from glob import globtype
import argparse
from tqdm import tqdm
import json


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=None, type=str, help='')

    args = parser.parse_args()

    return args


def horizontal_flipping(img):
    return cv2.flip(img)


def rotations(img, angles):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    imgs = list()
    for angle in amgles:
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        imgs.append(cv2.warpAffine(img, M, (w, h)))
    
    return imgs


def rescalings(img, scales):
    size = img.shape
    
    imgs = list()
    for scale in scales:
        img = img.resize((int(size[0]*scale), int(size[1]*size)))
        imgs.append(img)
        
    return imgs
    


def brightness_modifyings(img, coeffs):
    img = img.astype(np.float64)
    
    imgs = list()
    for coeff in coeffs:
        img *= coeff
        img = np.where(img > 255, 255, img)
        img = np.where(img < 0, 0, img)
        img = img.astype(np.uint8)
        imgs.append(img)
    
    return imgs


def augment_data(class_folder, video_subject, modality):
    src_folder = os.join(class_folder, modality)
                
    for img_file in glob('{}/*.jpg'.format(src_folder)):
        # Flip image
        aug_img = horizontal_flipping(img)
        aug_folder = os.join(class_folder, '{}-flip'.format(video_subject))
        os.mkdir(aug_folder)
        rgb_aug_folder = os.join(aug_folder, modality)
        os.mkdir(rgb_aug_folder)
        cv2.imwrite(os.join(rgb_aug_folder, img_file), aug_img)
        
        # Rotate image
        i = 0
        for aug_img in rotations(img, angles):
            aug_folder = os.join(class_folder, '{}-rot_{}'.format(video_subject, angles[i]))
            os.mkdir(aug_folder)
            rgb_aug_folder = os.join(aug_folder, modality)
            os.mkdir(rgb_aug_folder)
            cv2.imwrite(os.join(rgb_aug_folder, img_file), aug_img)
            i += 1
        
        # Rescale image
        i = 0
        for aug_img in rescalings(img, scales):
            aug_folder = os.join(class_folder, '{}-scale_{}'.format(video_subject, scales[i]))
            os.mkdir(aug_folder)
            rgb_aug_folder = os.join(aug_folder, modality)
            os.mkdir(rgb_aug_folder)
            cv2.imwrite(os.join(rgb_aug_folder, img_file), aug_img)
            i += 1
        
        # Brightness modifying
        i = 0
        for aug_img in brightness_modifyings(img, bright_coeffs):
            bright_folder = os.join(class_folder, '{}-bright_{}'.format(video_subject, bright_coeffs[i]))
            os.mkdir(aug_folder)
            rgb_aug_folder = os.join(aug_folder, modality)
            os.mkdir(rgb_aug_folder)
            cv2.imwrite(os.join(rgb_aug_folder, img_file), aug_img)
            i += 1


if __name__ == "__main__":    
    opt = parse_opts()
    
    angles = [-10, -5, 5, 10]
    scales = [-10, -5, 5, 10]
    bright_coeffs = [0.8, 0.9, 1.1, 1.2]
    
    # modality_folders = os.listdir(opt.dataset_path)
    modality_folders = ['RGB-D_frames', 'OF_frames']    # flipping, rotating, rescaling, brightness
    # modality_folders = ['MHI_frames']                   # flipping, rotating, rescaling
    
    for folder in modality_folders:
        modality_folder = os.join(opt.dataset_path, folder)
        
        class_iter = tqdm(os.listdir(modality_folder))
        for video_class in class_iter:
            class_folder = os.join(modality_folder, video_class)
            
            for video_subject in os.listdir(class_folder):
                subject_folder = os.join(class_folder, video_subject)
                
                augment_data(class_folder, video_subject, 'sk_color')
                augment_data(class_folder, video_subject, 'sk_depth')