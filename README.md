# Gesture Recognition benchmarking

## Requirements

## Pre-trained models

## Dataset Preparation

### 20BN-Jester
* Download videos [here](https://20bn.com/datasets/jester#download).
* Generate n_frames files using ```utils/n_frames_jester.py```

```bash
python utils/n_frames_jester.py dataset_directory
```
Used
```bash
python3 utils/n_frames_jester.py datasets/jester
```

* Generate annotation file in json format similar to ActivityNet using ```utils/jester_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist.txt, vallist.txt

```bash
python utils/jester_json.py annotation_dir_path
```
Used
```bash
python3 utils/jester_json.py annotation_Jester
```

### ChaLearn LAP IsoGD
* Download videos [here](http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html).
* Convert from avi to jpg files using ```utils/video_jpg_chalearn_isogd.py```

```bash
python utils/video_jpg_chalearn_isogd.py avi_video_directory jpg_video_directory
```
Used
```bash
python3 utils/video_jpg_chalearn_isogd.py datasets/chalearn_isogd datasets/chalearn_isogd_frame_video
```

* Generate n_frames files using ```utils/n_frames_chalearn_isogd.py```

```bash
python utils/n_frames_chalearn_isogd.py jpg_video_directory
```
Used
```bash
python3 utils/n_frames_chalearn_isogd.py datasets/chalearn_isogd_frame_video
```

* Generate annotation file in json format similar to ActivityNet using ```utils/chalearn_isogd_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist.txt, vallist.txt, testlist.txt

```bash
python utils/chalearn_isogd_json.py annotation_dir_path
```
Used
```bash
python3 utils/chalearn_isogd_json.py annotation_ChaLearn_IsoGD
```

## Training

### ChaLearn LAP IsoGD
* Training from pre-trained model
```bash
python3 main.py --root_path ./ \
	--video_path datasets/chalearn_isogd_frame_video \
	--annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json \
	--result_path results/chalearn_isogd \
	--pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth \
	--dataset isogd \
	--n_classes 27 \
	--n_finetune_classes 249 \
	--ft_portion last_layer \
	--model resnext \
    --model_depth 101 \
	--groups 3 \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 16 \
	--downsample 1 \
	--batch_size 64 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
```

```bash
RESNEXT-101
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --aggr_type none

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities D --aggr_type none

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities MHI --aggr_type none
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities MHI_D --aggr_type none

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities OF --aggr_type none
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities OF_D --aggr_type none


python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB D --aggr_type avg --SSA_loss

MOBILENET V2
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality D

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/MHI_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality MHI
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/MHI_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality MHI_D

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF_D

Res3D+ConvLSTM+MobileNet
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --model res3d_clstm_mn --train_crop random --scale_step 0.95 --n_epochs 30 --lr_linear_decay 0.001 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --aggr_type none

RAAR3DNet
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --model raar3d --train_crop random --scale_step 0.95 --n_epochs 50 --lr_patience 4 --learning_rate 0.01 --weight_decay 0.0003 --sample_size 224 --sample_duration 16 --downsample 2 --batch_size 8 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --aggr_type none

```

### NVIDIA Gesture
```bash
 -W ignore
RESNEXT-101
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality D

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/OF_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/OF_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF_D

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/MHI_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality MHI
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/MHI_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality MHI_D

MOBILENET V2
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality D

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/OF_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/OF_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF_D

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/MHI_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality MHI
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/MHI_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model mobilenetv2 --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_patience 2 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 32 --checkpoint 1 --n_val_samples 1 --no_hflip --modality MHI_D
```


## Testing

### 20BN-Jester
```bash
python3 main.py --root_path ./ --video_path datasets/jester --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 64 --n_threads 16 --test --model_depth 101 --no_train --no_val --test_subset val --preds_per_video 5 --ft_portion none
python3 main.py --root_path ./ --video_path ../../Efficient-3DCNNs-master/datasets/jester --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path ../../Efficient-3DCNNs-master/pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --model mobilenetv2 --width_mult 1.0 --sample_duration 16 --downsample 2 --batch_size 64 --n_threads 16 --test --no_train --no_val --test_subset val --ft_portion none
```

### ChaLearn LAP IsoGD
```bash
RESNEXT-101
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_resnext_RGB_D_MLP_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB D --aggr_type MLP --feat_fusion --preds_per_video 249


python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_resnext_RGB_none_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --aggr_type none --preds_per_video 249

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_resnext_D_none_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities D --aggr_type none --preds_per_video 249

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/MHI_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_resnext_1.0x_MHI_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality MHI --preds_per_video 249
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/MHI_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_resnext_1.0x_MHI_D_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality MHI_D --preds_per_video 249

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_resnext_1.0x_OF_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality OF --preds_per_video 249
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_resnext_1.0x_OF_D_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality OF_D --preds_per_video 249

MOBILENET V2
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_mobilenetv2_1.0x_RGB_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality RGB --preds_per_video 249

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_mobilenetv2_1.0x_D_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality D --preds_per_video 249

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/MHI_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_mobilenetv2_1.0x_MHI_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality MHI --preds_per_video 249
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/MHI_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_mobilenetv2_1.0x_MHI_D_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality MHI_D --preds_per_video 249

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_mobilenetv2_1.0x_OF_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality OF --preds_per_video 249
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_mobilenetv2_1.0x_OF_D_16_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality OF_D --preds_per_video 249

Res3D+ConvLSTM+MobileNet

```

### NVIDIA Gesture
```bash
RESNEXT-101
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_resnext_1.0x_RGB_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality RGB --preds_per_video 25

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_resnext_1.0x_D_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality D --preds_per_video 25

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/OF_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_resnext_1.0x_OF_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality OF --preds_per_video 25
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/OF_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_resnext_1.0x_OF_D_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality OF_D --preds_per_video 25

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/MHI_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_resnext_1.0x_MHI_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality MHI --preds_per_video 25
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/MHI_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_resnext_1.0x_MHI_D_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model resnext --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality MHI_D --preds_per_video 25

MOBILENET V2
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_mobilenetv2_1.0x_RGB_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality RGB --preds_per_video 25

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_mobilenetv2_1.0x_D_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality D --preds_per_video 25

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/OF_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_mobilenetv2_1.0x_OF_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality OF --preds_per_video 25
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/OF_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_mobilenetv2_1.0x_OF_D_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality OF_D --preds_per_video 25

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/MHI_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_mobilenetv2_1.0x_MHI_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality MHI --preds_per_video 25
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/MHI_frames --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path results/nvgesture/nvgesture_mobilenetv2_1.0x_MHI_D_16_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model mobilenetv2 --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 32 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modality MHI_D --preds_per_video 25
```

# 2D-CNNs Framework


## Pre-train on 20BN-Jester
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester/RGB_frames --annotation_path annotation_Jester/jester.json --result_path results/jester --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --aggr_type LSTM
```
Test pretraining
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester/RGB_frames --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_avg_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --cnn_dim 2 --model mobilenetv2_2d --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --no_train --no_val --test --test_subset test --preds_per_video 27 --no_hflip --modalities RGB --aggr_type avg
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester/RGB_frames --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_MLP_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --cnn_dim 2 --model mobilenetv2_2d --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --no_train --no_val --test --test_subset test --preds_per_video 27 --no_hflip --modality RGB --aggr_type MLP
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester/RGB_frames --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_LSTM_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --cnn_dim 2 --model mobilenetv2_2d --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --no_train --no_val --test --test_subset test --preds_per_video 27 --no_hflip --modality RGB --aggr_type LSTM
```

## Training


### ChaLearn LAP IsoGD

```bash
MOBILENET V2
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_avg_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion last_layer --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --aggr_type avg
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_MLP_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --aggr_type MLP
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_LSTM_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --aggr_type LSTM

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --aggr_type avg


python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_avg_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality D --aggr_type avg
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_MLP_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality D --aggr_type MLP
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_LSTM_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality D --aggr_type LSTM

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/RGB-D_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modality D --aggr_type avg


python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_avg_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF --aggr_type avg
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_MLP_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF --aggr_type MLP
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_LSTM_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF --aggr_type LSTM

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/OF_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modality OF --aggr_type avg


python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd/MHI_frames --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modality MHI --aggr_type avg

```

### NVIDIA Gesture

```bash
MOBILENET V2
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_avg_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 25 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --aggr_type avg
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_MLP_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 25 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --aggr_type MLP
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/nvgesture/RGB-D_frames_aug --annotation_path annotation_NVGesture/nvgesture.json --result_path results/nvgesture --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_LSTM_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 25 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 30 --lr_patience 3 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --aggr_type LSTM

```

## Testing

### ChaLearn LAP IsoGD
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_mobilenetv2_2d_RGB_avg_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --cnn_dim 2 --model mobilenetv2_2d --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --aggr_type avg --preds_per_video 249
```