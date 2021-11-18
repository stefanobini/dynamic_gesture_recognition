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
## Pre-training

### 20BN-Jester
ResNeXt-101
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --weight_decay 0.001 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none --gpu 3
```

MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester --annotation_path annotation_Jester/jester.json --result_path results/jester --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --model resnext --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --weight_decay 0.001 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none --gpu 3
```

Res3D + ConvLSTM + MobileNet
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester --annotation_path annotation_Jester/jester.json --result_path results/jester --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --model res3d_clstm_mn --train_crop random --scale_step 0.95 --n_epochs 30 --lr_linear_decay 0.001 --learning_rate 0.1 --sample_duration 16 --downsample 2 --batch_size 8 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none --gpu 2
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

ResNeXt-101
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model resnext --n_epochs 20 --lr_steps 8 15 --learning_rate 0.1 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path results --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --model resnext --n_epochs 20 --lr_steps 8 15 --learning_rate 0.1 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB D OF_D --mod_aggr MLP --gpu 1
```

MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model mobilenetv2 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.1 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path results --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --model mobilenetv2 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.1 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB D --mod_aggr MLP --gpu 3
```

Res3D+ConvLSTM+MobileNet
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path pretrained_models/jester_res3d_clstm_mn_RGB_none_best.pth --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --model res3d_clstm_mn --train_crop random --scale_step 0.95 --n_epochs 15 --lr_linear_decay 0.0001 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 8 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none --gpu 3
```

RAAR3DNet
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --model raar3d --train_crop random --scale_step 0.95 --n_epochs 50 --lr_patience 4 --learning_rate 0.01 --weight_decay 0.0003 --sample_size 224 --sample_duration 16 --downsample 2 --batch_size 8 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none
```

NI3D
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --model ni3d --train_crop random --scale_step 0.95 --n_epochs 50 --lr_patience 5 --learning_rate 0.01 --weight_decay 0.0003 --sample_size 224 --sample_duration 32 --downsample 1 --batch_size 8 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none --gpu 3

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path results/chalearn_isogd/ni3d/isogd_ni3d_RGB_none_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --ft_portion complete --model raar3d --train_crop random --scale_step 0.95 --n_epochs 15 --lr_patience 2 --learning_rate 0.01 --weight_decay 0.0003 --sample_size 224 --sample_duration 32 --downsample 1 --batch_size 8 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --mod_aggr none --gpu 3
```

### NVIDIA Gesture
 -W ignore
ResNeXt-101
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model resnext --n_epochs 35 --lr_steps 17 27 --learning_rate 0.1 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities MHI_D --mod_aggr none --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --pretrain_path results --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --ft_portion complete --model resnext --n_epochs 35 --lr_steps 17 27 --learning_rate 0.1 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB MHI --mod_aggr MLP --gpu 1
```

MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset nvgesture --n_classes 27 --n_finetune_classes 25 --ft_portion complete --model mobilenetv2 --n_epochs 35 --lr_steps 17 27 --learning_rate 0.1 --batch_size 32 --n_threads 16 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities MHI_D --mod_aggr none --gpu 2
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --pretrain_path results --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --ft_portion complete --model mobilenetv2 --n_epochs 35 --lr_steps 17 27 --learning_rate 0.1 --batch_size 32 --n_threads 4 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB MHI --mod_aggr MLP --gpu 1
```


## Testing

### 20BN-Jester
ResNeXt-101
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_Jester/jester.json --result_path results --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --model resnext --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 16 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --mod_aggr none --preds_per_video 27 --gpu 3
```

MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_Jester/jester.json --result_path results --pretrain_path pretrained_models/jester_mobilenetv2_1.0x_RGB_16_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --model mobilenetv2 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 16 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --mod_aggr none --preds_per_video 27 --gpu 3
```

### ChaLearn LAP IsoGD
ResNeXt-101
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path results --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --batch_size 16 --n_threads 4 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --mod_aggr none --preds_per_video 249 --gpu 3

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path results/isogd/resnext/isogd_resnext_D_OF_D_MLP_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model resnext --batch_size 16 --n_threads 4 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities D OF_D --mod_aggr MLP --preds_per_video 249 --gpu 3
```

MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path results --dataset isogd --n_classes 249 --n_finetune_classes 249 --model mobilenetv2 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --mod_aggr none --preds_per_video 249 --gpu 3

python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --pretrain_path results/isogd/mobilenetv2/isogd_mobilenetv2_RGB_OF_D_MLP_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --model mobilenetv2 --batch_size 16 --n_threads 8 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB OF_D --mod_aggr MLP --preds_per_video 249 --gpu 1
```

```
Res3D+ConvLSTM+MobileNet
```

### NVIDIA Gesture
ResNeXt-101
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --pretrain_path results --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model resnext --batch_size 16 --n_threads 16 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --mod_aggr none --preds_per_video 25 --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --pretrain_path results/nvgesture/resnext/nvgesture_resnext_D_OF_D_MLP_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model resnext --batch_size 16 --n_threads 8 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities D OF_D --mod_aggr MLP --preds_per_video 25 --gpu 3
```

MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --pretrain_path results --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model mobilenetv2 --batch_size 16 --n_threads 16 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --mod_aggr none --preds_per_video 25 --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --pretrain_path results/nvgesture/mobilenetv2/nvgesture_mobilenetv2_D_OF_MLP_best.pth --dataset nvgesture --n_classes 25 --n_finetune_classes 25 --model mobilenetv2 --batch_size 16 --n_threads 8 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities D OF --mod_aggr MLP --preds_per_video 25 --gpu 3
```

# 2D-CNNs Framework


## Pre-train on 20BN-Jester
ResNeXt-101
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_Jester/jester.json --result_path results --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --cnn_dim 2 --model resnext_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 4 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr avg --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_Jester/jester.json --result_path results --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --cnn_dim 2 --model resnext_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr MLP --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_Jester/jester.json --result_path results --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --cnn_dim 2 --model resnext_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr LSTM --gpu 3
```

MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_Jester/jester.json --result_path results --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr avg --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_Jester/jester.json --result_path results --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr MLP --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_Jester/jester.json --result_path results --dataset jester --n_classes 27 --n_finetune_classes 27 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 60 --lr_steps 30 45 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr LSTM --gpu 3
```

Test pretraining
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester/RGB_frames --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_avg_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --cnn_dim 2 --model mobilenetv2_2d --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --no_train --no_val --test --test_subset test --preds_per_video 27 --no_hflip --modalities RGB --temp_aggr avg --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester/RGB_frames --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_MLP_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --cnn_dim 2 --model mobilenetv2_2d --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --no_train --no_val --test --test_subset test --preds_per_video 27 --no_hflip --modality RGB --temp_aggr MLP --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/jester/RGB_frames --annotation_path annotation_Jester/jester.json --result_path results/jester --pretrain_path pretrained_models/jester_mobilenetv2_2d_1.0x_RGB_16_LSTM_best.pth --dataset jester --n_classes 27 --n_finetune_classes 27 --cnn_dim 2 --model mobilenetv2_2d --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 8 --no_train --no_val --test --test_subset test --preds_per_video 27 --no_hflip --modality RGB --temp_aggr LSTM --gpu 3
```

## Training


### ChaLearn LAP IsoGD
ResNeXt-101
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model resnext_2d --model_depth 101 --groups 3 --train_crop random --scale_step 0.95 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr avg --gpu 3
```

MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr avg --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr MLP --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results --dataset isogd --n_classes 27 --n_finetune_classes 249 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modalities RGB --temp_aggr LSTM --gpu 3
```

### NVIDIA Gesture
MobileNet v2
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --dataset isogd --n_classes 27 --n_finetune_classes 25 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --temp_aggr avg --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --dataset isogd --n_classes 27 --n_finetune_classes 25 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --temp_aggr MLP --gpu 3
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini --annotation_path annotation_NVGesture/nvgesture.json --result_path results --dataset isogd --n_classes 27 --n_finetune_classes 25 --ft_portion complete --cnn_dim 2 --model mobilenetv2_2d --train_crop random --scale_step 0.95 --n_epochs 20 --lr_steps 8 15 --learning_rate 0.01 --sample_duration 16 --downsample 2 --batch_size 32 --n_threads 7 --checkpoint 1 --n_val_samples 1 --no_hflip --modality RGB --temp_aggr LSTM --gpu 3
```

## Testing

### ChaLearn LAP IsoGD
```bash
python3 main.py --root_path ./ --video_path ../../../../mnt/sdc1/sbini/isogd --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json --result_path results/chalearn_isogd --pretrain_path results/chalearn_isogd/isogd_mobilenetv2_2d_RGB_avg_best.pth --dataset isogd --n_classes 249 --n_finetune_classes 249 --cnn_dim 2 --model mobilenetv2_2d --groups 3 --sample_duration 16 --downsample 2 --batch_size 16 --n_threads 8 --model_depth 101 --n_val_samples 1 --no_train --no_val --test --test_subset test --modalities RGB --temp_aggr avg --preds_per_video 249
```