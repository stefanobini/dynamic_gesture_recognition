# Benchmarking deep neural networks for gesture recognition on embedded devices

Implementation of the framework used in [Benchmarking deep neural networks for gesture recognition on embedded devices](link). The workflow is developed to perform a benchmarking focusing not only on the accuracy but also on the computational burden, involving two different architectures (2D and 3D), with two different backbones (MobileNet, ResNeXt) and four types of input modalities (RGB, Depth, Optical Flow, Motion History Image) and their combinations.
The system performs gesture recognition following the following scheme.

![alt text](https://github.com/stefanobini/gesture_recognition/blob/main/figures/workflow.png)

Modalities can refer to data acquired directly by sensors (RGB or Depth stream) or by pre-processing raw data so as to obtain for instance OF or MHI. Each modality feeds a deep learning based feature extractor, which takes into account both spatial and temporal information. Finally, the classification is performed over the different feature vectors extracted and a fusion at decision level is performed. The dashed edges of a module refers to the fact that it is not mandatory.


## Results
The results analyze are analyzed on ChaLearn LAP IsoGD and NVIDIA Dynamic Gesture dataset, and are accuracy of the classification, memory occupancy and inference time of the model.

The following tables show the experimental results in terms of \textit{percentage accuracy}. The analyzed modalities are: RGB, Depth (D), dense Optical Flow (OF), Motion History Image (MHI), and their combinations. Tested networks are 2D and 3D versions of MobileNet v2 and ResNeXt-101.

### ChaLearn LAP IsoGD
![alt text](https://github.com/stefanobini/gesture_recognition/blob/main/figures/isogd_results.png)

### NVIDIA Dynamic Gesture
![alt text](https://github.com/stefanobini/gesture_recognition/blob/main/figures/nvgesture_results.png)

### Computational Cost
Memory requirements and time analysis for feature extraction and classification steps. The numbered columns, from 1 to 4, indicate the number of streams of the multi-stream network (one for each modality).

![alt text](https://github.com/stefanobini/gesture_recognition/blob/main/figures/comp_cost_results.png)

## Datasets
The framework can work on any dataset, but the dataloaders have been implemented only for the most interesting datasets, namely: 20BN-Jester, ChaLearn LAP IsoGD, and NVIDIA Dynamic Gesture.
Such dataloaders expect to find datasets in their respective formats presented below. 

20BN-Jester
```bash
|- RGB_frames
|--- 1
|----- 00000.jpg
|----- ...
|----- n_frames
|--- ...
```

ChaLearn LAP IsoGD
```bash
|- MHI_frames
|--- test
|----- 001
|------- K_00000
|--------- image_00000.jpg
|--------- ...
|--------- n_frames
|------- ...
|------- M_00000
|--------- ...
|--- ...
|--- train
|----- ...
|--- valid
|----- ...
|- OF_frames
|--- ...
|- RGB-D_frames
|--- ...
```

NVIDIA Dynamic Gesture
```bash
|- MHI_frames
|---- class_01
|------ subject10_r0
|-------- sk_color
|---------- image_00001.jpg
|---------- ...
|---------- n_frames
|-------- sk_dept
|------...
|- OF_frames
|--- ...
|- RGB-D_frames
|--- ...
```

## Requirements
The main requirements are include in the following list.

```bash
av==8.0.1
einops==0.3.2
fvcore==0.1.5
matplotlib==3.4.2
numpy==1.19.5
opencv_python==4.5.2.54
pandas==1.3.0
Pillow==9.0.0
psutil==5.8.0
pytorch_model_summary==0.1.2
scikit_image==0.18.2
scikit_learn==1.0.2
scipy==1.7.1
setuptools==57.1.0
simplejson==3.17.6
skimage==0.0
slowfast==1.0
submitit==1.4.1
timm==0.4.12
torch==1.9.0
torchinfo==1.5.2
torchsummary==1.5.1
torchvision==0.10.0
tqdm==4.61.2
visdom==0.1.8.9
```

## Pre-training
All the results reported are obtained starting from model pretrained on the 20BN-Jester dataset.
The pre-trained models can be downloaded from [here](link).
The implemented models are:
- 2D - MobileNet V2
- 2D - ResNeXt-101
- 3D - MobileNet V2
- 3D - ResNeXt-101

The command for pre-train the system is the following.

```bash
python3 main.py \
  --root_path ./ \
  --video_path ../datasets/ \
  --annotation_path annotation_Jester/jester.json \
  --result_path results \
  --dataset jester \
  --n_classes 27 \
  --n_finetune_classes 27 \
  --ft_portion complete \
  --model resnext \
  --n_epochs 60 \
  --lr_steps 30 45 \
  --learning_rate 0.01 \
  --batch_size 32 \
  --n_threads 8 \
  --no_hflip \
  --modalities RGB \
  --mod_aggr none \
  --gpu 0
```

## Training
The training phase can be done with the following command. It contains the main required parameters.

```bash
python3 main.py \
  --root_path ./ \
  --video_path ../datasets/ \
  --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json \
  --result_path results \
  --pretrain_path pretrained_models/jester_resnext_101_RGB_16_best.pth \
  --dataset isogd \
  --n_classes 27 \
  --n_finetune_classes 249 \
  --ft_portion complete \
  --model resnext \
  --n_epochs 20 \
  --lr_steps 8 15 \
  --learning_rate 0.1 \
  --batch_size 32 \
  --n_threads 8 \
  --no_hflip \
  --modalities RGB \
  --mod_aggr none \
  --gpu 0
```


## Testing
The following command load a specific model for test the system.

```bash
python3 main.py \
  --root_path ./ \
  --video_path ../datasets/ \
  --annotation_path annotation_ChaLearn_IsoGD/chalearn_isogd.json \
  --result_path results \
  --pretrain_path results \
  --dataset isogd \
  --n_classes 249 \
  --n_finetune_classes 249 \
  --model resnext \
  --batch_size 16 \
  --n_threads 4 \
  --no_train \
  --no_val \
  --test \
  --test_subset test \
  --modalities RGB \
  --mod_aggr none \
  --preds_per_video 249 \
  --gpu 0
```

## Citation
Please cite the following [paper](link) if you feel this repository useful.
```bibtext
@article{bini2022benchmarking,
  author={Bini, Stefano and Greco, Antonio and Saggese, Alessia and Vento, Mario},
  booktitle={2022 31th IEEE International Conference on Robot   Human Interactive Communication (RO-MAN)},
  title={Benchmarking deep neural networks for gesture recognition on embedded devices},
  year={2022},
  volume={},
  number={},
  pages={},
  doi={}
```