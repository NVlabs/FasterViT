# FasterViT: Fast Vision Transformers with Hierarchical Attention
## Object Tracking with MOTRv2

Official PyTorch implementation of [**FasterViT: Fast Vision Transformers with Hierarchical Attention**](https://arxiv.org/abs/2306.06189).

---

In this section, we introduce the FasterViT object tracking repository with [MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors](https://arxiv.org/abs/2211.09791).

## Main Results

### DanceTrack

The FasterViT-4-21K-224 model demonstrated superior performance in both the validation and test datasets, outperforming the MOTRv2 algorithm with ResNet50.

| Backbone/Train Recipes | Validset | Testset | Backbone/Train Recipes | Validset | Testset |
|:----------------------:|:--------:|:-------:|:----------------------:|:-------:|:-------:|
| ResNet50 | 65.3 | 69.9 | <sup>*</sup>FasterViT | 67.4 (2.1↑) | 71.0 (1.1↑) |
| ResNet50 + <sup>*</sup>TrainVal | - | 70.9 | FasterViT + TrainVal ([model](https://drive.google.com/file/d/1LilNKtkUfmbaFsVtNIbrOqJA2IVKIYTb/view?usp=sharing)) | - | 73.7 (2.8↑) |
| ResNet50 + TrainVal + 4 Model Ensemble | - | 72.9 | FasterViT + TrainVal + 4 Model Ensemble | - | WIP |
| ResNet50 + TrainVal + 4 Model Ensemble + Extra Association | - | 73.4 | FasterViT + TrainVal + 4 Model Ensemble + Extra Association | - | WIP |

<sup>*</sup> *TrainVal: Jointly trained on the training and validation sets.*

<sup>*</sup> *FasterViT: Utilized the FasterViT-4-21K-224 model as the backbone.*

## Installation and Dataset Preparation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [MOTRv2](https://github.com/megvii-research/MOTRv2).
We recommend following the [installation instructions](https://github.com/megvii-research/MOTRv2?tab=readme-ov-file#installation) and [dataset preparation](https://github.com/megvii-research/MOTRv2?tab=readme-ov-file#dataset-preparation).


## Usage

### Training

To initiate training, start by downloading the pretrained weights for COCO from [Deformable DETR (+ iterative bounding box refinement)](https://github.com/fundamentalvision/Deformable-DETR#:~:text=config%0Alog-,model,-%2B%2B%20two%2Dstage%20Deformable). Then, modify the `--pretrained` argument with the path to the downloaded weights. Proceed to train MOTR on 8 GPUs using the following command:

```bash 
./tools/ddp_train.sh downstream/object_tracking/motrv2/configs/motrv2.args downstream/object_tracking/motrv2/results /data/Dataset/mot /data/Dataset/mot/det_db_motrv2.json 8
```

### Inference on DanceTrack Testset

For running inference on the DanceTrack testset with multiple trained weights, execute the following command:

```bash
# ./tools/simple_inference_test_parallel.sh <config-file> <weights-dir> <gpu-num> <output-dir> <mot-path-dir> <db-file> 
./tools/simple_inference_test_parallel.sh configs/motrv2.args downstream/object_tracking/motrv2/results 8 submit/ /data/Dataset/mot /data/Dataset/mot/det_db_motrv2.json
```

## Acknowledgements

- [MOTR](https://github.com/megvii-research/MOTR)
- [MOTRv2](https://github.com/megvii-research/MOTRv2)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [BDD100K](https://github.com/bdd100k/bdd100k)
