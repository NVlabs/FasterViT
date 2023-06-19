# Training Instructions

In this section, we provide exact training details of all FasterViT models. We use 4 nodes (8 x A100 NVIDIA GPUs) for 
training all models, unless otherwise specified.  


## ImageNet-1K Training 
A FasterViT model can be trained by specifying the model variant `--model` (or config file `--config`), with local (per GPU)
batch size `--batch-size`, drop path rate `--drop-path-rate`, optimizer `--opt`, [MESA](https://arxiv.org/pdf/2205.14083.pdf) hyper-parameter
`--mesa`, weight `--weight-decay`, learning rate `--lr`, input with image resolution `--input-size` and by enabling
AMP `--amp` and EMA `--model-ema`. The experiment name and data directory are passed by specifying `--tag` and `--data_dir`
, respectively.

For all experiments, the effective batch size  is `--nnodes` * `--batch_size` * `4` which is `8*128*4 = 4096`. 

FasterViT-0 - Hardware: 4 x 8 A100 (40G) NVIDIA GPUs

```
torchrun --nnodes=4 --nproc_per_node=8 train.py \
--config configs/faster_vit_0_224_1k.yml
--model faster_vit_0_224
--tag faster_vit_0_224_exp_1 
--batch-size 128
--drop-path-rate 0.2
--lr 0.005
--mesa 0.1
--model-ema
--opt adamw
--weight-decay 0.005
--amp
--input-size 3 224 224
--data_dir "/imagenet/ImageNet2012" \
``` 

FasterViT-1 - Hardware: 4 x 8 A100 (40G) NVIDIA GPUs

```
torchrun --nnodes=4 --nproc_per_node=8 train.py \
--config configs/faster_vit_1_224_1k.yml
--model faster_vit_1_224
--tag faster_vit_1_224_exp_1 
--batch-size 128
--drop-path-rate 0.2
--lr 0.005
--mesa 0.2
--model-ema
--opt adamw
--weight-decay 0.005
--amp
--input-size 3 224 224
--data_dir "/imagenet/ImageNet2012" \
``` 


FasterViT-2 - Hardware: 4 x 8 A100 (40G) NVIDIA GPUs

```
torchrun --nnodes=4 --nproc_per_node=8 train.py \
--config configs/faster_vit_2_224_1k.yml
--model faster_vit_2_224
--tag faster_vit_2_224_exp_1 
--batch-size 128
--drop-path-rate 0.2
--lr 0.005
--mesa 0.5
--model-ema
--opt adamw
--weight-decay 0.005
--amp
--input-size 3 224 224
--data_dir "/imagenet/ImageNet2012" \
``` 

FasterViT-3 - Hardware: 4 x 8 A100 (40G) NVIDIA GPUs

```
torchrun --nnodes=4 --nproc_per_node=8 train.py \
--config configs/faster_vit_3_224_1k.yml
--model faster_vit_3_224
--tag faster_vit_3_224_exp_1 
--batch-size 128
--drop-path-rate 0.3
--lr 0.005
--mesa 0.5
--model-ema
--opt adamw
--weight-decay 0.005
--amp
--input-size 3 224 224
--data_dir "/imagenet/ImageNet2012" \
``` 

FasterViT-4 - Hardware: 4 x 8 A100 (80G) NVIDIA GPUs

```
torchrun --nnodes=4 --nproc_per_node=8 train.py \
--config configs/faster_vit_4_224_1k.yml
--model faster_vit_4_224
--tag faster_vit_4_224_exp_1 
--batch-size 128
--drop-path-rate 0.3
--lr 0.005
--mesa 5.0
--model-ema
--opt lamb
--weight-decay 0.12
--amp
--input-size 3 224 224
--data_dir "/imagenet/ImageNet2012" \
``` 

FasterViT-5 - Hardware: 4 x 8 A100 (80G) NVIDIA GPUs

```
torchrun --nnodes=4 --nproc_per_node=8 train.py \
--config configs/faster_vit_5_224_1k.yml
--model faster_vit_5_224
--tag faster_vit_5_224_exp_1 
--batch-size 128
--drop-path-rate 0.3
--lr 0.005
--mesa 5.0
--model-ema
--opt lamb
--weight-decay 0.12
--amp
--input-size 3 224 224
--data_dir "/imagenet/ImageNet2012" \
``` 

FasterViT-6 - Hardware: 4 x 8 A100 (80G) NVIDIA GPUs

```
torchrun --nnodes=4 --nproc_per_node=8 train.py \
--config configs/faster_vit_6_224_1k.yml
--model faster_vit_6_224
--tag faster_vit_6_224_exp_1 
--batch-size 128
--drop-path-rate 0.5
--lr 0.005
--mesa 5.0
--model-ema
--opt lamb
--weight-decay 0.12
--amp
--input-size 3 224 224
--data_dir "/imagenet/ImageNet2012" \
``` 
### Why using MESA ? 

[MESA](https://arxiv.org/pdf/2205.14083.pdf) is an effective way of addressing the overfitting issue which can hinder the peformance, especially for larger models. The idea is to start training without MESA until reaching a quarter of all training epochs (the exact starting point of MESA is also a hyper-parameter which can be tuned [here](./fastervit/train.py#L351
)). At this stage, we use the EMA model to act as a teacher and help the base model to learn more robust representations. We have tuned MESA hyper-parameters for various FasterViT models, as shown above.

### LAMB optimizer

For larger FasterViT models (i.e. 4,5 and 6 variants), we use LAMB optimizer as it is more stable during training (e.g. avoiding NaN) and is suited better for larger batch sizes and learning rates. 


### Data Preparation

Please download the ImageNet dataset from its official website. The training and validation images need to have
sub-folders for each class with the following structure:

```bash
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
