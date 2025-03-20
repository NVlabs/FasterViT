# FasterViT: Fast Vision Transformers with Hierarchical Attention

Official PyTorch implementation of [**FasterViT: Fast Vision Transformers with Hierarchical Attention**](https://arxiv.org/abs/2306.06189).

[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/FasterViT.svg?style=social)](https://github.com/NVlabs/FasterViT/stargazers)

[Ali Hatamizadeh](https://research.nvidia.com/person/ali-hatamizadeh),
[Greg Heinrich](https://developer.nvidia.com/blog/author/gheinrich/),
[Hongxu (Danny) Yin](https://hongxu-yin.github.io/),
[Andrew Tao](https://developer.nvidia.com/blog/author/atao/),
[Jose M. Alvarez](https://alvarezlopezjosem.github.io/),
[Jan Kautz](https://jankautz.com/), 
[Pavlo Molchanov](https://www.pmolchanov.com/).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

--- 

FasterViT achieves a new SOTA Pareto-front in
terms of Top-1 accuracy and throughput without extra training data !

<p align="center">
<img src="https://github.com/NVlabs/FasterViT/assets/26806394/6357de9e-5d7f-4e03-8009-2bad1373096c" width=62% height=62% 
class="center">
</p>

We introduce a new self-attention mechanism, denoted as Hierarchical
Attention (HAT), that captures both short and long-range information by learning
cross-window carrier tokens.

![teaser](./fastervit/assets/hierarchial_attn.png)

Note: Please use the [**latest NVIDIA TensorRT release**](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html) to enjoy the benefits of optimized FasterViT ops. 

## 💥 News 💥
- **[03.25.2025]** We have updated the download links for each model. All models are accecible via HuggingFace. 
- **[04.02.2024]** 🔥 Updated [manuscript](https://arxiv.org/abs/2306.06189) now available on arXiv !
- **[01.24.2024]** 🔥🔥🔥 **Object Tracking with MOTRv2 + FasterViT** is now open-sourced ([link](./downstream/object_tracking/motrv2/README.md)) ! 
- **[01.17.2024]** 🔥🔥🔥 FasterViT paper has been accepted to [ICLR 2024](https://openreview.net/group?id=ICLR.cc/2024/Conference#tab-your-consoles) !
- **[10.14.2023]** 🔥🔥 We have added the FasterViT [object detection repository](./downstream/object_detection/dino/README.md) with [DINO](https://arxiv.org/abs/2203.03605) !
- **[08.24.2023]** 🔥 FasterViT Keras models with pre-trained weights published in [keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/fastervit) !  
- **[08.20.2023]** 🔥🔥 We have added ImageNet-21K SOTA pre-trained models for various resolutions !   
- **[07.20.2023]** We have created official NVIDIA FasterViT [HuggingFace](https://huggingface.co/nvidia/FasterViT) page.
- **[07.06.2023]** FasterViT checkpoints are now also accecible in HuggingFace!
- **[07.04.2023]** ImageNet pretrained FasterViT models can now be imported with **1 line of code**. Please install the latest FasterViT pip package to use this functionality (also supports Any-resolution FasterViT models).
- **[06.30.2023]** We have further improved the [TensorRT](https://developer.nvidia.com/tensorrt-getting-started) throughput of FasterViT models by 10-15% on average across different models. Please use the [**latest NVIDIA TensorRT release**](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html) to use these throughput performance gains. 
- **[06.29.2023]** Any-resolution FasterViT model can now be intitialized from pre-trained ImageNet resolution (224 x 244) models.
- **[06.18.2023]** We have released the FasterViT [pip package](https://pypi.org/project/fastervit/) !
- **[06.17.2023]** [Any-resolution FasterViT](./fastervit/models/faster_vit_any_res.py)  model is now available ! the model can be used for variety of applications such as detection and segmentation or high-resolution fine-tuning with arbitrary input image resolutions.
- **[06.09.2023]** 🔥🔥 We have released source code and ImageNet-1K FasterViT-models !

## Quick Start

### Object Detection

Please see FasterViT [object detection repository](./object_detection/README.md) with [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605) for more details. 

### Classification

We can import pre-trained FasterViT models with **1 line of code**. Firstly, FasterViT can be simply installed:

```bash
pip install fastervit
```
Note: Please upgrate the package to ```fastervit>=0.9.8``` if you have already installed the package to use the pretrained weights. 

A pretrained FasterViT model with default hyper-parameters can be created as in:

```python
>>> from fastervit import create_model

# Define fastervit-0 model with 224 x 224 resolution

>>> model = create_model('faster_vit_0_224', 
                          pretrained=True,
                          model_path="/tmp/faster_vit_0.pth.tar")
```

`model_path` is used to set the directory to download the model.

We can also simply test the model by passing a dummy input image. The output is the logits:

```python
>>> import torch

>>> image = torch.rand(1, 3, 224, 224)
>>> output = model(image) # torch.Size([1, 1000])
```

We can also use the any-resolution FasterViT model to accommodate arbitrary image resolutions. In the following, we define an any-resolution FasterViT-0
model with input resolution of 576 x 960, window sizes of 12 and 6 in 3rd and 4th stages, carrier token size of 2 and embedding dimension of
64:

```python
>>> from fastervit import create_model

# Define any-resolution FasterViT-0 model with 576 x 960 resolution
>>> model = create_model('faster_vit_0_any_res', 
                          resolution=[576, 960],
                          window_size=[7, 7, 12, 6],
                          ct_size=2,
                          dim=64,
                          pretrained=True)
```
Note that the above model is intiliazed from the original ImageNet pre-trained FasterViT with original resolution of 224 x 224. As a result, missing keys and mis-matches could be expected since we are addign new layers (e.g. addition of new carrier tokens, etc.) 

We can test the model by passing a dummy input image. The output is the logits:

```python
>>> import torch

>>> image = torch.rand(1, 3, 576, 960)
>>> output = model(image) # torch.Size([1, 1000])
```



## Catalog
- [x] ImageNet-1K training code
- [x] ImageNet-1K pre-trained models
- [x] Any-resolution FasterViT
- [x] FasterViT pip-package release
- [x] Add capablity to initialize any-resolution FasterViT from ImageNet-pretrained weights. 
- [x] ImageNet-21K pre-trained models
- [x] Detection code + models

--- 

## Results + Pretrained Models

### ImageNet-1K
**FasterViT ImageNet-1K Pretrained Models**

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>Throughput(Img/Sec)</th>
    <th>Resolution</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Download</th>
  </tr>

<tr>
    <td>FasterViT-0</td>
    <td>82.1</td>
    <td>95.9</td>
    <td>5802</td>
    <td>224x224</td>
    <td>31.4</td>
    <td>3.3</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_0_224_1k.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-1</td>
    <td>83.2</td>
    <td>96.5</td>
    <td>4188</td>
    <td>224x224</td>
    <td>53.4</td>
    <td>5.3</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_1_224_1k.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-2</td>
    <td>84.2</td>
    <td>96.8</td>
    <td>3161</td>
    <td>224x224</td>
    <td>75.9</td>
    <td>8.7</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_2_224_1k.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-3</td>
    <td>84.9</td>
    <td>97.2</td>
    <td>1780</td>
    <td>224x224</td>
    <td>159.5</td>
    <td>18.2</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_3_224_1k.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-4</td>
    <td>85.4</td>
    <td>97.3</td>
    <td>849</td>
    <td>224x224</td>
    <td>424.6</td>
    <td>36.6</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_224_1k.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-5</td>
    <td>85.6</td>
    <td>97.4</td>
    <td>449</td>
    <td>224x224</td>
    <td>975.5</td>
    <td>113.0</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_5_224_1k.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-6</td>
    <td>85.8</td>
    <td>97.4</td>
    <td>352</td>
    <td>224x224</td>
    <td>1360.0</td>
    <td>142.0</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_6_224_1k.pth.tar">model</a></td>
</tr>

</table>

### ImageNet-21K
**FasterViT ImageNet-21K Pretrained Models (ImageNet-1K Fine-tuned)**

<table>
  <tr>
    <th>Name</th>
    <th>Acc@1(%)</th>
    <th>Acc@5(%)</th>
    <th>Resolution</th>
    <th>#Params(M)</th>
    <th>FLOPs(G)</th>
    <th>Download</th>
  </tr>

<tr>
    <td>FasterViT-4-21K-224</td>
    <td>86.6</td>
    <td>97.8</td>
    <td>224x224</td>
    <td>271.9</td>
    <td>40.8</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_224_w14.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-4-21K-384</td>
    <td>87.6</td>
    <td>98.3</td>
    <td>384x384</td>
    <td>271.9</td>
    <td>120.1</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_384_w24.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-4-21K-512</td>
    <td>87.8</td>
    <td>98.4</td>
    <td>512x512</td>
    <td>271.9</td>
    <td>213.5</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_512_w32.pth.tar">model</a></td>
</tr>

<tr>
    <td>FasterViT-4-21K-768</td>
    <td>87.9</td>
    <td>98.5</td>
    <td>768x768</td>
    <td>271.9</td>
    <td>480.4</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_768_w48.pth.tar">model</a></td>
</tr>

</table>

Raw pre-trained ImageNet-21K model weights for FasterViT-4 is also available for download in this [link](https://drive.google.com/file/d/1T3jDrzlTmTcZVS1Dh01Fl3J2LXZHWKdL/view?usp=sharing).
### Robustness (ImageNet-A - ImageNet-R - ImageNet-V2)


All models use `crop_pct=0.875`. Results are obtained by running inference on ImageNet-1K pretrained models without finetuning.
<table>
  <tr>
    <th>Name</th>
    <th>A-Acc@1(%)</th>
    <th>A-Acc@5(%)</th>
    <th>R-Acc@1(%)</th>
    <th>R-Acc@5(%)</th>
    <th>V2-Acc@1(%)</th>
    <th>V2-Acc@5(%)</th>
  </tr>

<tr>
    <td>FasterViT-0</td>
    <td>23.9</td>
    <td>57.6</td>
    <td>45.9</td>
    <td>60.4</td>
    <td>70.9</td>
    <td>90.0</td>
</tr>

<tr>
    <td>FasterViT-1</td>
    <td>31.2</td>
    <td>63.3</td>
    <td>47.5</td>
    <td>61.9</td>
    <td>72.6</td>
    <td>91.0</td>
</tr>

<tr>
    <td>FasterViT-2</td>
    <td>38.2</td>
    <td>68.9</td>
    <td>49.6</td>
    <td>63.4</td>
    <td>73.7</td>
    <td>91.6</td>
</tr>

<tr>
    <td>FasterViT-3</td>
    <td>44.2</td>
    <td>73.0</td>
    <td>51.9</td>
    <td>65.6</td>
    <td>75.0</td>
    <td>92.2</td>
</tr>

<tr>
    <td>FasterViT-4</td>
    <td>49.0</td>
    <td>75.4</td>
    <td>56.0</td>
    <td>69.6</td>
    <td>75.7</td>
    <td>92.7</td>
</tr>

<tr>
    <td>FasterViT-5</td>
    <td>52.7</td>
    <td>77.6</td>
    <td>56.9</td>
    <td>70.0</td>
    <td>76.0</td>
    <td>93.0</td>
</tr>

<tr>
    <td>FasterViT-6</td>
    <td>53.7</td>
    <td>78.4</td>
    <td>57.1</td>
    <td>70.1</td>
    <td>76.1</td>
    <td>93.0</td>
</tr>

</table>

A, R and V2 denote ImageNet-A, ImageNet-R and ImageNet-V2 respectively. 

## Installation

We provide a [docker file](./Dockerfile). In addition, assuming that a recent [PyTorch](https://pytorch.org/get-started/locally/) package is installed, the dependencies can be installed by running:

```bash
pip install -r requirements.txt
```

## Training

Please see [TRAINING.md](TRAINING.md) for detailed training instructions of all models. 

## Evaluation

The FasterViT models can be evaluated on ImageNet-1K validation set using the following: 

```
python validate.py \
--model <model-name>
--checkpoint <checkpoint-path>
--data_dir <imagenet-path>
--batch-size <batch-size-per-gpu
``` 

Here `--model` is the FasterViT variant (e.g. `faster_vit_0_224_1k`), `--checkpoint` is the path to pretrained model weights, `--data_dir` is the path to ImageNet-1K validation set and `--batch-size` is the number of batch size. We also provide a sample script [here](./fastervit/validate.sh). 

## ONNX Conversion

We provide ONNX conversion script to enable dynamic batch size inference. For instance, to generate ONNX model for `faster_vit_0_any_res` with resolution 576 x 960 and ONNX opset number 17, the following can be used. 

```bash 
python onnx_convert --model-name faster_vit_0_any_res --simplify --resolution-h 576 --resolution-w 960 --onnx-opset 17

```

## CoreML Conversion

To generate FasterViT CoreML models, please install `coremltools==5.2.0` and use our provided [script](./coreml_convert.py). 

It is recommended to benchmark the performance by using [Xcode14](https://developer.apple.com/documentation/xcode-release-notes/xcode-14-release-notes) or newer releases. 



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=NVlabs/FasterViT&type=Date)](https://star-history.com/#NVlabs/FasterViT&Date)


## Third-party Extentions
We always welcome third-party extentions/implementations and usage for other purposes. The following represent third-party contributions by other users.

| Name | Link | Contributor | Framework
|:---:|:---:|:---:|:---------:|
|keras_cv_attention_models|[Link](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/fastervit)| @leondgarse | Keras

If you would like your work to be listed in this repository, please raise an issue and provide us with detailed information.  

## Citation

Please consider citing FasterViT if this repository is useful for your work. 

```
@article{hatamizadeh2023fastervit,
  title={FasterViT: Fast Vision Transformers with Hierarchical Attention},
  author={Hatamizadeh, Ali and Heinrich, Greg and Yin, Hongxu and Tao, Andrew and Alvarez, Jose M and Kautz, Jan and Molchanov, Pavlo},
  journal={arXiv preprint arXiv:2306.06189},
  year={2023}
}
```


## Licenses

Copyright © 2025, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

For license information regarding the timm repository, please refer to its [repository](https://github.com/rwightman/pytorch-image-models).

For license information regarding the ImageNet dataset, please see the [ImageNet official website](https://www.image-net.org/). 

## Acknowledgement
This repository is built on top of the [timm](https://github.com/huggingface/pytorch-image-models) repository. We thank [Ross Wrightman](https://rwightman.com/) for creating and maintaining this high-quality library.  
