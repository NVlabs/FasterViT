# FasterViT: Fast Vision Transformers with Hierarchical Attention
## Object Detection with DINO

Official PyTorch implementation of [**FasterViT: Fast Vision Transformers with Hierarchical Attention**](https://arxiv.org/abs/2306.06189).

--- 

In this section, we present FasterViT object detection repository with [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605).

Please stay tuned for more pre-trained checkpoints ! 

## Model Zoo 

### 12 epoch setting


<table>
  <tr>
    <th>Model</th>
    <th>Backbone</th>
    <th>Box AP</th>
    <th>Download</th>
  </tr>

<tr>
    <td>DINO-4scale</td>
    <td>FasterViT-4-21K-224</td>
    <td>55.16</td>
    <td><a href="https://huggingface.co/ahatamiz/FasterViT/resolve/main/DINO_4scale_faster_vit_4_21k_224_ms_coco.pth">model</a></td>
</tr>

</table>



## Licenses

Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

For license information regarding the DINO repository, please refer to its [repository](https://github.com/IDEA-Research/DINO).


## Acknowledgement
This repository is built on top of the [DINO repository](https://github.com/IDEA-Research/DINO) repository. We thank the authors for their amazing work and releasing their code base. 