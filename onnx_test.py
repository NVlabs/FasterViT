# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import onnx
import onnxruntime as ort

onnx_model = onnx.load("./faster_vit_0_224.onnx")
onnx.checker.check_model(onnx_model)
x = torch.randn((1, 3, 1024, 1024))
ort_sess = ort.InferenceSession('faster_vit_0_224.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})
print(f'Predicted shape: "{outputs[0].shape}""')
