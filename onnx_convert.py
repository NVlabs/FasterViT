# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import timm
import io
import onnx
import os
import argparse
import onnx_graphsurgeon as gs
from fastervit.models.faster_vit import *
from fastervit.models.faster_vit_any_res import *

parser = argparse.ArgumentParser(description='Export FasterVit model XYZ to ONNX file XYZ.onnx')
parser.add_argument('--model-name', type=str, default="faster_vit_0_224")
parser.add_argument('--result-dir', type=str, default="./")
parser.add_argument('--onnx-opset', type=int, default=17)
parser.add_argument('--resolution-h', type=int, default=224)
parser.add_argument('--resolution-w', type=int, default=224)
args = parser.parse_args()

def main():
    model_name = args.model_name
    resolution_h = args.resolution_h
    resolution_w = args.resolution_w
    onnx_opset = args.onnx_opset
    result_dir = args.result_dir
    if "_224" in model_name:
        assert resolution_h == resolution_w

    model = timm.create_model(
        model_name,
        resolution=resolution_h if "_224" in model_name else [resolution_h, resolution_w],
        exportable=True)

    in_size = (1, 3, resolution_h, resolution_w)
    model = model.cuda()
    model.eval()
    imgs = torch.randn(in_size,
                       device="cuda",
                       requires_grad=True)

    export_onnx(model,
                imgs,
                onnx_file_name=model_name+'.onnx',
                export_params=True,
                opset_version=onnx_opset,
                result_dir=result_dir)

def export_onnx(
    model: torch.nn.Module,
    sample_inputs,
    export_params: bool = False,
    opset_version: int = 17,
    result_dir: str = "",
    batch_first: bool = True,
    is_training: bool = False,
    onnx_file_name: str ="",
):
    f = io.BytesIO()
    torch.onnx.export(
        model,
        # ONNX has issue to unpack the tuple of parameters to the model.
        # https://github.com/pytorch/pytorch/issues/11456
        (sample_inputs,) if type(sample_inputs) == tuple else sample_inputs,
        f,
        export_params=export_params,
        training=torch.onnx.TrainingMode.TRAINING
        if is_training
        else torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=["input"] if batch_first else None,
        output_names=["output"] if batch_first else None,
        dynamic_axes={"input": [0], "output": [0]} if batch_first else None,
    )
    onnx_model = onnx.load_model_from_string(f.getvalue(), onnx.ModelProto)
    f.close()
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    # Constant folding to simplify the ONNX
    graph = gs.import_onnx(onnx_model)
    graph.fold_constants().cleanup()

    onnx.save(
        gs.export_onnx(graph),
        os.path.join(
            result_dir, onnx_file_name
        ),
    )
    return onnx_model


if __name__ == "__main__":
    main()
