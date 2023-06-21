import torch
import timm
import io
import onnx
import os
from fastervit.models.faster_vit import *
from fastervit.models.faster_vit_any_res import *

def main():
    model_name='faster_vit_0_224'
    resolution = 224

    model = timm.create_model(
        model_name,
        resolution=resolution,
        exportable=True)

    in_size = (1, 3, resolution, resolution)
    model = model.cuda()
    model.eval()
    imgs = torch.randn(in_size,
                       device="cuda",
                       requires_grad=True)

    export_onnx(model,
                imgs,
                onnx_file_name=model_name+'.onnx',
                export_params=True)

def export_onnx(
    model: torch.nn.Module,
    sample_inputs,
    export_params: bool = False,
    opset_version: int = 13,
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
    onnx.save(
        onnx_model,
        os.path.join(
            result_dir, onnx_file_name
        ),
    )
    return onnx_model


if __name__ == "__main__":
    main()
