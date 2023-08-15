import torch
import coremltools
from fastervit import create_model

model = create_model('faster_vit_0_224').eval()
input_size = 224
bs_size = 1
file_name = 'faster_vit_0_224.mlmodel'
img = torch.randn((bs_size, 3, input_size, input_size), dtype=torch.float32)
model_jit_trace = torch.jit.trace(model, img)
model = coremltools.convert(model_jit_trace, inputs=[coremltools.ImageType(shape=img.shape)])
model.save(file_name)
