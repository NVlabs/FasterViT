FROM nvcr.io/nvidia/pytorch:23.07-py3
RUN pip install tensorboardX==2.6.2.2
RUN pip install timm==0.9.6
RUN pip install onnx==1.14.1
RUN pip install onnx_graphsurgeon==0.3.27
RUN pip install onnxruntime==1.15.1
RUN pip install polygraphy==0.47.1

WORKDIR /app
COPY . /app
