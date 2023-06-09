FROM nvcr.io/nvidia/pytorch:23.01-py3
RUN pip install tensorboardX
RUN pip install pyyaml
RUN pip install yacs
RUN pip install termcolor
RUN pip install opencv-python
RUN pip install timm
WORKDIR /app
COPY . /app
