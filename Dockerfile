FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS conda_builder

ARG USER
ARG UID=1000
ARG CODE_SERVER_PORT=8888
ARG CMAKE_CUDA_ARCHITECTURES=86

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CODE_SERVER_PORT=${CODE_SERVER_PORT}
ENV USER=${USER}
ENV CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Use bash as the default shell
SHELL ["/bin/bash", "--login", "-c"]

# Copy Miniconda from base image
COPY --from=continuumio/miniconda3:23.10.0-1 /opt/conda /opt/conda

# Set environment paths
ENV PATH=/opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip ffmpeg libsm6 libxext6 wget git git-lfs \
    && pip3 install python-dotenv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Set up working directories
WORKDIR /home/user/worker
# Copy files
COPY demo/ ./demo/
# COPY results/ ./results/
COPY zim_anything/ ./zim_anything/
COPY zim_anything.egg-info/ ./zim_anything.egg-info/
COPY zim_config/ ./zim_config/
COPY setup.py ./setup.py
COPY requirements.txt ./requirements.txt


# Download pretrained model weights
RUN mkdir -p /home/user/worker/results/
RUN mkdir -p /home/user/worker/results/zim_vit_l_2092
WORKDIR /home/user/worker/results/zim_vit_l_2092

RUN wget -O decoder.onnx https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx?download=true
RUN wget -O encoder.onnx https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx?download=true


WORKDIR /home/user/worker

# Disable SSL verification (if really necessary)
RUN conda config --set ssl_verify false

# Create conda environment
RUN conda create -n zim python=3.10 -y

# Activate and install in that environment
# RUN echo "\
# source activate zim && \
# conda install -y -c pytorch -c nvidia \
#     pytorch==2.4.0 \
#     torchvision==0.19.0 \
#     pytorch-cuda=11.8 && \
# pip install -r /home/user/worker/requirements.txt" > install.sh && \
# bash install.sh && \
# rm install.sh

RUN echo "\
source activate zim && \
conda install -y -c pytorch -c nvidia \
    pytorch==2.4.0 \
    torchvision==0.19.0 \
    pytorch-cuda=11.8 && \
pip install -r /home/user/worker/requirements.txt && \
pip install -e /home/user/worker" > install.sh && \
bash install.sh && \
rm install.sh


# Add Blender to PATH
# ENV PATH="/opt/blender:${PATH}"
ENV QT_QPA_PLATFORM=offscreen

# setup gradio port
EXPOSE 11928
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Return to working directory
WORKDIR /home/user/worker

# Final stage: Run the image
FROM conda_builder

# Use root user
USER root

# Default command
ENTRYPOINT ["conda", "run", "-n", "zim", "python", "demo/DEMO_gradio_demo_thres.py"]