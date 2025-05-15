# Use a miniconda base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install dependencies for conda and python
RUN apt-get update && apt-get install -y wget bzip2 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Copy environment.yaml to container
COPY environment.yaml /tmp/environment.yaml

# Create conda environment from file
RUN conda env create -f /tmp/environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "vit-jax", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Expose port
EXPOSE 8000

# Run app using conda environment
CMD ["conda", "run", "-n", "vit-jax", "uvicorn", "API.Main:API", "--host", "0.0.0.0", "--port", "8000"]
