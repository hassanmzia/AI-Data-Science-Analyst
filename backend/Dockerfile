FROM nvidia/cuda:12.8.0-cudnn-runtime-rockylinux9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python 3.12 and system dependencies (PyTorch cu128 for RTX 5090 sm_120 requires Python 3.12+)
# Note: curl-minimal is pre-installed in CUDA image, skip curl to avoid conflicts
RUN dnf install -y epel-release && \
    dnf install -y \
    python3.12 \
    python3.12-pip \
    python3.12-devel \
    gcc \
    gcc-c++ \
    make \
    libpq-devel \
    git \
    && alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && alternatives --install /usr/bin/pip pip /usr/bin/pip3.12 1 \
    && dnf clean all

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.8 support (sm_120 for RTX 5090 Blackwell)
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir nvidia-cusparselt-cu12

# Register all NVIDIA pip package lib directories with the dynamic linker
RUN find /usr/local/lib/python3.12/site-packages/nvidia -name "*.so*" -exec dirname {} \; 2>/dev/null | sort -u > /etc/ld.so.conf.d/nvidia-python.conf && \
    find /usr/local/lib64/python3.12/site-packages/nvidia -name "*.so*" -exec dirname {} \; 2>/dev/null >> /etc/ld.so.conf.d/nvidia-python.conf && \
    sort -u -o /etc/ld.so.conf.d/nvidia-python.conf /etc/ld.so.conf.d/nvidia-python.conf && \
    ldconfig

COPY . .

# Create media directories
RUN mkdir -p /app/media/uploads /app/media/exports /app/media/visualizations \
    /app/media/documents /app/ml_models /app/static

EXPOSE 8050

CMD ["daphne", "-b", "0.0.0.0", "-p", "8050", "core.asgi:application"]
