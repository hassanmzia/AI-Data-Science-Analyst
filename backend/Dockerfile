FROM nvidia/cuda:12.4.1-cudnn-runtime-rockylinux9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python 3.11 and system dependencies
# Note: curl-minimal is pre-installed in CUDA image, skip curl to avoid conflicts
RUN dnf install -y epel-release && \
    dnf install -y \
    python3.11 \
    python3.11-pip \
    python3.11-devel \
    gcc \
    gcc-c++ \
    make \
    libpq-devel \
    git \
    && alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && alternatives --install /usr/bin/pip pip /usr/bin/pip3.11 1 \
    && dnf clean all

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.4 support and CUPTI library
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir nvidia-cuda-cupti-cu12

# Symlink libcupti.so.12 to a standard lib path so the dynamic linker finds it
RUN CUPTI_PATH=$(find / -name "libcupti.so.12" 2>/dev/null | head -1) && \
    if [ -n "$CUPTI_PATH" ]; then \
        ln -sf "$CUPTI_PATH" /usr/local/lib64/libcupti.so.12 && \
        echo "Linked $CUPTI_PATH -> /usr/local/lib64/libcupti.so.12"; \
    else \
        echo "WARNING: libcupti.so.12 not found"; \
    fi && \
    ldconfig

COPY . .

# Create media directories
RUN mkdir -p /app/media/uploads /app/media/exports /app/media/visualizations \
    /app/media/documents /app/ml_models /app/static

EXPOSE 8050

CMD ["daphne", "-b", "0.0.0.0", "-p", "8050", "core.asgi:application"]
