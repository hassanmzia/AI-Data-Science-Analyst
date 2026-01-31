FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create media directories
RUN mkdir -p /app/media/uploads /app/media/exports /app/media/visualizations \
    /app/media/documents /app/ml_models /app/static

EXPOSE 8050

CMD ["daphne", "-b", "0.0.0.0", "-p", "8050", "core.asgi:application"]
