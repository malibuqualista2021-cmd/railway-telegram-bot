# Railway Deploy için Dockerfile
# Python 3.11 slim image
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python paketlerini kopyala
COPY requirements.txt .

# Paketleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# Bot kodunu kopyala
COPY telegram_hybrid_bot.py .

# Persistent storage dizini oluştur
RUN mkdir -p /data/storage

# Telegram bot'u başlat
CMD ["python", "-u", "telegram_hybrid_bot.py"]
