# Railway Deploy için Dockerfile
# Python 3.11 slim image
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python paketlerini kopyala
COPY requirements.txt .

# Paketleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# Bot kodunu kopyala
COPY telegram_hybrid_bot.py .

# Persistent storage dizini oluştur
RUN mkdir -p /data/storage

# Railway healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Portu aç (Railway otomatik ayarlar)
EXPOSE 8000

# Bot'u başlat
CMD ["python", "-u", "telegram_hybrid_bot.py"]
