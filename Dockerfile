FROM python:3.11-slim

WORKDIR /app

# Requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the main bot file
COPY telegram_bot_v3.py ./telegram_bot.py

EXPOSE 8080

CMD ["python", "telegram_bot.py"]