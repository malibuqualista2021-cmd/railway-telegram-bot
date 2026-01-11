# Railway Cloud Bot - Telegram Asistan

## Özellikler

- PC kapalıyken Railway'de 24/7 çalışır
- Notları Railway volume'da saklar (persistent)
- Groq Llama 3.3 ile AI yanıtlar
- PC açılınca yerel sistemle senkronize olur

## Railway Deploy

### 1. Repository'yu Push Et

```bash
cd railway_bot
git init
git add .
git commit -m "Railway bot"

# GitHub'da yeni repo oluştur, sonra:
git remote add origin https://github.com/KULLANICI/railway-bot.git
git push -u origin main
```

### 2. Railway Projesi Oluştur

1. [railway.app](https://railway.app)'a git
2. **New Project** → **Deploy from GitHub repo**
3. Repository'yi seç
4. Otomatik deploy başlar

### 3. Environment Variables Ayarla

Railway projende:
1. **Variables** sekmesine git
2. Aşağıdaki değişkenleri ekle:

| Key | Value |
|-----|-------|
| `TELEGRAM_TOKEN` | BotFather'dan aldığın token |
| `GROQ_API_KEY` | Groq console'dan aldığın key |

### 4. Persistent Storage (Volume) Ekle

1. **Storage** sekmesine git
2. **New Volume** → `data` adında volume oluştur
3. Volume path: `/data`
4. Bu notların kalıcı olması için gerekli

### 5. Deploy Kontrol

Railway'de **Deployments** sekmesinden logları izle.
Başarılı deploy sonrası bot hemen çalışmaya başlar.

## Yapılandırma

```
railway_bot/
├── telegram_hybrid_bot.py  # Ana bot kodu
├── Dockerfile              # Railway için
├── railway.json            # Railway config
├── requirements.txt        # Python paketleri
└── .env.example            # Örnek env değişkenleri
```

## Senkronizasyon Mimarisi

```
┌─────────────────┐         ┌─────────────────┐
│   TELEGRAM      │         │   YEREL PC      │
│   KULLANICI     │◄───────►│   (v4.2 bot)    │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │        ┌──────────────────┘
         │        │
         ▼        ▼
┌─────────────────────────────────┐
│      RAILWAY CLOUD BOT          │
│  ┌─────────────────────────┐   │
│  │  Groq Llama 3.3         │   │
│  │  + Persistent Storage   │   │
│  └─────────────────────────┘   │
│         ↓         ↑             │
│    PC kapalı    PC açık         │
└─────────────────────────────────┘
```

## Komutlar

| Komut | Açıklama |
|-------|----------|
| `/start` | Botu başlat, menüyü göster |
| Mesaj | Not olarak kaydet |
| Soru | Notlarda ara + AI yanıt |

## Test

Telegram'dan botu bulup `/start` yaz.
Aşağıdaki menü gelirse çalışıyor:

```
🚂 Railway Bot - 24/7 Aktif

[📝 Notlarım] [🔍 Ara]
[📊 Durum] [🔄 Bekleyen]
```

## Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| Bot cevap vermiyor | TELEGRAM_TOKEN kontrol et |
| AI çalışmıyor | GROQ_API_KEY kontrol et |
| Notlar kayboluyor | Volume mount kontrol et (/data) |
| Deploy hatası | Logları kontrol et, requirements.txt'i kontrol et |
