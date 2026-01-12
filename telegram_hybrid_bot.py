#!/usr/bin/env python3
"""
Telegram Asistan - Railway Cloud Bot + Hatırlatıcı Sistemi
PC kapalıyken Railway'de çalışır, notları depolar
Hatırlatıcı sistemi ile istediğiniz zaman sizi uyarır

Environment Variables:
- TELEGRAM_TOKEN: Telegram bot token
- GROQ_API_KEY: Groq API key
- SYNC_TOKEN: Senkronizasyon token (optional)
- RAILWAY_VOLUME_URL: Persistent storage path
"""
import os
import sys
import json
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dateutil import parser, rrule
from dateutil.relativedelta import relativedelta

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Voice
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.ext import ContextTypes
from groq import Groq

# Flask API için
from flask import Flask, request, jsonify
from flask_cors import CORS

# Logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==================== CONFIG: ZERO-DEPENDENCY PATTERN ====================
# Config class'ı kaldırıldı - her çağrıda fresh os.getenv() kullanılır
# Bu, Railway container'ında env var load timing sorununu çözer

def get_env(key: str, default: str = "") -> str:
    """Environment variable oku - her çağrıda fresh değer"""
    return os.getenv(key, default)

# ==================== DEBUG: ENV VARIABLES ====================
logger.info("=== ENVIRONMENT VARIABLES DEBUG ===")
for key in sorted(os.environ.keys()):
    if 'TOKEN' in key or 'KEY' in key or 'API' in key or 'GROQ' in key or 'DEEPGRAM' in key:
        value = os.environ[key]
        masked = value[:8] + "..." if len(value) > 8 else "***"
        logger.info(f"{key} = {masked}")
logger.info("====================================\n")


# ==================== STORAGE ====================
class RailwayStorage:
    """Railway persistent storage"""

    def __init__(self, storage_path: str = "/data/storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.notes_file = self.storage_path / "notes.json"
        self.reminders_file = self.storage_path / "reminders.json"
        self.routines_file = self.storage_path / "routines.json"

        self.notes = self._load_json(self.notes_file, [])
        self.reminders = self._load_json(self.reminders_file, [])
        self.routines = self._load_json(self.routines_file, [])
        self.lock = threading.Lock()

    def _load_json(self, path, default):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except:
                pass
        return default

    def _save_json(self, path, data):
        try:
            with self.lock:
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8'
                )
        except Exception as e:
            logger.error(f"Save error {path}: {e}")

    def add_note(self, user_id: int, text: str, source: str = "railway") -> str:
        with self.lock:
            note = {
                "id": f"{source}_{user_id}_{datetime.now().timestamp()}",
                "user_id": user_id,
                "text": text,
                "created": datetime.now().isoformat(),
                "source": source
            }
            self.notes.append(note)
            self._save_json(self.notes_file, self.notes)
            return note["id"]

    def get_notes(self, user_id: int, limit: int = 50) -> List[Dict]:
        user_notes = [n for n in self.notes if n["user_id"] == user_id]
        return user_notes[-limit:]

    def search_notes(self, user_id: int, query: str) -> List[Dict]:
        query_lower = query.lower()
        results = []
        for note in self.notes:
            if note["user_id"] == user_id and query_lower in note["text"].lower():
                results.append(note)
        return results[-10:]

    # ===== REMINDERS =====
    def add_reminder(self, user_id: int, text: str, remind_time: str, note_id: str = None) -> str:
        """Tek seferlik hatırlatıcı ekle"""
        with self.lock:
            reminder = {
                "id": f"rem_{user_id}_{datetime.now().timestamp()}",
                "user_id": user_id,
                "text": text,
                "remind_time": remind_time,  # ISO format
                "note_id": note_id,
                "sent": False,
                "created": datetime.now().isoformat()
            }
            self.reminders.append(reminder)
            self._save_json(self.reminders_file, self.reminders)
            return reminder["id"]

    def get_pending_reminders(self) -> List[Dict]:
        """Bekleyen hatırlatıcıları getir"""
        now = datetime.now().isoformat()
        pending = []
        for r in self.reminders:
            if not r.get("sent", False) and r["remind_time"] <= now:
                pending.append(r)
        return pending

    def mark_reminder_sent(self, reminder_id: str):
        """Hatırlatıcıyı gönderildi olarak işaretle"""
        with self.lock:
            for r in self.reminders:
                if r["id"] == reminder_id:
                    r["sent"] = True
            self._save_json(self.reminders_file, self.reminders)

    def get_user_reminders(self, user_id: int) -> List[Dict]:
        """Kullanıcının hatırlatıcılarını getir"""
        return [r for r in self.reminders if r["user_id"] == user_id and not r.get("sent", False)]

    # ===== ROUTINES =====
    def add_routine(self, user_id: int, text: str, frequency: str, time: str) -> str:
        """
        Rutin hatırlatıcı ekle
        frequency: 'daily', 'weekly', 'monthly' veya 'Pazartesi', 'Salı', vb.
        time: 'HH:MM' format
        """
        with self.lock:
            routine = {
                "id": f"rut_{user_id}_{datetime.now().timestamp()}",
                "user_id": user_id,
                "text": text,
                "frequency": frequency,
                "time": time,
                "last_sent": None,
                "created": datetime.now().isoformat()
            }
            self.routines.append(routine)
            self._save_json(self.routines_file, self.routines)
            return routine["id"]

    def get_routines(self) -> List[Dict]:
        return self.routines

    def get_user_routines(self, user_id: int) -> List[Dict]:
        return [r for r in self.routines if r["user_id"] == user_id]

    def update_routine_last_sent(self, routine_id: str):
        with self.lock:
            for r in self.routines:
                if r["id"] == routine_id:
                    r["last_sent"] = datetime.now().isoformat()
            self._save_json(self.routines_file, self.routines)

    def delete_routine(self, routine_id: str) -> bool:
        with self.lock:
            for i, r in enumerate(self.routines):
                if r["id"] == routine_id:
                    self.routines.pop(i)
                    self._save_json(self.routines_file, self.routines)
                    return True
        return False

    def get_stats(self) -> Dict:
        return {
            "total_notes": len(self.notes),
            "pending_reminders": len([r for r in self.reminders if not r.get("sent")]),
            "active_routines": len(self.routines)
        }


# Global storage
storage = None


# ==================== GROQ AGENT ====================
class GroqAgent:
    SYSTEM = """Sen yardımcı bir Türkçe asistanısın.
Kısa, öz ve dostça yanıtlar ver."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.chat_model = "llama-3.3-70b-versatile"
        self.whisper_model = "whisper-large-v3"  # Groq'un desteklediği model

    def chat(self, text: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": text}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq error: {e}")
            return None

    def transcribe(self, audio_file: bytes) -> Optional[str]:
        """
        Ses dosyasını metne çevir (Deepgram API)
        
        Aksiyomatik Analiz:
        - Telegram voice: OPUS codec, OGA/OGG container
        - Deepgram: audio/ogg destekler, detect=true ile auto-detect
        - Fallback: detect_language=false ile sadece Türkçe
        """
        import tempfile

        # ===== ADIM 1: API KEY KONTROLÜ =====
        deepgram_key = get_env("DEEPGRAM_API_KEY")
        logger.info(f"[TRANSCRIBE-1] API Key check: {'EXISTS (' + deepgram_key[:10] + '...)' if deepgram_key else 'MISSING'}")
        
        if not deepgram_key:
            logger.error("[TRANSCRIBE-1] CRITICAL: DEEPGRAM_API_KEY is not set!")
            return None

        # ===== ADIM 2: AUDIO DATA VALİDASYONU =====
        if not audio_file or len(audio_file) < 100:
            logger.error(f"[TRANSCRIBE-2] Audio data invalid: {len(audio_file) if audio_file else 0} bytes")
            return None
        
        logger.info(f"[TRANSCRIBE-2] Audio size: {len(audio_file)} bytes ({len(audio_file)/1024:.1f} KB)")
        
        # OGG magic bytes kontrolü (OggS)
        if audio_file[:4] == b'OggS':
            logger.info("[TRANSCRIBE-2] Audio format: Valid OGG container detected")
        else:
            logger.warning(f"[TRANSCRIBE-2] Audio format: Unknown (magic: {audio_file[:4]})")

        try:
            # ===== ADIM 3: GEÇİCİ DOSYA =====
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                tmp.write(audio_file)
                tmp_path = tmp.name
            logger.info(f"[TRANSCRIBE-3] Temp file: {tmp_path}")

            # ===== ADIM 4: DEEPGRAM API ÇAĞRISI =====
            # Parametreler:
            # - model=nova-2: En iyi genel model
            # - language=tr: Türkçe
            # - smart_format=true: Noktalama işaretleri
            # - punctuate=true: Ek noktalama
            url = "https://api.deepgram.com/v1/listen"
            params = {
                "model": "nova-2",
                "language": "tr",
                "smart_format": "true",
                "punctuate": "true"
            }
            
            headers = {
                "Authorization": f"Token {deepgram_key}",
                "Content-Type": "audio/ogg"
            }

            logger.info(f"[TRANSCRIBE-4] Sending to Deepgram...")
            logger.info(f"[TRANSCRIBE-4] URL: {url}")
            logger.info(f"[TRANSCRIBE-4] Params: {params}")

            with open(tmp_path, "rb") as audio:
                audio_bytes = audio.read()
                logger.info(f"[TRANSCRIBE-4] Sending {len(audio_bytes)} bytes...")
                
                response = requests.post(
                    url,
                    params=params,
                    headers=headers,
                    data=audio_bytes,
                    timeout=60  # Timeout artırıldı
                )

            logger.info(f"[TRANSCRIBE-4] Response status: {response.status_code}")
            logger.info(f"[TRANSCRIBE-4] Response headers: {dict(response.headers)}")

            # ===== ADIM 5: GEÇİCİ DOSYA TEMİZLİĞİ =====
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"[TRANSCRIBE-5] Could not delete temp file: {e}")

            # ===== ADIM 6: RESPONSE İŞLEME =====
            if response.status_code == 200:
                result = response.json()
                
                # Ham response log (debug için)
                response_str = json.dumps(result, ensure_ascii=False)
                logger.info(f"[TRANSCRIBE-6] Raw response (first 800 chars): {response_str[:800]}")
                
                # Deepgram response yapısı:
                # {
                #   "results": {
                #     "channels": [{
                #       "alternatives": [{
                #         "transcript": "metin",
                #         "confidence": 0.95
                #       }]
                #     }]
                #   }
                # }
                
                try:
                    channels = result.get("results", {}).get("channels", [])
                    if not channels:
                        logger.error("[TRANSCRIBE-6] No channels in response")
                        return None
                    
                    alternatives = channels[0].get("alternatives", [])
                    if not alternatives:
                        logger.error("[TRANSCRIBE-6] No alternatives in response")
                        return None
                    
                    transcript = alternatives[0].get("transcript", "").strip()
                    confidence = alternatives[0].get("confidence", 0)
                    
                    logger.info(f"[TRANSCRIBE-6] Transcript: '{transcript}'")
                    logger.info(f"[TRANSCRIBE-6] Confidence: {confidence}")
                    
                    if not transcript:
                        logger.warning("[TRANSCRIBE-6] Empty transcript - audio may be silence or unclear")
                        return None
                    
                    return transcript
                    
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"[TRANSCRIBE-6] Parse error: {type(e).__name__}: {e}")
                    logger.error(f"[TRANSCRIBE-6] Full response: {result}")
                    return None
                    
            elif response.status_code == 401:
                logger.error("[TRANSCRIBE-6] ERROR 401: Invalid API key!")
                return None
            elif response.status_code == 402:
                logger.error("[TRANSCRIBE-6] ERROR 402: Payment required - free quota exceeded!")
                return None
            elif response.status_code == 400:
                logger.error(f"[TRANSCRIBE-6] ERROR 400: Bad request - {response.text[:300]}")
                return None
            else:
                logger.error(f"[TRANSCRIBE-6] ERROR {response.status_code}: {response.text[:300]}")
                return None

        except requests.exceptions.Timeout:
            logger.error("[TRANSCRIBE] TIMEOUT: Deepgram did not respond in 60 seconds")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[TRANSCRIBE] CONNECTION ERROR: {e}")
            return None
        except Exception as e:
            logger.error(f"[TRANSCRIBE] UNEXPECTED ERROR: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[TRANSCRIBE] Traceback: {traceback.format_exc()}")
            return None

    def classify_intent(self, text: str) -> str:
        """Metnin niyetini sınıflandır"""
        system_prompt = """Sen bir asistan köprüsüsün. Kullanıcı mesajının niyetini sınıflandır ve SADECE şu kelimelerden birini döndür:

- reminder: Kullanıcı gelecekte bir şey hatırlatmak istiyor (zaman ifade eder)
- routine: Kullanıcı tekrarlayan bir rutin belirtiyor (her gün, her hafta vb.)
- note: Sadece bilgi/not kaydediyor
- chat: Sadece sohbet ediyor, soru soruyor

Örnekler:
"Yarın toplantı var" → reminder
"Her sabah 9'da kahve" → routine
"Toplantıda X kararı alındı" → note
"Merhaba, nasılsın?" → chat
"Toplantı ne zaman?" → chat"""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=10,
                temperature=0
            )
            intent = response.choices[0].message.content.strip().lower()
            logger.info(f"Intent classified: {intent} for: {text[:50]}")
            return intent
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return "note"  # Varsayılan


# ==================== REMINDER HELPERS ====================
def parse_reminder_time(time_str: str) -> Optional[str]:
    """
    Zaman stringini ISO formatına çevir
    Örnekler:
    - "15:30" → Bugün 15:30
    - "yarın 10:00" → Yarın 10:00
    - "yarın 20:00" → Yarın 20:00
    - "Pazartesi 14:00" → Gelecek Pazartesi 14:00
    - "2026-01-15 09:00" → O tarih
    """
    import re
    
    logger.info(f"[PARSE_TIME] Input: '{time_str}'")
    
    try:
        time_str = time_str.strip()
        now = datetime.now()
        
        # Saat pattern'i bul (HH:MM formatı)
        time_pattern = re.search(r'(\d{1,2}):(\d{2})', time_str)
        
        if time_pattern:
            hour = int(time_pattern.group(1))
            minute = int(time_pattern.group(2))
            logger.info(f"[PARSE_TIME] Found time: {hour:02d}:{minute:02d}")
        else:
            # Saat bulunamadı, varsayılan kullan
            hour, minute = 9, 0
            logger.info(f"[PARSE_TIME] No time found, using default: {hour:02d}:{minute:02d}")

        time_str_lower = time_str.lower()

        # "yarın" kontrolü
        if "yarın" in time_str_lower:
            target = now + timedelta(days=1)
            target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            logger.info(f"[PARSE_TIME] 'yarın' detected, target: {target.isoformat()}")
            return target.isoformat()

        # "bugün" kontrolü
        if "bugün" in time_str_lower:
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            logger.info(f"[PARSE_TIME] 'bugün' detected, target: {target.isoformat()}")
            return target.isoformat()

        # Gün isimleri (Pazartesi, Salı, ...)
        days_tr = {
            "pazartesi": 0, "salı": 1, "çarşamba": 2, "perşembe": 3,
            "cuma": 4, "cumartesi": 5, "pazar": 6
        }
        for day_tr, day_idx in days_tr.items():
            if day_tr in time_str_lower:
                current_day = now.weekday()
                days_ahead = (day_idx - current_day) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target = now + timedelta(days=days_ahead)
                target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
                logger.info(f"[PARSE_TIME] '{day_tr}' detected, target: {target.isoformat()}")
                return target.isoformat()

        # Sadece saat varsa (HH:MM) → bugün veya yarın
        if time_pattern and len(time_str) <= 10:
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target < now:
                target += timedelta(days=1)
            logger.info(f"[PARSE_TIME] Time only, target: {target.isoformat()}")
            return target.isoformat()

        # dateutil ile parse etmeyi dene
        logger.info(f"[PARSE_TIME] Trying dateutil parser...")
        target = parser.parse(time_str, fuzzy=True, dayfirst=True)
        
        # Geçmiş tarihse yarına al
        if target < now:
            if target.date() == now.date():
                target += timedelta(days=1)
        
        logger.info(f"[PARSE_TIME] dateutil result: {target.isoformat()}")
        return target.isoformat()

    except Exception as e:
        logger.error(f"[PARSE_TIME] Error parsing '{time_str}': {type(e).__name__}: {e}")
        return None


def parse_routine_frequency(freq_str: str) -> tuple:
    """
    Rutin frekansını çözümle
    Returns: (frequency_type, time)
    frequency_type: 'daily', 'weekly', 'monthly', 'weekday'
    """
    freq_str = freq_str.strip().lower()
    time_str = "09:00"  # varsayılan

    # Saat çıkart
    if ":" in freq_str:
        parts = freq_str.split(":")
        time_str = f"{parts[-2]}:{parts[-1][:2]}"
        freq_str = freq_str.replace(time_str, "").strip()

    # Günlük
    if any(w in freq_str for w in ["günlük", "her gün", "daily"]):
        return "daily", time_str

    # Haftalık
    if any(w in freq_str for w in ["haftalık", "her hafta", "weekly"]):
        return "weekly", time_str

    # Aylık
    if any(w in freq_str for w in ["aylık", "her ay", "monthly"]):
        return "monthly", time_str

    # Gün isimleri
    days_tr = ["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"]
    for day in days_tr:
        if day in freq_str:
            return day.capitalize(), time_str

    return freq_str, time_str


# ==================== SYNC API (Flask) ====================
sync_app = Flask(__name__)
CORS(sync_app)


@sync_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "railway-bot", "timestamp": datetime.now().isoformat()})


def run_flask():
    port = int(get_env("PORT", "8080"))
    logger.info(f"Sync API starting on port {port}")
    sync_app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)


# ==================== TELEGRAM BOT ====================
class RailwayBot:
    def __init__(self):
        self.groq = GroqAgent(get_env("GROQ_API_KEY"))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"=== START COMMAND RECEIVED from {update.effective_user.id} ===")
        user_id = update.effective_user.id
        stats = storage.get_stats()

        keyboard = [
            [InlineKeyboardButton("📝 Not Al", callback_data=f"note_{user_id}"),
             InlineKeyboardButton("🔍 Ara", callback_data=f"search_{user_id}")],
            [InlineKeyboardButton("⏰ Hatırlatıcı", callback_data=f"reminder_{user_id}"),
             InlineKeyboardButton("🔄 Rutin", callback_data=f"routine_{user_id}")],
            [InlineKeyboardButton("📊 Durum", callback_data=f"status_{user_id}")]
        ]

        reply = f"""🚂 **Asistan Bot - 24/7 Aktif**

Merhaba {update.effective_user.first_name}!

**Özellikler:**
• 📝 Not alma
• ⏰ Hatırlatıcı (tarih/saat)
• 🔄 Rutin hatırlatmalar
• 🔍 Notlarda arama

**Durum:**
📝 Not: {stats['total_notes']}
⏰ Bekleyen hatırlatıcı: {stats['pending_reminders']}
🔄 Aktif rutin: {stats['active_routines']}

**Komutlar:**
/remind → Hatırlatıcı ekle
/routine → Rutin ekle
/list → Listele"""

        await update.message.reply_text(
            reply,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )

    async def remind_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/remind komutu"""
        user_id = update.effective_user.id

        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                """⏰ **Hatırlatıcı Ekle**

Kullanım:
/remind <zaman> <mesaj>

Örnekler:
/remind 15:30 Toplantı
/remind yarın 09:00 Fatura ödeme
/remind Pazartesi 10:00 Haftalık toplantı

Zaman formatları:
• 15:30 → Bugün saat 15:30
• Yarın 09:00 → Yarın saat 09:00
• Pazartesi 14:00 → Gelecek pazartesi""",
                parse_mode='Markdown'
            )
            return

        # Zaman ve mesajı ayrıştır
        time_str = context.args[0]
        message = " ".join(context.args[1:])

        # Zamanı çözümle
        remind_time = parse_reminder_time(time_str)
        if not remind_time:
            await update.message.reply_text(f"❌ Zaman formatı anlaşılamadı: {time_str}")
            return

        # Hatırlatıcı ekle
        reminder_id = storage.add_reminder(user_id, message, remind_time)

        # Okunabilir tarih
        dt = parser.parse(remind_time)
        readable_time = dt.strftime("%d.%m.%Y %H:%M")

        await update.message.reply_text(
            f"✅ Hatırlatıcı ayarlandı!\n\n"
            f"⏰ {readable_time}\n"
            f"📝 {message}"
        )

    async def routine_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/routine komutu"""
        user_id = update.effective_user.id

        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                """🔄 **Rutin Hatırlatıcı Ekle**

Kullanım:
/routine <sıklık> <saat> <mesaj>

Örnekler:
/routine günlük 09:00 Güne başla
/routine Pazartesi 10:00 Haftalık toplantı
/routine haftalık 14:30 Spor yap

Sıklık seçenekleri:
• günlük / her gün
• Pazartesi, Salı, ... (gün isimleri)
• haftalık
• aylık""",
                parse_mode='Markdown'
            )
            return

        # Frekans ve saati ayrıştır
        freq, time_str = parse_routine_frequency(context.args[0])
        message = " ".join(context.args[1:])

        # Saat varsa ayıkla
        if ":" in context.args[1]:
            time_str = context.args[1]
            message = " ".join(context.args[2:])

        # Rutin ekle
        routine_id = storage.add_routine(user_id, message, freq, time_str)

        await update.message.reply_text(
            f"✅ Rutin hatırlatıcı ayarlandı!\n\n"
            f"🔄 {freq.capitalize()} • {time_str}\n"
            f"📝 {message}"
        )

    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/list komutu - hatırlatıcı ve rutin listesi"""
        user_id = update.effective_user.id

        reminders = storage.get_user_reminders(user_id)
        routines = storage.get_user_routines(user_id)

        reply = "📋 **Hatırlatıcılarınız**\n\n"

        if reminders:
            reply += "⏰ *Bekleyen Hatırlatıcılar:*\n"
            for r in reminders[-5:]:
                dt = parser.parse(r["remind_time"])
                readable = dt.strftime("%d.%m.%Y %H:%M")
                reply += f"• {readable}: {r['text'][:40]}...\n"
        else:
            reply += "⏰ Bekleyen hatırlatıcı yok\n"

        reply += "\n"

        if routines:
            reply += "🔄 *Rutinler:*\n"
            for r in routines[-5:]:
                reply += f"• {r['frequency']} - {r['time']}: {r['text'][:40]}...\n"
        else:
            reply += "🔄 Rutin yok\n"

        await update.message.reply_text(reply, parse_mode='Markdown')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        text = update.message.text

        await update.message.chat.send_action("typing")

        # Soru mu, not mu?
        is_question = any(w in text.lower() for w in ["?", "nedir", "nasıl", "kim", "nerede"])

        if is_question:
            await self._handle_question(update, user_id, text)
        else:
            storage.add_note(user_id, text, source="railway")
            ai_response = self.groq.chat(f"Kullanıcı not aldı: '{text}'. Kısa teyit.")
            response = ai_response or "✅ Not kaydedildi"
            await update.message.reply_text(response)

    async def _handle_question(self, update: Update, user_id: int, query: str):
        results = storage.search_notes(user_id, query)

        if results:
            reply = f"🔍 **Bulunanlar ({len(results)}):**\n\n"
            for note in results[-5:]:
                reply += f"• {note['text'][:80]}...\n"
            await update.message.reply_text(reply, parse_mode='Markdown')
        else:
            ai_response = self.groq.chat(query)
            if ai_response:
                await update.message.reply_text(f"🤖 **AI:**\n\n{ai_response}", parse_mode='Markdown')

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        data = query.data
        parts = data.split('_')
        action = parts[0]
        user_id = int(parts[1]) if len(parts) > 1 else 0

        if action == "note":
            await query.edit_message_text("📝 Notunuzu yazın...")
        elif action == "search":
            await query.edit_message_text("🔍 Aramak istediğinizi yazın...")
        elif action == "reminder":
            await query.edit_message_text(
                "⏰ Hatırlatıcı eklemek için:\n\n/remind <zaman> <mesaj>\n\n"
                "Örnek: /remind 15:30 Toplantı"
            )
        elif action == "routine":
            await query.edit_message_text(
                "🔄 Rutin eklemek için:\n\n/routine <sıklık> <saat> <mesaj>\n\n"
                "Örnek: /routine günlük 09:00 Kahve"
            )
        elif action == "status":
            stats = storage.get_stats()
            reply = f"📊 **Durum**\n\n📝 Not: {stats['total_notes']}\n⏰ Hatırlatıcı: {stats['pending_reminders']}\n🔄 Rutin: {stats['active_routines']}"
            await query.edit_message_text(reply, parse_mode='Markdown')

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Sesli mesaj işle - Deepgram transkripsiyon + AI sınıflandırma"""
        user_id = update.effective_user.id
        voice = update.message.voice
        duration = voice.duration

        # ===== DEBUG LOG =====
        logger.info("=" * 60)
        logger.info("=== VOICE MESSAGE RECEIVED ===")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Duration: {duration}s")
        logger.info(f"File ID: {voice.file_id}")
        logger.info(f"File size: {voice.file_size} bytes")
        logger.info(f"MIME type: {voice.mime_type}")
        
        # Environment check
        deepgram_key = get_env('DEEPGRAM_API_KEY')
        logger.info(f"DEEPGRAM_API_KEY exists: {bool(deepgram_key)}")
        if deepgram_key:
            logger.info(f"DEEPGRAM_API_KEY preview: {deepgram_key[:10]}...{deepgram_key[-4:]}")
        else:
            logger.error("DEEPGRAM_API_KEY is MISSING!")
        logger.info("=" * 60)

        # 10 dakikadan uzunsa reddet
        if duration > 600:
            await update.message.reply_text("⚠️ Ses kaydı çok uzun (max 10 dakika)")
            return

        # Çok kısa sesler için uyarı
        if duration < 1:
            await update.message.reply_text("⚠️ Ses kaydı çok kısa, en az 1 saniye olmalı")
            return

        await update.message.chat.send_action("record_voice")
        status_msg = await update.message.reply_text("🎤 Ses işleniyor...")

        try:
            # Ses dosyasını indir
            logger.info("[VOICE] Downloading audio file from Telegram...")
            new_file = await voice.get_file()
            audio_data = await new_file.download_as_bytearray()

            logger.info(f"[VOICE] Downloaded: {len(audio_data)} bytes ({len(audio_data)/1024:.1f} KB)")

            # Audio data kontrolü
            if len(audio_data) < 100:
                logger.error(f"[VOICE] Audio data too small: {len(audio_data)} bytes")
                await status_msg.edit_text("❌ Ses dosyası indirilemedi")
                return

            # Deepgram ile transkripsiyon
            logger.info("[VOICE] Starting transcription...")
            transcript = self.groq.transcribe(bytes(audio_data))

            logger.info(f"[VOICE] Transcription result: '{transcript}'" if transcript else "[VOICE] Transcription returned None")

            if not transcript:
                # Detaylı hata mesajı
                error_msg = "❌ Ses anlaşılamadı\n\n"
                error_msg += "Olası nedenler:\n"
                error_msg += "• Ses çok kısa veya sessiz\n"
                error_msg += "• Arka plan gürültüsü fazla\n"
                error_msg += "• Mikrofon sorunu\n\n"
                error_msg += "💡 Daha uzun ve net konuşmayı deneyin"
                
                await status_msg.edit_text(error_msg)
                return

            logger.info(f"[VOICE] SUCCESS! Transcript for user {user_id}: {transcript[:100]}")

            # Başarılı transkripsiyon mesajı sil
            await status_msg.delete()

            # AI ile niyet sınıflandırması
            intent = self.groq.classify_intent(transcript)
            logger.info(f"[VOICE] Intent classified as: {intent}")

            # Niyete göre işlem
            if intent == "reminder":
                await self._process_reminder_from_voice(update, transcript)
            elif intent == "routine":
                await self._process_routine_from_voice(update, transcript)
            elif intent == "note":
                storage.add_note(user_id, f"[Ses] {transcript}", source="voice")
                await update.message.reply_text(f"📝 Not alındı:\n\n{transcript}")
            else:  # chat
                ai_response = self.groq.chat(transcript)
                if ai_response:
                    await update.message.reply_text(f"🤖 **AI:**\n\n{ai_response}", parse_mode='Markdown')

        except Exception as e:
            logger.error(f"[VOICE] EXCEPTION: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[VOICE] Traceback:\n{traceback.format_exc()}")
            await status_msg.edit_text(f"❌ İşlem hatası: {type(e).__name__}")

    async def _process_reminder_from_voice(self, update: Update, transcript: str):
        """Sesten hatırlatıcı çıkar"""
        user_id = update.effective_user.id
        logger.info(f"[REMINDER] Processing reminder from voice for user {user_id}")
        logger.info(f"[REMINDER] Transcript: {transcript}")

        # AI ile zaman ve mesajı çıkar
        prompt = f"""Bu metinden hatırlatıcı zamanı ve mesajını çıkar. JSON formatında döndür:
{{"time": "YYYY-MM-DD HH:MM veya yarın HH:MM veya HH:MM", "message": "mesaj"}}

Örnekler:
- "yarın akşam 8'de spor" → {{"time": "yarın 20:00", "message": "spor yapmam lazım"}}
- "saat 3'te toplantı" → {{"time": "15:00", "message": "toplantı"}}
- "pazartesi 9'da doktor" → {{"time": "Pazartesi 09:00", "message": "doktor randevusu"}}

Metin: {transcript}

Sadece JSON döndür, başka bir şey yazma."""

        try:
            logger.info("[REMINDER] Calling Groq API to extract time...")
            response = self.groq.client.chat.completions.create(
                model=self.groq.chat_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200
            )

            raw_content = response.choices[0].message.content
            logger.info(f"[REMINDER] Groq raw response: {raw_content}")

            import json
            result = json.loads(raw_content)
            time_str = result.get("time", "")
            message = result.get("message", transcript)

            logger.info(f"[REMINDER] Extracted time: '{time_str}'")
            logger.info(f"[REMINDER] Extracted message: '{message}'")

            if time_str:
                logger.info(f"[REMINDER] Parsing time: {time_str}")
                remind_time = parse_reminder_time(time_str)
                logger.info(f"[REMINDER] Parsed remind_time: {remind_time}")
                
                if remind_time:
                    storage.add_reminder(user_id, message, remind_time)
                    dt = parser.parse(remind_time)
                    readable = dt.strftime("%d.%m.%Y %H:%M")
                    logger.info(f"[REMINDER] SUCCESS! Reminder set for {readable}")
                    await update.message.reply_text(
                        f"⏰ Hatırlatıcı ayarlandı!\n\n📅 {readable}\n📝 {message}"
                    )
                    return
                else:
                    logger.warning(f"[REMINDER] parse_reminder_time returned None for: {time_str}")

            # Zaman çıkarılamazsa tümünü not olarak kaydet
            logger.info("[REMINDER] Could not parse time, saving as note")
            storage.add_note(user_id, f"[Ses] {transcript}", source="voice")
            await update.message.reply_text(f"📝 Not alındı (zaman anlaşılamadı):\n\n{transcript}")

        except json.JSONDecodeError as e:
            logger.error(f"[REMINDER] JSON parse error: {e}")
            logger.error(f"[REMINDER] Raw content was: {raw_content}")
            storage.add_note(user_id, f"[Ses] {transcript}", source="voice")
            await update.message.reply_text(f"📝 Not alındı:\n\n{transcript}")
        except Exception as e:
            logger.error(f"[REMINDER] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[REMINDER] Traceback: {traceback.format_exc()}")
            storage.add_note(user_id, f"[Ses] {transcript}", source="voice")
            await update.message.reply_text(f"📝 Not alındı:\n\n{transcript}")

    async def _process_routine_from_voice(self, update: Update, transcript: str):
        """Sesten rutin çıkar"""
        user_id = update.effective_user.id

        # AI ile rutini çıkar
        prompt = f"""Bu metinden rutin sıklığını, saatini ve mesajını çıkar. JSON formatında döndür:
{{"frequency": "günlük/haftalık/aylık/gün adı", "time": "HH:MM", "message": "mesaj"}}

Metin: {transcript}

Sadece JSON döndür."""

        try:
            response = self.groq.client.chat.completions.create(
                model=self.groq.chat_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200
            )

            import json
            result = json.loads(response.choices[0].message.content)
            freq = result.get("frequency", "günlük")
            time_str = result.get("time", "09:00")
            message = result.get("message", transcript)

            storage.add_routine(user_id, message, freq, time_str)
            await update.message.reply_text(
                f"🔄 Rutin ayarlandı!\n\n{freq.capitalize()} • {time_str}\n📝 {message}"
            )

        except Exception as e:
            logger.error(f"Routine extraction error: {e}")
            storage.add_note(user_id, f"[Ses] {transcript}", source="voice")
            await update.message.reply_text(f"📝 Not alındı:\n\n{transcript}")


# ==================== REMINDER CHECKER ====================
async def check_reminders_job(app: Application):
    """Periyodik hatırlatıcı kontrolü"""
    logger.info("Checking reminders...")

    pending = storage.get_pending_reminders()

    for reminder in pending:
        try:
            user_id = reminder["user_id"]
            text = reminder["text"]
            remind_time = parser.parse(reminder["remind_time"])
            readable_time = remind_time.strftime("%d.%m.%Y %H:%M")

            await app.bot.send_message(
                chat_id=user_id,
                text=f"⏰ **HATIRLATICI**\n\n{readable_time}\n📝 {text}",
                parse_mode='Markdown'
            )

            storage.mark_reminder_sent(reminder["id"])
            logger.info(f"Reminder sent to {user_id}: {text[:30]}")

        except Exception as e:
            logger.error(f"Error sending reminder: {e}")


async def check_routines_job(app: Application):
    """Rutin hatırlatıcı kontrolü"""
    logger.info("Checking routines...")

    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_day = now.strftime("%A")  # Monday, Tuesday, etc.
    current_day_tr = now.weekday()  # 0=Monday, 6=Sunday

    days_tr_map = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba",
                   3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}

    routines = storage.get_routines()

    for routine in routines:
        try:
            should_send = False
            freq = routine["frequency"].lower()
            routine_time = routine["time"]

            # Saat kontrolü
            if routine_time != current_time:
                continue

            # Frekans kontrolü
            if freq == "daily" or freq == "günlük":
                should_send = True
            elif freq == "weekly" or freq == "haftalık":
                # Haftalık - her pazartesi veya haftanın ilk günü
                if current_day_tr == 0:  # Pazartesi
                    should_send = True
            elif freq == "monthly" or freq == "aylık":
                # Aylık - ayın 1'i
                if now.day == 1:
                    should_send = True
            elif freq in days_tr_map.values():
                # Gün ismi
                if days_tr_map[current_day_tr] == freq.capitalize():
                    should_send = True

            # Last sent kontrolü (aynı gün içinde tekrar gönderme)
            if routine.get("last_sent"):
                last_sent = parser.parse(routine["last_sent"])
                if (now - last_sent).days < 1:
                    continue

            if should_send:
                user_id = routine["user_id"]
                text = routine["text"]

                await app.bot.send_message(
                    chat_id=user_id,
                    text=f"🔄 **RUTİN HATIRLATICI**\n\n{routine['frequency']} • {routine_time}\n📝 {text}",
                    parse_mode='Markdown'
                )

                storage.update_routine_last_sent(routine["id"])
                logger.info(f"Routine sent to {user_id}: {text[:30]}")

        except Exception as e:
            logger.error(f"Error sending routine: {e}")


# ==================== MAIN ====================
def main():
    global storage

    telegram_token = get_env("TELEGRAM_TOKEN")
    groq_key = get_env("GROQ_API_KEY")

    if not telegram_token or not groq_key:
        logger.error("TELEGRAM_TOKEN or GROQ_API_KEY not set!")
        sys.exit(1)

    storage = RailwayStorage(get_env("RAILWAY_VOLUME_URL", "/data/storage"))

    # Flask thread
    flask_thread = threading.Thread(target=run_flask, daemon=False)
    flask_thread.start()
    logger.info("Sync API thread started")

    # Telegram bot
    bot = RailwayBot()
    app = Application.builder().token(telegram_token).build()

    # Handlers
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("remind", bot.remind_command))
    app.add_handler(CommandHandler("routine", bot.routine_command))
    app.add_handler(CommandHandler("list", bot.list_command))
    app.add_handler(MessageHandler(filters.VOICE, bot.handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    app.add_handler(CallbackQueryHandler(bot.button_callback))

    # Error handler - tüm hataları log'la
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Exception while handling an update: {context.error}")
        if update:
            logger.error(f"Update: {update}")

    app.add_error_handler(error_handler)

    # Job queue - her dakika kontrol
    job_queue = app.job_queue

    # Hatırlatıcı kontrolü - her dakika
    job_queue.run_repeating(check_reminders_job, interval=60, first=10)

    # Rutin kontrolü - her dakika
    job_queue.run_repeating(check_routines_job, interval=60, first=15)

    logger.info("=" * 50)
    logger.info("Railway Bot + Reminder System Starting...")
    logger.info(f"Storage: {get_env('RAILWAY_VOLUME_URL', '/data/storage')}")
    logger.info(f"Sync API: Port {get_env('PORT', '8080')}")
    logger.info("AI: Groq Llama 3.3")
    logger.info("Reminders: Active")
    logger.info("=" * 50)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
