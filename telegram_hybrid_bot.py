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
import tempfile
import shutil
from google_calendar_mgr import GoogleCalendarManager
import pytz
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dateutil import parser, rrule
from dateutil.relativedelta import relativedelta
from functools import wraps
import time

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Voice, ReplyKeyboardMarkup, KeyboardButton
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

# Timezone: Default is Turkey (UTC+3)
USER_TZ = pytz.timezone("Europe/Istanbul")

# ==================== RESILIENCE: ALGORITHMIC INSURANCE ====================
def retry_on_failure(retries=3, delay=1):
    """API ve harici servisler için otomatik tekrar deneme dekoratörü"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"[RETRY] {func.__name__} failed ({i+1}/{retries}): {e}")
                    time.sleep(delay * (i + 1))
            logger.error(f"[FATAL] {func.__name__} failed after {retries} retries.")
            return None
        return wrapper
    return decorator

def async_retry_on_failure(retries=3, delay=1):
    """Async fonksiyonlar için tekrar deneme dekoratörü"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"[RETRY-ASYNC] {func.__name__} failed ({i+1}/{retries}): {e}")
                    await asyncio.sleep(delay * (i + 1))
            logger.error(f"[FATAL-ASYNC] {func.__name__} failed after {retries} retries.")
            return None
        return wrapper
    return decorator

# ==================== RESILIENCE: BLACK SWAN PROTECTIONS ====================
SAFE_MODE = False
FAILURE_CREDITS = 10  # Max total failures before Safe Mode
error_counter = 0

def trigger_safe_mode():
    global SAFE_MODE
    if not SAFE_MODE:
        SAFE_MODE = True
        logger.critical("!!! BLACK SWAN ALERT: SYSTEM ENTERING SAFE MODE !!!")
        logger.critical("Minimal Deterministic Protocol Activated. AI Offline.")

class CircuitBreaker:
    """AI servisleri için soğuma süresi mekanizması"""
    def __init__(self, threshold=3, recovery_time=300):
        self.threshold = threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.last_failure_time = 0
        self.open = False

    def report_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.open = True
            logger.warning(f"[CIRCUIT_BREAKER] AI services paused for {self.recovery_time}s")

    def can_proceed(self):
        if not self.open:
            return True
        if time.time() - self.last_failure_time > self.recovery_time:
            self.open = False
            self.failures = 0
            logger.info("[CIRCUIT_BREAKER] AI services resumed.")
            return True
        return False

ai_breaker = CircuitBreaker()

def check_error_threshold():
    global error_counter
    error_counter += 1
    if error_counter >= FAILURE_CREDITS:
        trigger_safe_mode()

def get_now_utc():
    return datetime.now(pytz.UTC)

def get_now_local():
    return datetime.now(USER_TZ)


# ==================== CONFIG: ZERO-DEPENDENCY PATTERN ====================
# Config class'ı kaldırıldı - her çağrıda fresh os.getenv() kullanılır
# Bu, Railway container'ında env var load timing sorununu çözer

def get_env(key: str, default: str = "") -> str:
    """Environment variable oku - her çağrıda fresh değer"""
    return os.getenv(key, default)

# ==================== ARCHITECTURE: MODEL REGISTRY ====================
AI_MODELS = {
    "chat": "llama-3.3-70b-versatile",
    "vision": "llama-3.2-11b-vision-preview",
    "whisper": "whisper-large-v3"
}

# ==================== DEBUG: ENV VARIABLES ====================
logger.info("=== ENVIRONMENT VARIABLES DEBUG ===")
for key in sorted(os.environ.keys()):
    if 'TOKEN' in key or 'KEY' in key or 'API' in key or 'GROQ' in key or 'DEEPGRAM' in key:
        value = os.environ[key]
        masked = value[:8] + "..." if len(value) > 8 else "***"
        logger.info(f"{key} = {masked}")
logger.info("====================================\n")


# ==================== DETERMINISM: BOUNDARY GUARDS ====================
MAX_TEXT_LENGTH = 4000

def sanitize_text(text: str) -> str:
    """Deterministic Input Sanitization"""
    if not text:
        return ""
    # Normalize unicode and strip control characters
    import unicodedata
    text = "".join(ch for ch in unicodedata.normalize('NFKC', text) if unicodedata.category(ch)[0] != 'C' or ch in '\n\r\t')
    return text.strip()[:MAX_TEXT_LENGTH]

def generate_correlation_id() -> str:
    """Support: Generate a short unique ID for error tracking"""
    import uuid
    return str(uuid.uuid4())[:8].upper()

# ==================== STORAGE ====================
class RailwayStorage:
    """Railway persistent storage"""

    def __init__(self, storage_path: str = "/data/storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.notes_file = self.storage_path / "notes.json"
        self.reminders_file = self.storage_path / "reminders.json"
        self.routines_file = self.storage_path / "routines.json"

        self.lock = threading.Lock()
        self.notes = self._load_json(self.notes_file, [])
        self.reminders = self._load_json(self.reminders_file, [])
        self.routines = self._load_json(self.routines_file, [])

    def _load_json(self, path, default):
        with self.lock:
            # 1. Primary dosyayı dene
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                    if not isinstance(data, list):
                        raise ValueError("Data must be a list")
                    return data
                except Exception as e:
                    logger.error(f"Load error primary {path}: {e}")
                    # Move to Dead Letter Queue (DLQ) if corrupted
                    self._isolate_corrupt_data(path)
            
            # 2. Hata varsa veya yoksa yedeği dene (Resilience)
            bak_path = path.with_suffix(".json.bak")
            if bak_path.exists():
                try:
                    logger.info(f"Loading from backup: {bak_path}")
                    data = json.loads(bak_path.read_text(encoding='utf-8'))
                    if isinstance(data, list):
                        return data
                except Exception as e:
                    logger.error(f"Load error backup {bak_path}: {e}")
            return default

    def _isolate_corrupt_data(self, path):
        """İzole et: Bozuk veriyi Dead Letter Queue'ya taşı"""
        try:
            dlq_path = self.storage_path / f"corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{path.name}"
            if path.exists():
                shutil.move(path, dlq_path)
                logger.warning(f"[DLQ] Corrupt data isolated to {dlq_path}")
        except Exception as e:
            logger.error(f"[DLQ] Isolation failed: {e}")

    def _prune_data(self, data_list, filename, max_items=1000):
        """
        Teleological Memory Guard:
        - Hatırlatıcılar (Reminders): Sadece gönderilmiş olanları buda.
        - Notlar (Notes): Silme (External Brain mission). Sadece 5000+ limitinde çok eskileri uyararak buda.
        """
        if len(data_list) <= max_items:
            return data_list

        if "reminders" in str(filename):
            # Sadece gönderilmiş (sent=True) olanları silmeye çalış
            sent = [r for r in data_list if r.get("sent")]
            pending = [r for r in data_list if not r.get("sent")]
            
            if len(pending) > max_items:
                # Eğer bekleyenler bile çok fazlaysa, mecburen eskileri buda
                return data_list[-max_items:]
            
            # Değilse, bekleyenleri tut, gönderilmişlerden yer aç
            space_left = max_items - len(pending)
            return data_list[-(len(pending) + space_left):]
        
        elif "notes" in str(filename):
            # Notlar kutsaldır. 5000 limitine kadar dokunma.
            if len(data_list) > 5000:
                logger.warning(f"[MEMORY_GUARD] Notes exceeding 5000! Pruning oldest.")
                return data_list[-5000:]
            return data_list
            
        return data_list[-max_items:]

    def _save_json(self, path, data):
        """Atomic write + Sentinel Backup"""
        with self.lock:
            try:
                # Önce mevcut dosyayı yedeğe kopyala (eğer varsa ve boyutu > 0 ise)
                if path.exists() and path.stat().st_size > 0:
                    shutil.copy2(path, path.with_suffix(".json.bak"))

                # Create a temporary file
                fd, temp_path = tempfile.mkstemp(dir=self.storage_path, prefix=path.name + ".tmp")
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                
                # Pruning: Purpose-aware scaling
                data = self._prune_data(data, path.name)

                # Atomic rename
                shutil.move(temp_path, path)
            except Exception as e:
                logger.error(f"Save error {path}: {e}")
                check_error_threshold()

    def _generate_id(self, prefix: str, user_id: int) -> str:
        """Symmetrical ID generation for all entities"""
        return f"{prefix}_{user_id}_{get_now_utc().timestamp()}"

    def _transactional_update(self, file_path, data_list, update_func):
        """Unified Lock + Change + Save pattern"""
        result = None
        with self.lock:
            result = update_func(data_list)
        self._save_json(file_path, data_list)
        return result

    def add_note(self, user_id: int, text: str, source: str = "railway", category: str = "Genel") -> str:
        text = sanitize_text(text)
        note = {
            "id": self._generate_id(source, user_id),
            "user_id": user_id,
            "text": text,
            "category": category,
            "created": get_now_utc().isoformat(),
            "source": source
        }
        return self._transactional_update(self.notes_file, self.notes, lambda d: d.append(note) or note["id"])

    def get_notes(self, user_id: int, limit: int = 50) -> List[Dict]:
        user_notes = [n for n in self.notes if n["user_id"] == user_id]
        return user_notes[-limit:]

    def search_notes(self, user_id: int, query: str) -> List[Dict]:
        """Smarter search: Multiple keywords + Case insensitive"""
        keywords = query.lower().split()
        results = []
        for note in self.notes:
            if note["user_id"] == user_id:
                content = note["text"].lower()
                if all(kw in content for kw in keywords):
                    results.append(note)
        return results[-15:]

    def delete_note(self, note_id: str) -> bool:
        """Notu sil"""
        def update(data):
            for i, n in enumerate(data):
                if n["id"] == note_id:
                    data.pop(i)
                    return True
            return False
        return self._transactional_update(self.notes_file, self.notes, update)

    # ===== REMINDERS =====
    def add_reminder(self, user_id: int, text: str, remind_time: str, note_id: str = None) -> str:
        """Tek seferlik hatırlatıcı ekle"""
        text = sanitize_text(text)
        reminder = {
            "id": self._generate_id("rem", user_id),
            "user_id": user_id,
            "text": text,
            "remind_time": remind_time,  # ISO format (UTC)
            "note_id": note_id,
            "sent": False,
            "created": get_now_utc().isoformat()
        }
        return self._transactional_update(self.reminders_file, self.reminders, lambda d: d.append(reminder) or reminder["id"])

    def get_pending_reminders(self) -> List[Dict]:
        """Bekleyen hatırlatıcıları getir"""
        now = get_now_utc().isoformat()
        with self.lock:
            pending = []
            for r in self.reminders:
                if not r.get("sent", False) and r["remind_time"] <= now:
                    pending.append(r)
            return pending

    def claim_reminder(self, reminder_id: str) -> bool:
        """Idempotent Trigger: 'Sending' durumuna çekerek double-send engeller"""
        def update(data):
            for r in data:
                if r["id"] == reminder_id:
                    if r.get("sent") or r.get("processing"):
                        return False
                    r["processing"] = True
                    return True
            return False
        return self._transactional_update(self.reminders_file, self.reminders, update)

    def mark_reminder_sent(self, reminder_id: str):
        """Hatırlatıcıyı gönderildi olarak işaretle"""
        def update(data):
            for r in data:
                if r["id"] == reminder_id:
                    r["sent"] = True
                    r["processing"] = False
            return True
        self._transactional_update(self.reminders_file, self.reminders, update)

    def delete_reminder(self, reminder_id: str) -> bool:
        """Hatırlatıcıyı sil"""
        def update(data):
            for i, r in enumerate(data):
                if r["id"] == reminder_id:
                    data.pop(i)
                    return True
            return False
        return self._transactional_update(self.reminders_file, self.reminders, update)

    def reschedule_reminder(self, reminder_id: str, new_time: str) -> bool:
        """Ontological: 'Snoozing' is a state update of the same entity"""
        def update(data):
            for r in data:
                if r["id"] == reminder_id:
                    r["remind_time"] = new_time
                    r["sent"] = False  # Reset if it was sent
                    return True
            return False
        return self._transactional_update(self.reminders_file, self.reminders, update)

    def get_user_reminders(self, user_id: int) -> List[Dict]:
        """Kullanıcının hatırlatıcılarını getir"""
        return [r for r in self.reminders if r["user_id"] == user_id and not r.get("sent", False)]

    # ===== ROUTINES =====
    def add_routine(self, user_id: int, text: str, frequency: str, time: str) -> str:
        text = sanitize_text(text)
        routine = {
            "id": self._generate_id("rut", user_id),
            "user_id": user_id,
            "text": text,
            "frequency": frequency,
            "time": time,
            "last_sent": None,
            "created": get_now_utc().isoformat()
        }
        return self._transactional_update(self.routines_file, self.routines, lambda d: d.append(routine) or routine["id"])

    def get_routines(self) -> List[Dict]:
        with self.lock:
            return list(self.routines)

    def get_user_routines(self, user_id: int) -> List[Dict]:
        with self.lock:
            return [r for r in self.routines if r["user_id"] == user_id]

    def update_routine_last_sent(self, routine_id: str) -> bool:
        """Preemptive update to prevent double-send within the same job tick"""
        def update(data):
            for r in data:
                if r["id"] == routine_id:
                    now = get_now_utc().isoformat()
                    # Check if already updated in this day (extra safety)
                    if r.get("last_sent"):
                        ls = parser.parse(r["last_sent"])
                        if ls.date() == get_now_utc().date():
                            return False
                    r["last_sent"] = now
                    return True
            return False
        return self._transactional_update(self.routines_file, self.routines, update)

    def delete_routine(self, routine_id: str) -> bool:
        def update(data):
            for i, r in enumerate(data):
                if r["id"] == routine_id:
                    data.pop(i)
                    return True
            return False
        return self._transactional_update(self.routines_file, self.routines, update)

    def clear_all_reminders(self, user_id: int) -> int:
        """Kullanıcının tüm bekleyen hatırlatıcılarını sil"""
        def update(data):
            initial_count = len(data)
            # data[:] modifies the list in-place
            data[:] = [r for r in data if r["user_id"] != user_id or r.get("sent")]
            return initial_count - len(data)
        return self._transactional_update(self.reminders_file, self.reminders, update)

    def clear_all_routines(self, user_id: int) -> int:
        """Kullanıcının tüm rutinlerini sil"""
        def update(data):
            initial_count = len(data)
            data[:] = [r for r in data if r["user_id"] != user_id]
            return initial_count - len(data)
        return self._transactional_update(self.routines_file, self.routines, update)

    def get_stats(self) -> Dict:
        return {
            "total_notes": len(self.notes),
            "pending_reminders": len([r for r in self.reminders if not r.get("sent")]),
            "active_routines": len(self.routines)
        }


# ==================== ARCHITECTURE: AI ABSTRACTION ====================
from abc import ABC, abstractmethod

class AIAgent(ABC):
    """Abstract interface for AI providers to prevent vendor lock-in"""
    @abstractmethod
    def chat(self, text: str) -> Optional[str]:
        pass

    @abstractmethod
    async def vision(self, image_data: bytes, prompt: str) -> Optional[str]:
        pass

    @abstractmethod
    def parse_intent(self, text: str) -> Dict:
        pass

# ==================== GROQ AGENT ====================
class GroqAgent(AIAgent):
    SYSTEM = """Sen yardımcı bir Türkçe asistanısın.
Kısa, öz ve dostça yanıtlar ver."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.chat_model = AI_MODELS["chat"]
        self.vision_model = AI_MODELS["vision"]
        self.whisper_model = AI_MODELS["whisper"]

    @retry_on_failure(retries=2)
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
            ai_breaker.report_failure()
            check_error_threshold()
            return None

    @async_retry_on_failure(retries=2)
    async def vision(self, image_data: bytes, prompt: str = "Resimdeki metni çıkar") -> Optional[str]:
        """Görüntüden metin çıkar veya görüntüyü analiz et"""
        import base64
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Vision error: {e}")
            ai_breaker.report_failure()
            check_error_threshold()
            return None

    def parse_intent(self, text: str) -> Dict:
        """UX: Extract intent and parameters from Natural Language"""
        prompt = f"""Şu kullanıcı mesajındaki niyeti (intent) ve parametreleri analiz et:
"{text}"

Yanıtı SADECE şu JSON formatında ver:
{{
  "intent": "note" | "reminder" | "routine" | "question" | "delete",
  "params": {{
    "text": "asıl mesaj içeriği",
    "time": "varsa zaman (HH:MM veya natural language)",
    "frequency": "routine ise sıklık",
    "category": "note ise kategori",
    "estimated_prep_minutes": "hazırlık süresi (int)",
    "target_query": "Eğer intent 'delete' ise, silinmek istenen öğeyi tanımlayan anahtar kelimeler (örn: 'video çekmek', 'toplantı', 'market listesi')"
  }}
}}

ÖNEMLİ (VETO KURALI): Eğer mesajın içinde 'sil', 'iptal', 'vazgeç', 'boşver' gibi bir niyet varsa, içinde saat/zaman geçse bile intent'i 'delete' yap. Silme/İptal niyeti, hatırlatıcı kurmaktan daha önceliklidir.
Önemli: Eğer bir zaman belirtilmemişse intent 'note' veya 'question' olmalıdır."""
        
        try:
            response = self.chat(prompt)
            # JSON temizleme
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Intent parsing error: {e}")
        return {"intent": "note", "params": {"text": text, "category": "Genel"}}

    @retry_on_failure(retries=2)
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
                    
                    # Sinyal Saflaştırma: Güven eşiği kontrolü
                    CONFIDENCE_THRESHOLD = 0.40  # Düşük ama gürültüden ayırmak için
                    if confidence < CONFIDENCE_THRESHOLD:
                        logger.warning(f"[TRANSCRIBE-6] Low confidence ({confidence}), signal might be noise.")
                        return f"__low_confidence__:{transcript}"
                    
                    if not transcript:
                        logger.warning("[TRANSCRIBE-6] Empty transcript")
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
            ai_breaker.report_failure()
            check_error_threshold()
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[TRANSCRIBE] CONNECTION ERROR: {e}")
            ai_breaker.report_failure()
            check_error_threshold()
            return None
        except Exception as e:
            logger.error(f"[TRANSCRIBE] UNEXPECTED ERROR: {type(e).__name__}: {e}")
            ai_breaker.report_failure()
            check_error_threshold()
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
        now_local = get_now_local()
        
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
        target_local = None

        # "yarın" kontrolü
        if "yarın" in time_str_lower:
            target_local = now_local + timedelta(days=1)
            target_local = target_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            logger.info(f"[PARSE_TIME] 'yarın' detected")

        # "bugün" kontrolü
        elif "bugün" in time_str_lower:
            target_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target_local < now_local:
                target_local += timedelta(days=1)
            logger.info(f"[PARSE_TIME] 'bugün' detected")

        else:
            # Gün isimleri
            days_tr = {"pazartesi": 0, "salı": 1, "çarşamba": 2, "perşembe": 3, "cuma": 4, "cumartesi": 5, "pazar": 6}
            for day_tr, day_idx in days_tr.items():
                if day_tr in time_str_lower:
                    days_ahead = (day_idx - now_local.weekday()) % 7
                    if days_ahead == 0: days_ahead = 7
                    target_local = now_local + timedelta(days=days_ahead)
                    target_local = target_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    logger.info(f"[PARSE_TIME] '{day_tr}' detected")
                    break

        # Sadece saat varsa
        if not target_local and time_pattern and len(time_str) <= 10:
            target_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target_local < now_local:
                target_local += timedelta(days=1)
            logger.info(f"[PARSE_TIME] Time only detected")

        if not target_local:
            logger.info(f"[PARSE_TIME] Falling back to dateutil parser...")
            # Eğer string YYYY-MM-DD ile başlıyorsa dayfirst=False olmalı
            is_iso_start = re.match(r'^\d{4}-\d{2}-\d{2}', time_str)
            target_local = parser.parse(time_str, fuzzy=True, dayfirst=not is_iso_start)
            
            if target_local.tzinfo is None:
                target_local = USER_TZ.localize(target_local)
            
            # Eğer parser geçmiş bir saat döndürdüyse (ve sadece tarih verilmişse) bugüne/yarına çek
            if target_local < now_local and len(time_str) <= 10:
                target_local += timedelta(days=1)

        # Convert to UTC for storage
        if target_local.tzinfo is None:
            target_local = USER_TZ.localize(target_local)
        
        target_utc = target_local.astimezone(pytz.UTC)
        
        # --- TEMPORAL GUARD: Hallucination Prevention ---
        # 1 yıldan fazla ileriye hatırlatıcı kurulmasını engelle (AI hatası olabilir)
        if target_local.year > now_local.year + 1:
            logger.warning(f"[PARSE_TIME] Hallucination suspected: {target_local.year}")
            return None

        logger.info(f"[PARSE_TIME] Final: Local {target_local} -> UTC {target_utc.isoformat()}")
        return target_utc.isoformat()

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
    return jsonify({
        "status": "ok", 
        "service": "railway-bot", 
        "timestamp": get_now_utc().isoformat(),
        "storage": "connected" if storage else "disconnected"
    })

def check_sync_auth():
    token = request.headers.get("X-Sync-Token")
    expected = get_env("SYNC_TOKEN")
    if not expected or expected == "change-me-secure-token":
        logger.warning("INSECURE SYNC ATTEMPT: SYNC_TOKEN is missing or default!")
        return False
    return token == expected

@sync_app.route("/sync/from-local", methods=["POST"])
def from_local():
    if not check_sync_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.json
        notes = data.get("notes", [])
        user_id = data.get("user_id")
        
        added = 0
        with storage.lock:
            for note in notes:
                if not any(n.get("id") == note.get("id") for n in storage.notes):
                    note["synced_from"] = "local"
                    storage.notes.append(note)
                    added += 1
            if added > 0:
                storage._save_json(storage.notes_file, storage.notes)
        
        return jsonify({"status": "ok", "added": added})
    except Exception as e:
        logger.error(f"Sync error: {e}")
        return jsonify({"error": str(e)}), 500

@sync_app.route("/sync/to-local", methods=["GET"])
def to_local():
    if not check_sync_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        user_id = request.args.get("user_id", type=int)
        pending = []
        with storage.lock:
            for note in storage.notes:
                if note.get("synced_from") != "local" and not note.get("synced_to_local", False):
                    if user_id is None or note.get("user_id") == user_id:
                        pending.append(note)
        return jsonify({"status": "ok", "notes": pending})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@sync_app.route("/sync/mark-local-synced", methods=["POST"])
def mark_local_synced():
    if not check_sync_auth():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.json
        note_ids = data.get("note_ids", [])
        count = 0
        with storage.lock:
            for note in storage.notes:
                if note.get("id") in note_ids:
                    note["synced_to_local"] = True
                    count += 1
            if count > 0:
                storage._save_json(storage.notes_file, storage.notes)
        return jsonify({"status": "ok", "marked": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_flask():
    port = int(get_env("PORT", "8080"))
    logger.info(f"Sync API starting on port {port}")
    sync_app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)


# ==================== ARCHITECTURE: SERVICE LAYER ====================

class ReminderService:
    """Encapsulates business logic for reminders"""
    def __init__(self, storage):
        self.storage = storage

    def create(self, user_id: int, text: str, time_str: str, prep_mins: int = 0) -> List[str]:
        dt_utc = parse_reminder_time(time_str)
        if not dt_utc:
            return []
            
        ids = []
        # Main reminder
        main_id = self.storage.add_reminder(user_id, text, dt_utc.isoformat())
        if main_id:
            ids.append(main_id)
            
        # Prep reminder
        if prep_mins > 0:
            prep_dt = dt_utc - timedelta(minutes=prep_mins)
            # Eğer hazırlık zamanı geçmişte değilse ekle
            if prep_dt > get_now_utc():
                prep_text = f"🚨 HAZIRLIK: {text}"
                prep_id = self.storage.add_reminder(user_id, prep_text, prep_dt.isoformat())
                if prep_id:
                    ids.append(prep_id)
                    
        return ids

    def list_pending(self, user_id: int):
        return self.storage.get_user_reminders(user_id)

class RoutineService:
    """Encapsulates business logic for routines"""
    def __init__(self, storage):
        self.storage = storage

    def create(self, user_id: int, freq_str: str, time_str: str, text: str) -> Optional[str]:
        freq, time = parse_routine_frequency(freq_str)
        if freq and time:
            return self.storage.add_routine(user_id, text, freq, time)
        return None

# ==================== TELEGRAM BOT ====================
class RailwayBot:
    def __init__(self, storage_instance):
        try:
            self.storage = storage_instance
            groq_key = get_env("GROQ_API_KEY")
            # [Architecture] Service Injection
            self.groq: AIAgent = GroqAgent(groq_key)
            self.reminder_service = ReminderService(self.storage)
            self.routine_service = RoutineService(self.storage)
            
            # Google Calendar
            google_creds_json = get_env("GOOGLE_CREDENTIALS")
            google_token_json = get_env("GOOGLE_TOKEN")
            
            if google_creds_json:
                # Env var üzerinden başlat
                logger.info("Initializing Google Calendar from environment variables")
                self.calendar = GoogleCalendarManager(google_creds_json, google_token_json, is_path=False)
            else:
                # Dosya üzerinden başlat (Fallback)
                creds_path = os.path.join(os.path.dirname(__file__), "credentials.json")
                storage_dir = get_env("RAILWAY_VOLUME_URL", "/data/storage")
                token_path = os.path.join(storage_dir, "token.json")
                
                if not os.path.exists(token_path) and os.path.exists("token.json"):
                    token_path = "token.json"
                
                self.calendar = GoogleCalendarManager(creds_path, token_path, is_path=True)
            
            # [Teleology] State for bypassing intent
            self.user_search_mode = {} 
            # UX: Adaptive Kickstart
            self.pending_digests = set()
            
            # [Decoupling] App will be bound later via bind_app
            self.app = None
        except Exception as e:
            logger.error(f"[ERROR] RailwayBot init failed: {e}")
            raise

    def bind_app(self, application):
        """Bind the application instance and register for jobs"""
        self.app = application
        self.app.bot_data['bot_instance'] = self
        logger.info("[DEBUG] RailwayBot bound to Application and registered in bot_data")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info(f"=== START COMMAND RECEIVED from {update.effective_user.id} ===")
        user_id = update.effective_user.id
        
        # Persistent Bottom Keyboard
        reply_kb = [
            [KeyboardButton("🌟 Menü"), KeyboardButton("📋 Listele")],
            [KeyboardButton("🔍 Hafızada Ara"), KeyboardButton("📊 Durum")]
        ]
        persistent_markup = ReplyKeyboardMarkup(reply_kb, resize_keyboard=True)

        # Rich Inline Menu (Feature Universe)
        inline_kb = [
            [InlineKeyboardButton("🧠 Hafıza & Arama", callback_data=f"menu_memory"),
             InlineKeyboardButton("⏰ Zaman & Plan", callback_data=f"menu_time")],
            [InlineKeyboardButton("🛠️ Yardımcı Araçlar", callback_data=f"menu_tools"),
             InlineKeyboardButton("📊 Sistem Durumu", callback_data=f"status_{user_id}")],
            [InlineKeyboardButton("✨ Tümünü Listele", callback_data=f"list_all")]
        ]

        reply = f"""🌟 **Railway Asistan: Özellikler Evreni** 🌟

Merhaba {update.effective_user.first_name}! Ben senin dijital dış zihninim. Aşağıdaki butonları kullanarak bana hükmedebilirsin:

🧠 **HAFIZA**: Not al, ses at veya geçmişi sorgula.
⏰ **ZAMAN**: Hatırlatıcı kur, rutinlerini yönet.
🛠️ **ARAÇLAR**: Ses çeviri, Fotoğraf analizi, Google Takvim.

Klavye üzerindeki hazır butonları da kullanabilirsin! 👇"""

        await update.message.reply_text(
            reply,
            reply_markup=persistent_markup, # Set persistent keyboard
            parse_mode='Markdown'
        )
        
        # Also send the inline menu for discovery
        await update.message.reply_text(
            "Keşfetmek istediğin alanı seç:",
            reply_markup=InlineKeyboardMarkup(inline_kb)
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

        # Service üzerinden oluştur
        reminder_id = self.reminder_service.create(user_id, message, time_str)
        if not reminder_id:
            await update.message.reply_text(f"❌ Zaman formatı anlaşılamadı: {time_str}")
            return

        # İptal butonu
        keyboard = [[InlineKeyboardButton("❌ İptal Et", callback_data=f"canrem_{reminder_id}")]]

        await update.message.reply_text(
            f"✅ Hatırlatıcı ayarlandı!\n\n"
            f"⏰ {time_str}\n"
            f"📝 {message}",
            reply_markup=InlineKeyboardMarkup(keyboard)
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

        # Frekans ve saati ayrıştır (Fallback case handles legacy arg order)
        freq_str = context.args[0]
        time_str = context.args[1]
        message = " ".join(context.args[2:])

        # Service üzerinden oluştur
        routine_id = self.routine_service.create(user_id, freq_str, time_str, message)
        
        if routine_id:
            await update.message.reply_text(
                f"✅ Rutin hatırlatıcı ayarlandı!\n\n"
                f"🔄 {freq_str.capitalize()} • {time_str}\n"
                f"📝 {message}"
            )
        else:
            await update.message.reply_text(
                "❌ Format hatası!\n"
                "Sıklık: günlük, haftalık, aylık veya gün ismi (Pazartesi vb.)\n"
                "Saat: HH:MM formatında olmalı (örn: 09:00)"
            )

    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/list komutu - hatırlatıcı ve rutin listesi"""
        user_id = update.effective_user.id

        reminders = self.storage.get_user_reminders(user_id)
        routines = self.storage.get_user_routines(user_id)

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

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/clear komutu - Toplu silme arayüzü"""
        keyboard = [
            [InlineKeyboardButton("⏰ Tüm Hatırlatıcıları Sil", callback_data="clear_rem")],
            [InlineKeyboardButton("🔄 Tüm Rutinleri Sil", callback_data="clear_ro")],
            [InlineKeyboardButton("📅 Takvimi Temizle (İlaç)", callback_data="clear_gcal_pharma")],
            [InlineKeyboardButton("❌ İptal", callback_data="clear_cancel")]
        ]
        await update.message.reply_text(
            "🗑️ **Toplu Silme Menüsü**\n\nHangi görevleri temizlemek istersiniz?",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )

    async def auth_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/auth - Google Calendar yetkilendirme"""
        if self.calendar.is_authenticated():
            await update.message.reply_text("✅ Google Takvim zaten bağlı!")
            return

        auth_url = self.calendar.get_auth_url()
        reply = (
            "🔗 **Google Takvim Bağlantısı**\n\n"
            "1. [Buraya tıklayarak giriş yapın](" + auth_url + ")\n"
            "2. Çıkan ekranda izinleri onaylayın.\n"
            "3. Tarayıcıda 'bağlanılamıyor' (localhost) hatası alacaksınız, sorun değil.\n"
            "4. Adres çubuğundaki **TÜM linki** buraya yapıştırıp bana gönderin."
        )
        await update.message.reply_text(reply, parse_mode='Markdown')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        text = update.message.text

        # --- UX: PERSISTENT KEYBOARD ROUTING ---
        if text == "🌟 Menü":
            await self.start(update, context)
            return
        elif text == "📋 Listele":
            await self.list_command(update, context)
            return
        elif text == "🔍 Hafızada Ara":
            self.user_search_mode[user_id] = True
            await update.message.reply_text("🔍 Aramak istediğiniz kelimeleri yazın...")
            return
        elif text == "📊 Durum":
            await self.button_callback(update, context, forced_data=f"status_{user_id}")
            return

        # Google Auth linki mi? (Critical setup bypass)
        if "localhost" in text and "code=" in text:
            try:
                self.calendar.finalize_auth(text)
                await update.message.reply_text("✅ Google Takvim başarıyla bağlandı! Artık hatırlatıcılar otomatik senkronize edilecek.")
                return
            except Exception as e:
                corr_id = generate_correlation_id()
                logger.error(f"[{corr_id}] Auth finalization failed: {e}")
                await update.message.reply_text(f"❌ Bağlantı hatası (ID: {corr_id}). Lütfen tekrar deneyin veya desteğe yazın.")
                return

        # --- UX: ADAPTIVE KICKSTART ---
        if user_id in self.pending_digests:
            self.pending_digests.remove(user_id)
            await self._send_daily_digest(user_id)
            # Devam et (notu kaydet veya soruyu sor)

        # --- SUPPORT: SAFE MODE / RECOVERY VISIBILITY ---
        if SAFE_MODE or not ai_breaker.can_proceed():
            corr_id = generate_correlation_id()
            self.storage.add_note(user_id, text, source=f"failsafe-{corr_id}", category="Genel")
            reason = "Aşırı yük veya bağlantı sorunu" if not ai_breaker.can_proceed() else "Güvenli Mod aktif"
            await update.message.reply_text(
                f"🛡️ **Zarif Çöküş Protokolü Devrede**\n\n"
                f"Şu an {reason} nedeniyle anlama motorunu dinlendiriyorum. "
                f"Veriniz kaybolmadı, standart bir not olarak kaydedildi.\n\n"
                f"🔍 Takip ID: `{corr_id}`",
                parse_mode='Markdown'
            )
            return

        await update.message.chat.send_action("typing")

        # --- TELEOLOGY: Bypass Intent if in Search Mode ---
        if self.user_search_mode.get(user_id):
            self.user_search_mode[user_id] = False # Clear mode
            await self._perform_deterministic_search(update, user_id, text)
            return

        # --- SAFE MODE CHECK ---
        if SAFE_MODE or not ai_breaker.can_proceed():
            # Graceful Degradation: Sadece temel not kaydı
            self.storage.add_note(user_id, text, source="railway-safe", category="Genel")
            await update.message.reply_text("🛡️ [Güvenli Mod] Notunuz AI yardımı olmadan standart olarak kaydedildi.")
            return

        # --- UX: ZERO-COMMAND INTENT INCEPTION ---
        try:
            inception = self.groq.parse_intent(text)
            intent = inception.get("intent", "note")
            params = inception.get("params", {})
        except Exception as e:
            corr_id = generate_correlation_id()
            logger.error(f"[{corr_id}] Intent inception error: {e}")
            self.storage.add_note(user_id, text, source=f"error-fallback-{corr_id}", category="Genel")
            await update.message.reply_text(
                f"⚠️ **Anlama Hatası**\n\n"
                f"Mesajınızı tam çözümleyemedim ama güvenle notlarıma ekledim.\n"
                f"Sistem şu an biraz yorgun görünüyor.\n\n"
                f"🔍 Takip ID: `{corr_id}`",
                parse_mode='Markdown'
            )
            return

        # [Architecture] Intent Routing Strategy
        await self._handle_routed_intent(update, user_id, text, intent, params)
    async def _handle_routed_intent(self, update, user_id, text, intent, params):
        """Routes the parsed intent to the specialized handler"""
        if intent == "reminder" and params.get("time"):
            await self._handle_reminder_intent(update, user_id, text, params)
        elif intent == "routine" and params.get("time") and params.get("frequency"):
            await self._handle_routine_intent(update, user_id, params)
        elif intent == "question":
            await self._perform_semantic_inquiry(update, user_id, text)
        elif intent == "delete":
            await self._handle_delete_intent(update, user_id, params)
        else:
            await self._handle_note_intent(update, user_id, text, params)

    async def _handle_reminder_intent(self, update, user_id, text, params):
        rem_text = params.get("text") or text
        time_str = params.get("time")
        prep_mins = int(params.get("estimated_prep_minutes", 0))
        
        ids = self.reminder_service.create(user_id, rem_text, time_str, prep_mins)
        
        if not ids:
            await update.message.reply_text(f"❌ Zaman formatı anlaşılamadı: {time_str}")
            return
            
        if len(ids) > 1:
            await update.message.reply_text(
                f"⏰ **Çift Hatırlatıcı Kuruldu!**\n\n"
                f"🎯 **Hedef:** {time_str} - {rem_text}\n"
                f"🛠️ **Hazırlık:** {prep_mins} dakika öncesine (`🚨 HAZIRLIK`) bir uyarı daha ekledim."
            )
        else:
            await update.message.reply_text(f"⏰ Tamamdır! Hatırlatıcı eklendi: {time_str}\n📝 {rem_text}")

    async def _handle_routine_intent(self, update, user_id, params):
        if self.routine_service.create(user_id, params["frequency"], params["time"], params["text"]):
            await update.message.reply_text(f"🔄 Rutin eklendi: {params['frequency']} @ {params['time']}\n📝 {params['text']}")

    async def _handle_note_intent(self, update, user_id, text, params):
        category = params.get("category") or "Genel"
        self.storage.add_note(user_id, text, source="railway-inception", category=category)
        ai_confirm = self.groq.chat(f"Kullanıcının şu notunu '{category}' kategorisine kaydettim: '{text}'. Çok kısa ve zekice bir teyit ver.")
        await update.message.reply_text(ai_confirm or f"✅ Not kaydedildi. (#{category})")

    async def _handle_delete_intent(self, update, user_id, params):
        """Autonomous Search-and-Destroy for notes/reminders"""
        query = params.get("target_query")
        if not query:
            await update.message.reply_text("🤔 Neyi silmek istediğinizi tam anlayamadım.")
            return

        # 1. Önce hatırlatıcılarda ara
        reminders = self.storage.get_user_reminders(user_id)
        target_rem = None
        for r in reversed(reminders): # En günceli bul
            if query.lower() in r["text"].lower():
                target_rem = r
                break
        
        if target_rem:
            if self.storage.delete_reminder(target_rem["id"]):
                await update.message.reply_text(f"🗑️ Hatırlatıcı silindi: {target_rem['text']}")
                return

        # 2. Notlarda ara
        notes = self.storage.search_notes(user_id, query)
        if notes:
            target_note = notes[-1] # En günceli bul
            if self.storage.delete_note(target_note["id"]):
                await update.message.reply_text(f"🗑️ Not silindi: {target_note['text']}")
                return
        
        await update.message.reply_text(f"🔍 '{query}' ile ilgili silinecek bir kayıt bulamadım.")

    async def _send_daily_digest(self, user_id: int):
        """UX: Send daily summary on first interaction"""
        now_local = get_now_local()
        reminders = self.storage.get_user_reminders(user_id)
        routines = self.storage.get_user_routines(user_id)
        
        if not reminders and not routines:
            return
            
        reply = f"☀️ **GÜNAYDIN! Günlük Özetiniz**\n"
        reply += f"📅 {now_local.strftime('%d %B %Y %A')}\n\n"
        
        if reminders:
            reply += "⏰ **Bugünkü Hatırlatıcılar:**\n"
            today_count = 0
            for r in reminders:
                dt = parser.parse(r["remind_time"])
                if dt.date() == now_local.date():
                    reply += f"• {dt.strftime('%H:%M')}: {r['text'][:40]}\n"
                    today_count += 1
            if today_count == 0:
                reply += "_Bugün için bekleyen hatırlatıcı yok._\n"
        
        reply += "\n"
        
        if routines:
            reply += "🔄 **Rutinler:**\n"
            for r in routines:
                reply += f"• {r['time']}: {r['text'][:40]}\n"
        
        await self.app.bot.send_message(chat_id=user_id, text=reply, parse_mode='Markdown')
        logger.info(f"Adaptive daily digest sent to {user_id}")

    async def _perform_deterministic_search(self, update: Update, user_id: int, query: str):
        """Ontological: 'Finding' action (Keyword matching)"""
        matching_knowledge = self.storage.search_notes(user_id, query)
        
        if not matching_knowledge:
            await update.message.reply_text("🔍 Eşleşen bir bilgi bulunamadı.")
            return

        reply = "🔍 **Bulunan Sonuçlar:**\n\n"
        for item in matching_knowledge:
            reply += f"• {item['text']}\n"
        await update.message.reply_text(reply, parse_mode='Markdown')

    async def _perform_semantic_inquiry(self, update: Update, user_id: int, inquiry_text: str):
        """Ontological: 'Inquiring' action (AI Reasoning)"""
        if SAFE_MODE:
             await update.message.reply_text("🛡️ [Güvenli Mod] Hafıza araması şu an devre dışı. Lütfen daha sonra deneyin.")
             return

        # 1. Ham arama yap (Keyword bazlı context toplama)
        relevant_context = self.storage.search_notes(user_id, inquiry_text)
        
        # 2. Eğer hiç sonuç yoksa, geniş kapsamlı arama (son 30 not)
        if not relevant_context:
            relevant_context = self.storage.get_notes(user_id, limit=30)

        if relevant_context:
            # 3. AI'ya Bağlam (Context) olarak sun
            context_text = "\n".join([f"- [{n['category']}] {n['text']}" for n in relevant_context])
            
            prompt = f"""Kullanıcının geçmiş notları aşağıda verilmiştir:
---
{context_text}
---
Kullanıcı sorusu: "{inquiry_text}"

Lütfen SADECE yukarıdaki notlara dayanarak soruyu yanıtla. 
- Eğer bilgi yoksa "Bu konuda notlarımda bir bilgi bulamadım" de.
- Bilgi varsa özetle ve kategorileri belirt.
- Yanıtı Türkçe ve samimi bir dille ver."""

            ai_response = self.groq.chat(prompt)
            if ai_response:
                await update.message.reply_text(f"🤖 **Hafıza:**\n\n{ai_response}", parse_mode='Markdown')
        else:
            await update.message.reply_text(f"🤖 **AI:**\n\n{ai_response}", parse_mode='Markdown')

    async def _update_ui(self, update: Update, text: str, reply_markup=None, parse_mode=None):
        """Support: Handle UI updates for both callback queries and direct replies"""
        if update.callback_query:
            try:
                await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
            except Exception:
                await update.callback_query.message.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
        else:
            await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE, forced_data: str = None):
        query = update.callback_query
        user_id = update.effective_user.id
        data = forced_data or query.data
        
        if query:
            await query.answer()

        parts = data.split('_')
        action = parts[0]

        if action == "note":
            await query.edit_message_text("📝 Notunuzu yazın...")
        elif action == "search":
            self.user_search_mode[user_id] = True
            await query.edit_message_text("🔍 Aramak istediğiniz kelimeleri yazın...")
        elif action == "menu":
            target = parts[1]
            if target == "memory":
                text = "🧠 **Hafıza & AI Soruları**\n\n• **Not Al:** Sadece yaz veya ses at, gerisini bana bırak.\n• **Soru Sor:** \"Geçen hafta ne demiştik?\", \"Video notumu bul\" gibi sorularla geçmişi sorgula.\n• **Arama:** /list komutuyla veya aşağıdaki 'Ara' butonuyla kelime bazlı arama yap."
                kb = [[InlineKeyboardButton("🔍 Kelime İle Ara", callback_data=f"search_{user_id}"),
                       InlineKeyboardButton("🔙 Geri", callback_data="start_menu")]]
                await self._update_ui(update, text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
            elif target == "time":
                text = "⏰ **Zaman & Planlama**\n\n• **Hatırlatıcı:** \"Yarın 10:00 toplantı\" yazman yeterli.\n• **Rutin:** /routine ile tekrarlanan görevler oluştur.\n• **Akıllı Hazırlık:** Önemli işlerde senin için otomatik hazırlık uyarıları kurarım."
                kb = [[InlineKeyboardButton("📋 Bekleyenleri Listele", callback_data="list_all"),
                       InlineKeyboardButton("🔙 Geri", callback_data="start_menu")]]
                await self._update_ui(update, text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
            elif target == "tools":
                text = "🛠️ **Yardımcı Araçlar**\n\n• 🎙️ **Sesli Mesaj:** Uzun sesleri anında metne çevirip özetlerim.\n• 🖼️ **Görsel Analiz:** Fotoğraf at, içindeki bilgileri not alayım.\n• 🗓️ **Google Takvim:** /auth ile bağla, her şey senkron kalsın."
                kb = [[InlineKeyboardButton("🔙 Geri", callback_data="start_menu")]]
                await self._update_ui(update, text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
        elif data == "start_menu":
            await self.start(update, context) # Re-send main menu
        elif data == "list_all":
            await self.list_command(update, context)
        elif action == "reminder":
            await self._update_ui(update, 
                "⏰ Hatırlatıcı eklemek için:\n\n/remind <zaman> <mesaj>\n\n"
                "Örnek: /remind 15:30 Toplantı"
            )
        elif action == "routine":
            await self._update_ui(update, 
                "🔄 Rutin eklemek için:\n\n/routine <sıklık> <saat> <mesaj>\n\n"
                "Örnek: /routine günlük 09:00 Kahve"
            )
        elif action == "status":
            stats = self.storage.get_stats()
            reply = f"📊 **Durum**\n\n📝 Not: {stats['total_notes']}\n⏰ Hatırlatıcı: {stats['pending_reminders']}\n🔄 Rutin: {stats['active_routines']}"
            await self._update_ui(update, reply, parse_mode='Markdown')
        
        elif data == "clear_rem":
            count = self.storage.clear_all_reminders(user_id)
            await self._update_ui(update, f"✅ {count} adet bekleyen hatırlatıcı temizlendi.")
        
        elif data == "clear_ro":
            count = self.storage.clear_all_routines(user_id)
            await self._update_ui(update, f"✅ {count} adet rutin temizlendi.")
            
        elif data == "clear_cancel":
            await self._update_ui(update, "❌ İşlem iptal edildi.")
        
        elif action == "canrem":
            # Hatırlatıcı iptal
            # format: canrem_rem_user_timestamp
            reminder_id = "_".join(parts[1:])
            if self.storage.delete_reminder(reminder_id):
                await self._update_ui(update, "❌ Hatırlatıcı iptal edildi.")
            else:
                await self._update_ui(update, "⚠️ Hatırlatıcı bulunamadı veya zaten silinmiş.")
        
        elif data == "clear_gcal_pharma":
            if not self.calendar.is_authenticated():
                await self._update_ui(update, "❌ Önce bota takviminizi bağlamanız lazım: /auth")
                return
            
            count = self.calendar.clear_events_by_query("İLAÇ")
            await self._update_ui(update, f"✨ Takviminizdeki {count} adet ilaç hatırlatıcısı temizlendi!")
        
        elif action == "snooze":
            # Erteleme: snooze_rem_user_timestamp_dakika
            minutes = int(parts[-1])
            rem_id = "_".join(parts[1:-1])
            
            new_time = (get_now_utc() + timedelta(minutes=minutes)).isoformat()
            if self.storage.reschedule_reminder(rem_id, new_time):
                await self._update_ui(update, f"⏳ {minutes} dakika ertelendi.")
            else:
                await self._update_ui(update, "⚠️ Hatırlatıcı bulunamadı veya güncellenemedi.")
        
        # --- SUPPORT: Button Action Global Catch ---
        elif action:
            logger.info(f"Action triggered: {action} by {user_id}")

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Sesli mesaj işle - Deepgram transkripsiyon + AI sınıflandırma"""
        try:
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
                await status_msg.edit_text("❌ Ses anlaşılamadı (sessizlik veya teknik sorun)")
                return

            if transcript.startswith("__low_confidence__"):
                actual_text = transcript.split(":", 1)[1]
                logger.warning(f"[VOICE] Low confidence transcript: {actual_text}")
                await status_msg.edit_text(f"⚠️ Ses çok net değil, ama şunu anladım:\n\n\"{actual_text}\"\n\nLütfen daha net veya yazılı olarak deneyin.")
                return

            logger.info(f"[VOICE] SUCCESS! Transcript: {transcript}")
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
                # AI ile kategori tahmini
                category_prompt = f"Şu notun kategorisini (tek kelime, ör: İş, Kişisel, Finans, Sağlık) belirle: '{transcript}'. Sadece kelimeyi döndür."
                category = self.groq.chat(category_prompt) or "Genel"
                category = category.strip().strip("'").strip('"')
                
                self.storage.add_note(user_id, f"[Ses] {transcript}", source="voice", category=category)
                await update.message.reply_text(f"📝 Not alındı (#{category}):\n\n{transcript}")
            else:  # chat
                ai_response = self.groq.chat(transcript)
                if ai_response:
                    await update.message.reply_text(f"🤖 **AI:**\n\n{ai_response}", parse_mode='Markdown')

        except Exception as e:
            corr_id = generate_correlation_id()
            logger.error(f"[{corr_id}] [VOICE] EXCEPTION: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[{corr_id}] [VOICE] Traceback:\n{traceback.format_exc()}")
            await update.message.reply_text(
                f"🔇 **Ses İşleme Kesintisi**\n\n"
                f"Sesinizi şu an metne dökemiyorum. Teknik bir pürüz (ID: `{corr_id}`) oluştu. "
                f"Lütfen biraz sonra tekrar dener misiniz?\n\n"
                f"🔍 Takip ID: `{corr_id}`",
                parse_mode='Markdown'
            )

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Fotoğrafı işle - OCR ve analiz"""
        user_id = update.effective_user.id
        photo = update.message.photo[-1]  # En yüksek kalite
        
        await update.message.chat.send_action("upload_photo")
        status_msg = await update.message.reply_text("👁️ Görsel analiz ediliyor...")
        
        try:
            # Fotoğrafı indir
            file = await photo.get_file()
            img_bytearray = await file.download_as_bytearray()
            
            # Groq Vision ile analiz
            prompt = "Bu görseldeki metni oku ve bir not olarak özetle. Eğer bir belge değilse görselde neler olduğunu anlat."
            analysis = await self.groq.vision(bytes(img_bytearray), prompt)
            
            if not analysis:
                await status_msg.edit_text("❌ Görsel analiz edilemedi")
                return
                
            # AI ile kategori tahmini
            category_prompt = f"Şu görsel analizinin kategorisini belirle: '{analysis[:200]}'. Sadece kategori ismini (İş, Kişisel, Finans vb.) döndür."
            category = self.groq.chat(category_prompt) or "Görsel"
            category = category.strip().strip("'").strip('"')
            
            self.storage.add_note(user_id, f"[Görsel] {analysis}", source="photo", category=category)
            await status_msg.delete()
            await update.message.reply_text(f"📸 **Görsel Not (# {category}):**\n\n{analysis}")
            
        except Exception as e:
            corr_id = generate_correlation_id()
            logger.error(f"[{corr_id}] Photo handling error: {e}")
            await update.message.reply_text(
                f"🖼️ **Görsel Analiz Kesintisi**\n\n"
                f"Görselinizi şu an analiz edemiyorum. Dosya boyutu veya bağlantı kaynaklı bir durum olabilir.\n\n"
                f"🔍 Takip ID: `{corr_id}`",
                parse_mode='Markdown'
            )

    async def _process_reminder_from_voice(self, update: Update, transcript: str):
        """Sesten hatırlatıcı çıkar"""
        user_id = update.effective_user.id
        now_local = get_now_local()
        logger.info(f"[REMINDER] Processing reminder from voice for user {user_id}")
        logger.info(f"[REMINDER] Transcript: {transcript}")

        # AY İSİMLERİ (Ambiyans giderme)
        tr_months = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
            7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
        }
        
        # Ambiguity removal: Use words for months
        now_str_readable = f"{now_local.day} {tr_months[now_local.month]} {now_local.year} {now_local.strftime('%A %H:%M')}"
        now_iso = now_local.strftime("%Y-%m-%d %H:%M")
        
        logger.info(f"[REMINDER] Context Time: {now_str_readable}")

        prompt = f"""Şu anki zaman: {now_str_readable} (ISO: {now_iso})
Kullanıcı sesi: "{transcript}"

Bu ifadeden hatırlatıcı zamanını ve mesajını çıkar. JSON formatında dön:
{{
  "time": "YYYY-MM-DD HH:MM",
  "message": "mesaj",
  "is_relative": true/false (Dakika, saat, gün sonra gibi ifadeler varsa true)
}}

KRİTİK KURALLAR:
1. Bugün {now_local.day}. gündeyiz, ay {now_local.month}. ay ({tr_months[now_local.month]}).
2. Karıştırma: 12.01 (12 Ocak) ile 01.12 (1 Aralık) farklıdır. Mutlaka ISO (YYYY-MM-DD) kullan.
3. Eğer kullanıcı "1 dakika sonra", "yarım saat sonra" gibi nispeten küçük bir süre belirtiyorsa, gün ve ayı ASLA değiştirme.

Sadece JSON döndür."""

        try:
            logger.info("[REMINDER] Calling Groq API...")
            response = self.groq.client.chat.completions.create(
                model=self.groq.chat_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200
            )

            raw_content = response.choices[0].message.content
            logger.info(f"[REMINDER] Groq response: {raw_content}")

            result = json.loads(raw_content)
            time_str = result.get("time", "").strip()
            message = result.get("message", transcript).strip()
            is_relative = result.get("is_relative", False)

            if time_str:
                remind_time = None
                try:
                    # YYYY-MM-DD HH:MM formatı kontrolü
                    if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', time_str):
                        dt_parsed = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                        
                        # --- TEMPORAL HALLUCINATION GUARD (Fail-Safe) ---
                        # Eğer AI ayı yanlışlıkla (flipped) döndürdüyse ve işlem 'relative' ise düzelt
                        # Örnek: Ocak'tayız ama AI Aralık döndürdü.
                        if is_relative and dt_parsed.month != now_local.month:
                            # 1 aylık bir sapma normal olabilir (ayın sonunda yarın dendiğinde)
                            # Ama 11 aylık bir sapma (Jan vs Dec flip) kesinlikle hatadır.
                            if abs(dt_parsed.month - now_local.month) >= 10:
                                logger.warning(f"[FAIL-SAFE] Detected Month Hallucination! Correcting {dt_parsed.month} to {now_local.month}")
                                dt_parsed = dt_parsed.replace(month=now_local.month, day=now_local.day)

                        dt_local = USER_TZ.localize(dt_parsed)
                        remind_time = dt_local.astimezone(pytz.UTC).isoformat()
                        logger.info(f"[REMINDER] Sentinel Parse success: {remind_time}")
                    else:
                        remind_time = parse_reminder_time(time_str)
                except Exception as e:
                    logger.error(f"[REMINDER] Parse logic fail: {e}")
                    remind_time = parse_reminder_time(time_str)
                
                if remind_time:
                    # ID'yi yakalayalım
                    reminder_id = storage.add_reminder(user_id, message, remind_time)

                    # UTC'den yerel saate çevir
                    dt_utc = parser.parse(remind_time)
                    if dt_utc.tzinfo is None:
                        dt_utc = pytz.UTC.localize(dt_utc)
                    dt_local = dt_utc.astimezone(USER_TZ)
                    readable = dt_local.strftime("%d.%m.%Y %H:%M")
                    
                    logger.info(f"[REMINDER] SUCCESS! Reminder set for {readable}")
                    
                    # İptal butonu
                    keyboard = [[InlineKeyboardButton("❌ İptal Et", callback_data=f"canrem_{reminder_id}")]]

                    # Google Calendar Sync
                    if self.calendar.is_authenticated():
                        try:
                            self.calendar.add_event(f"⏰ {message}", remind_time)
                            logger.info(f"Synced to GCal: {message}")
                        except Exception as e:
                            logger.error(f"GCal Sync error: {e}")

                    await update.message.reply_text(
                        f"✅ Hatırlatıcı ayarlandı!\n\n📅 {readable}\n📝 {message}",
                        reply_markup=InlineKeyboardMarkup(keyboard)
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
            self.storage.add_note(user_id, f"[Ses] {transcript}", source="voice")
            await update.message.reply_text(f"📝 Not alındı:\n\n{transcript}")
        except Exception as e:
            logger.error(f"[REMINDER] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[REMINDER] Traceback: {traceback.format_exc()}")
            self.storage.add_note(user_id, f"[Ses] {transcript}", source="voice")
            await update.message.reply_text(f"📝 Not alındı:\n\n{transcript}")

    async def _process_routine_from_voice(self, update: Update, transcript: str):
        """Sesten rutin çıkar"""
        user_id = update.effective_user.id
        logger.info(f"[ROUTINE] Processing routine from voice for user {user_id}")

        # AI ile rutini çıkar
        now_str = get_now_local().strftime("%Y-%m-%d %H:%M")
        prompt = f"""Sistem Zamanı: {now_str}
Bu metinden rutin sıklığını, saatini ve mesajını çıkar. JSON formatında döndür:
{{"frequency": "günlük/haftalık/aylık/gün adı", "time": "HH:MM", "message": "mesaj"}}

Örnekler:
- "her gün sabah 8'de ilaç" → {{"frequency": "günlük", "time": "08:00", "message": "ilaç iç"}}
- "pazartesileri 9'da toplantı" → {{"frequency": "Pazartesi", "time": "09:00", "message": "toplantı"}}

Metin: {transcript}

Sadece JSON döndür."""

        try:
            logger.info("[ROUTINE] Calling Groq API...")
            response = self.groq.client.chat.completions.create(
                model=self.groq.chat_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200
            )

            import json
            result = json.loads(response.choices[0].message.content)
            freq = result.get("frequency", "daily")
            time_str = result.get("time", "09:00")
            message = result.get("message", transcript)

            self.storage.add_routine(user_id, message, freq, time_str)
            
            await update.message.reply_text(
                f"✅ Rutin ayarlandı!\n\n🔄 {freq.capitalize()} • {time_str}\n📝 {message}"
            )

        except Exception as e:
            logger.error(f"[ROUTINE] Error: {e}")
            self.storage.add_note(user_id, f"[Ses-Rutin] {transcript}", source="voice")
            await update.message.reply_text(f"📝 Not alındı (rutin anlaşılamadı):\n\n{transcript}")


# ==================== REMINDER CHECKER ====================
async def check_reminders_job(app: Application):
    """Periyodik hatırlatıcı kontrolü"""
    bot = app.bot_data.get('bot_instance')
    if not bot:
        logger.warning("[JOBS] check_reminders_job: bot_instance NOT FOUND in bot_data")
        return

    now_utc = get_now_utc().isoformat()
    pending = bot.storage.get_pending_reminders()

    for reminder in pending:
        # Idempotent Trigger: Önce "claim" et
        if not bot.storage.claim_reminder(reminder["id"]):
            continue

        try:
            user_id = reminder["user_id"]
            text = reminder["text"]
            # remind_time storage'da UTC ISO formatında
            dt_utc = parser.parse(reminder["remind_time"])
            if dt_utc.tzinfo is None:
                dt_utc = pytz.UTC.localize(dt_utc)
            
            # Kullanıcıya yerel saatle göster
            dt_local = dt_utc.astimezone(USER_TZ)
            readable_time = dt_local.strftime("%d.%m.%Y %H:%M")

            # Erteleme butonları
            keyboard = [
                [InlineKeyboardButton("15 dk", callback_data=f"snooze_{reminder['id']}_15"),
                 InlineKeyboardButton("1 saat", callback_data=f"snooze_{reminder['id']}_60"),
                 InlineKeyboardButton("Yarın", callback_data=f"snooze_{reminder['id']}_1440")]
            ]

            await app.bot.send_message(
                chat_id=user_id,
                text=f"⏰ **HATIRLATICI**\n\n{readable_time}\n📝 {text}",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='Markdown'
            )

            bot.storage.mark_reminder_sent(reminder["id"])
            logger.info(f"Reminder sent to {user_id}: {text[:30]}")

        except Exception as e:
            logger.error(f"Error sending reminder: {e}")


async def check_routines_job(app: Application):
    """Rutin hatırlatıcı kontrolü"""
    bot = app.bot_data.get('bot_instance')
    if not bot:
        logger.warning("[JOBS] check_routines_job: bot_instance NOT FOUND in bot_data")
        return

    now_local = get_now_local()
    current_time = now_local.strftime("%H:%M")
    current_weekday = now_local.weekday()  # 0=Monday

    days_tr_map = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba",
                   3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}

    routines = bot.storage.get_routines()

    for routine in routines:
        try:
            should_send = False
            freq = routine["frequency"].lower()
            routine_time = routine["time"]

            # Saat kontrolü (tam dakika eşleşmesi)
            if routine_time != current_time:
                continue

            # Frekans kontrolü
            if freq in ["daily", "günlük"]:
                should_send = True
            elif freq in ["weekly", "haftalık"]:
                if current_weekday == 0:  # Pazartesi
                    should_send = True
            elif freq in ["monthly", "aylık"]:
                if now_local.day == 1:
                    should_send = True
            elif freq.capitalize() in days_tr_map.values():
                if days_tr_map[current_weekday] == freq.capitalize():
                    should_send = True

            # Last sent kontrolü (aynı gün içinde tekrar gönderme)
            if routine.get("last_sent"):
                # last_sent UTC ISO formatında
                last_sent = parser.parse(routine["last_sent"])
                if last_sent.tzinfo is None:
                    last_sent = pytz.UTC.localize(last_sent)
                
                # Yerel tarihe çevirip gün farkına bak
                last_sent_local = last_sent.astimezone(USER_TZ)
                if last_sent_local.date() == now_local.date():
                    continue

            if should_send:
                # Preemptive update for determinism
                if not bot.storage.update_routine_last_sent(routine["id"]):
                    continue

                user_id = routine["user_id"]
                text = routine["text"]

                await app.bot.send_message(
                    chat_id=user_id,
                    text=f"🔄 **RUTİN HATIRLATICI**\n\n{routine['frequency']} • {routine_time}\n📝 {text}",
                    parse_mode='Markdown'
                )
                logger.info(f"Routine sent to {user_id}: {text[:30]}")

        except Exception as e:
            logger.error(f"Error sending routine: {e}")


async def daily_digest_job(app: Application):
    """Her sabah 08:30'da özetleri 'hazır' olarak işaretle"""
    # RailwayBot instance üzerinden pending_digests'e erişmemiz lazım.
    # Job'lar app.bot_data üzerinden veya global bot instance üzerinden çalışabilir.
    # Buradaki bot instance'ı RailwayBot içindeki self.app'tir.
    
    # Tüm kullanıcıları bul
    user_ids = set([r["user_id"] for r in storage.reminders] + 
                   [n["user_id"] for n in storage.notes] +
                   [ro["user_id"] for ro in storage.routines])
    
    # Global bot instance'ı bul (veya app.bot_data'ya koy)
    if 'bot_instance' in app.bot_data:
        bot_instance = app.bot_data['bot_instance']
        for user_id in user_ids:
            bot_instance.pending_digests.add(user_id)
        logger.info(f"Marked {len(user_ids)} users for adaptive digest")

# ==================== MAIN ====================
def main():
    global storage

    telegram_token = get_env("TELEGRAM_TOKEN")
    groq_key = get_env("GROQ_API_KEY")
    
    # Railway otomatik olarak PUBLIC_URL veya RAILWAY_PUBLIC_DOMAIN sağlar
    webhook_host = get_env("RAILWAY_PUBLIC_DOMAIN", "")
    port = int(get_env("PORT", "8080"))

    if not telegram_token or not groq_key:
        logger.error("TELEGRAM_TOKEN or GROQ_API_KEY not set!")
        sys.exit(1)
    
    if not webhook_host:
        logger.warning("RAILWAY_PUBLIC_DOMAIN not set - webhook mode requires public URL!")
        logger.warning("Falling back to polling mode (not recommended for production)")

    storage = RailwayStorage(get_env("RAILWAY_VOLUME_URL", "/data/storage"))

    # Telegram bot
    logger.info("[INIT] Creating RailwayBot...")
    bot = RailwayBot(storage)
    logger.info("[INIT] RailwayBot created")

    logger.info("[INIT] Building Telegram Application...")
    app = Application.builder().token(telegram_token).build()
    logger.info("[INIT] Application built")

    # [Architecture] Bind bot to app post-init to avoid circularity
    bot.bind_app(app)

    # Handlers
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("menu", bot.start))
    app.add_handler(CommandHandler("remind", bot.remind_command))
    app.add_handler(CommandHandler("routine", bot.routine_command))
    app.add_handler(CommandHandler("list", bot.list_command))
    app.add_handler(CommandHandler("clear", bot.clear_command))
    app.add_handler(CommandHandler("auth", bot.auth_command))
    app.add_handler(MessageHandler(filters.VOICE, bot.handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    app.add_handler(CallbackQueryHandler(bot.button_callback))

    # [UX] Register Bot Commands in Telegram Menu Button
    async def post_init(application: Application):
        from telegram import BotCommand
        commands = [
            BotCommand("menu", "Ana özellikleri gör"),
            BotCommand("list", "Bekleyen hatırlatıcıları listele"),
            BotCommand("auth", "Google Takvim'i bağla"),
            BotCommand("clear", "Temizlik yap")
        ]
        await application.bot.set_my_commands(commands)
        
        # Webhook'u ayarla veya temizle
        if webhook_host:
            webhook_url = f"https://{webhook_host}/telegram-webhook"
            await application.bot.set_webhook(url=webhook_url, allowed_updates=Update.ALL_TYPES)
            logger.info(f"[WEBHOOK] Set webhook to: {webhook_url}")
        else:
            await application.bot.delete_webhook(drop_pending_updates=True)
            logger.info("[POLLING] Webhook cleared, using polling mode")
        
        logger.info("[UX] Bot commands registered in Menu button")

    app.post_init = post_init

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

    # Günlük özet - her sabah 08:30 TSİ (05:30 UTC)
    from datetime import time as dt_time
    job_queue.run_daily(daily_digest_job, time=dt_time(hour=5, minute=30))

    logger.info("=" * 50)
    logger.info("🚀 MallibuSupportbot Starting...")
    logger.info(f"Storage: {get_env('RAILWAY_VOLUME_URL', '/data/storage')}")
    logger.info(f"Port: {port}")
    logger.info(f"Mode: {'WEBHOOK' if webhook_host else 'POLLING'}")
    if webhook_host:
        logger.info(f"Webhook URL: https://{webhook_host}/telegram-webhook")
    logger.info("AI: Groq Llama 3.3")
    logger.info("Reminders: Active")
    logger.info("=" * 50)

    if webhook_host:
        # ==================== WEBHOOK MODE ====================
        # Telegram webhook + Flask sync API birleşik sunucu
        
        @sync_app.route("/telegram-webhook", methods=["POST"])
        def telegram_webhook():
            """Telegram'dan gelen güncellemeleri işle"""
            try:
                update = Update.de_json(request.get_json(force=True), app.bot)
                asyncio.run(app.process_update(update))
                return "OK", 200
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return "Error", 500
        
        @sync_app.route("/health", methods=["GET"])
        def health():
            """Railway health check endpoint"""
            return jsonify({"status": "healthy", "mode": "webhook", "bot": "MallibuSupportbot"}), 200
        
        # Flask sunucusunu başlat (webhook + sync API)
        logger.info(f"[SERVER] Starting unified Flask server on port {port}")
        
        # Async initialization gerekiyor
        async def async_init():
            await app.initialize()
            await app.start()
            if app.post_init:
                await app.post_init(app)
        
        asyncio.run(async_init())
        
        # Flask sync API + Telegram webhook aynı sunucuda
        sync_app.run(host="0.0.0.0", port=port, use_reloader=False, threaded=True)
        
    else:
        # ==================== POLLING MODE (FALLBACK) ====================
        # Flask thread
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logger.info("Sync API thread started")
        
        # Polling başlat
        app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()

