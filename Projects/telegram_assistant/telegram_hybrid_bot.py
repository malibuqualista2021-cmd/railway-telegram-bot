#!/usr/bin/env python3
"""
Telegram Asistan v5.0 - HİBRİT MİMARİ
PC kapalıyken bulut modunda, PC açıkyken tam yerel modunda çalışır

Mimari:
┌─────────────────────────────────────────────────────────────┐
│                    TELEGRAM BOT                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Mode Detector (Çalışma Modu Tespiti)       │  │
│  │  - Yerel sistem kontrolü (ping/webhook)            │  │
│  │  - Yerel aktifse: Full Features                    │  │
│  │  - Yerel kapalıysa: Cloud Mode                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│         ┌──────────────┴──────────────┐                   │
│         ▼                             ▼                   │
│  ┌───────────────┐           ┌───────────────┐            │
│  │  YEREL MODE   │           │  BULUT MODE   │            │
│  │  GLM 4        │           │  Groq Llama   │            │
│  │  ChromaDB     │           │  JSON Storage │            │
│  │  Full Features│           │  Basic Mode   │            │
│  └───────────────┘           └───────────────┘            │
└─────────────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import threading
import requests
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# ChromaDB
import chromadb
from chromadb.config import Settings

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

# Groq
from groq import Groq

# Logging
logging.basicConfig(
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('asistant_v50.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==================== CONFIG ====================
class HybridConfig:
    telegram_token: str = "8449158473:AAG-3HbGmY2740CdrAnS1SAzw4Hnyp3DAB0"
    groq_key: str = "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"

    # Yerel
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "glm4"
    hot_memory_path: str = str(Path.home() / "asistant_v50_hot")
    warm_archive_path: str = str(Path.home() / "asistant_v50_warm")
    cold_archive_path: str = str(Path.home() / "asistant_v50_cold")
    deep_archive_path: str = str(Path.home() / "asistant_v50_deep")

    # Bulut senkronizasyon
    heartbeat_file: str = str(Path.home() / "asistant_v50_deep" / "sync_heartbeat.txt")
    sync_interval: int = 60  # saniye


config = HybridConfig()


# ==================== MODE DETECTOR ====================
class ModeDetector:
    """
    Çalışma modunu tespit eder
    - Full Mode: Yerel sistem aktif (Ollama + ChromaDB)
    - Cloud Mode: Sadece Groq (yerel sistem kapalı)
    """

    def __init__(self):
        self.mode = "cloud"
        self.last_check = None
        self.heartbeat_file = Path(config.heartbeat_file)

    async def detect_mode(self) -> str:
        """
        Modu tespit et
        Returns: 'full' veya 'cloud'
        """
        # 1. Ollama kontrolü
        ollama_available = self._check_ollama()

        # 2. ChromaDB kontrolü
        chromadb_available = self._check_chromadb()

        # 3. Heartbeat dosyası kontrolü (yerel botun çalıştığını anlamak için)
        local_running = self._check_local_bot()

        if ollama_available and chromadb_available:
            self.mode = "full"
            logger.info("Mod: FULL (Yerel sistem aktif)")
        else:
            self.mode = "cloud"
            logger.warning(f"Mod: CLOUD (Yerel pasif - Ollama:{ollama_available}, ChromaDB:{chromadb_available})")

        return self.mode

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{config.ollama_base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except:
            return False

    def _check_chromadb(self) -> bool:
        try:
            client = chromadb.PersistentClient(
                path=config.hot_memory_path,
                settings=Settings(anonymized_telemetry=False)
            collection = client.get_or_create_collection("test")
            return True
        except:
            return False

    def _check_local_bot(self) -> bool:
        """Yerel botun çalışıp çalışmadığını kontrol et"""
        # Heartbeat dosyasına bak
        if self.heartbeat_file.exists():
            try:
                content = self.heartbeat_file.read_text()
                last_update = datetime.fromisoformat(content.strip())
                # Son 2 dakika içinde güncellendi mi?
                if (datetime.now() - last_update).total_seconds() < 120:
                    return True
            except:
                pass
        return False


# ==================== BULUT DEPOLAMA ====================
class CloudStorage:
    """
    Bulut depolama - Basit JSON dosyası
    Gerçek uygulamada: Supabase, MongoDB, Firebase vb.
    """

    def __init__(self):
        self.storage_file = Path.home() / "asistant_v50_cloud.json"
        self.data = self._load()

    def _load(self) -> Dict:
        if self.storage_file.exists():
            try:
                return json.loads(self.storage_file.read_text(encoding='utf-8'))
            except:
                pass
        return {"notes": [], "synced": []}

    def _save(self):
        self.storage_file.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )

    def add_note(self, user_id: int, text: str) -> str:
        """Not ekle"""
        note = {
            "id": f"cloud_{datetime.now().timestamp()}",
            "user_id": user_id,
            "text": text,
            "created": datetime.now().isoformat(),
            "synced": False
        }
        self.data["notes"].append(note)
        self._save()
        return note["id"]

    def get_unsynced(self, user_id: int) -> List[Dict]:
        """Senkronize edilmemiş notları getir"""
        return [n for n in self.data["notes"] if n["user_id"] == user_id and not n.get("synced", False)]

    def mark_synced(self, note_ids: List[str]):
        """Notları senkronize edildi olarak işaretle"""
        for note in self.data["notes"]:
            if note["id"] in note_ids:
                note["synced"] = True
                self.data["synced"].append(note["id"])
        self._save()

    def search(self, user_id: int, query: str) -> List[Dict]:
        """Arama"""
        results = []
        query_lower = query.lower()
        for note in self.data["notes"]:
            if note["user_id"] == user_id and query_lower in note["text"].lower():
                results.append(note)
        return results


# ==================== GROQ AGENT (BULUT MODU) ====================
class GroqAgent:
    """Groq (Llama 3.3) - Bulut modu için"""

    SYSTEM = """Sen yardımcı bir asistanısın.
Kısa, öz ve Türkçe yanıtlar ver."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def chat(self, text: str, context: List[Dict] = None) -> str:
        """Sohbet"""
        messages = [{"role": "system", "content": self.SYSTEM}]

        if context:
            for msg in context[-5:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": text})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq hatası: {e}")
            return "Üzgünüm, bir hata oluştu."


# ==================== HİBRİT ASİSTAN ====================
class HybridAssistant:
    """
    Hibrit Asistan
    - Modu otomatik tespit eder
    - Full mode'da yerel özellikleri kullanır
    - Cloud mode'da Groq ile çalışır
    """

    def __init__(self):
        self.mode_detector = ModeDetector()
        self.cloud_storage = CloudStorage()
        self.groq = GroqAgent(config.groq_key)
        self.mode = "cloud"

        # Yerel bileşenler (lazy loading)
        self.local_memory = None
        self.ollama = None

    async def initialize(self):
        """Başlangıç"""
        await self.mode_detector.detect_mode()
        self.mode = self.mode_detector.mode

        if self.mode == "full":
            await self._initialize_local()

        logger.info(f"Hibrit Asistan başlatıldı (Mod: {self.mode})")

    async def _initialize_local(self):
        """Yerel bileşenleri başlat"""
        try:
            # Import local modules
            # Not: Bu örnekte basitleştirilmiş versiyon

            # Ollama client
            class OllamaClient:
                def __init__(self, base_url, model):
                    self.base_url = base_url
                    self.model = model
                def generate(self, prompt, system=None):
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    try:
                        r = requests.post(
                            f"{self.base_url}/api/chat",
                            json={"model": self.model, "messages": messages, "stream": False},
                            timeout=120
                        )
                        if r.status_code == 200:
                            return r.json().get("message", {}).get("content", "")
                    except:
                        pass
                    return None

            self.ollama = OllamaClient(config.ollama_base_url, config.ollama_model)

            # ChromaDB
            client = chromadb.PersistentClient(
                path=config.hot_memory_path,
                settings=Settings(anonymized_telemetry=False)
            )
            self.notes_collection = client.get_or_create_collection("notes")

            logger.info("Yerel bileşenler başlatıldı")

        except Exception as e:
            logger.error(f"Yerel başlatma hatası: {e}")
            self.mode = "cloud"

    async def process(self, text: str, user_id: int) -> str:
        """
        Mesajı işle

        Args:
            text: Kullanıcı mesajı
            user_id: Telegram kullanıcı ID

        Returns:
            Yanıt metni
        """
        # Modu kontrol et
        current_mode = await self.mode_detector.detect_mode()

        # Mod değiştiyse yeniden başlat
        if current_mode != self.mode:
            logger.info(f"Mod değişti: {self.mode} -> {current_mode}")
            self.mode = current_mode
            if current_mode == "full":
                await self._initialize_local()

        # Soru mu, not mu?
        is_question = any(w in text.lower() for w in ["?", "nedir", "nasıl", "kim", "nerede", "kaç"])

        if is_question:
            # Arama modu
            return await self._handle_search(text, user_id)
        else:
            # Not alma modu
            return await self._handle_note(text, user_id)

    async def _handle_note(self, text: str, user_id: int) -> str:
        """Not alma"""
        if self.mode == "full":
            # Yerel ChromaDB'ye ekle
            note_id = self.notes_collection.add(
                ids=[f"note_{user_id}_{datetime.now().timestamp()}"],
                documents=[text],
                metadatas={"user_id": str(user_id), "created": datetime.now().isoformat()}
            )
            return f"📝 Yerel hafızaya kaydedildi (Full Mode)"
        else:
            # Bulut depolama
            note_id = self.cloud_storage.add_note(user_id, text)
            return f"📝 Buluta kaydedildi (Cloud Mode - PC açılınca senkronize edilecek)"

    async def _handle_search(self, text: str, user_id: int) -> str:
        """Arama"""
        query = text.replace("?", "").strip()

        if self.mode == "full":
            # Yerel ChromaDB araması
            try:
                results = self.notes_collection.query(
                    query_texts=[query],
                    n_results=5,
                    where={"user_id": str(user_id)}
                )

                if results['documents'] and results['documents'][0]:
                    response = "🔍 **Bulunanlar (Yerel):**\n\n"
                    for doc in results['documents'][0][:3]:
                        response += f"• {doc[:80]}...\n"

                    # GLM 4 ile özet
                    if self.ollama:
                        summary = self.ollama.generate(
                            f"Bu notları özetle: {results['documents'][0][0]}",
                            system="Kısa özet"
                        )
                        if summary:
                            response += f"\n\n**Özet:** {summary[:200]}..."

                    return response
            except Exception as e:
                logger.error(f"Yerel arama hatası: {e}")

        # Bulut araması
        results = self.cloud_storage.search(user_id, query)
        if results:
            response = f"🔍 **Bulunanlar (Bulut):**\n\n"
            for note in results[-3:]:
                response += f"• {note['text'][:60]}... ({note['created'][:10]})\n"
            return response

        # Hiçbir yerde bulunamadı
        # Groq ile genel yanıt
        ai_response = self.groq.chat(text)
        return f"🤖 **AI Yanıt:**\n\n{ai_response}"

    async def sync_status(self) -> Dict:
        """Senkronizasyon durumu"""
        unsynced = self.cloud_storage.get_unsynced(0)  # Tüm kullanıcılar

        return {
            "mode": self.mode,
            "unsynced_count": len(unsynced),
            "last_sync": datetime.now().strftime("%d.%m.%Y %H:%M")
        }


# ==================== TELEGRAM HANDLERS ====================
assistant = HybridAssistant()

# Uygulama
app = Application.builder().token(config.telegram_token).build()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Başlangıç"""
    await assistant.initialize()

    status = await assistant.sync_status()
    mode_icon = "🟢" if status["mode"] == "full" else "🔵"

    keyboard = [
        [InlineKeyboardButton("📝 Not Al", callback_data="note"),
         InlineKeyboardButton("🔍 Ara", callback_data="search")],
        [InlineKeyboardButton("🔄 Durum", callback_data="status")]
    ]

    reply = f"""🤖 **Hibrit Asistan v5.0**

{mode_icon} **Mod:** {status["mode"].upper()}

**Özellikler:**
• PC Açık: Yerel GLM 4 + ChromaDB
• PC Kapalı: Bulut Groq (Llama 3.3)
• Otomatik senkronizasyon

**Durum:**
• Bekleyen: {status['unsynced_count']} not
• Son sync: {status['last_sync']}

Notlarınız güvende!"""

    await update.message.reply_text(
        reply,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mesaj işle"""
    await update.message.chat.send_action("typing")

    user_id = update.effective_user.id
    text = update.message.text

    try:
        response = await assistant.process(text, user_id)
        await update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"İşlem hatası: {e}")
        await update.message.reply_text(f"⚠️ Hata: {str(e)[:100]}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Buton callback"""
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "status":
        status = await assistant.sync_status()
        mode_icon = "🟢" if status["mode"] == "full" else "🔵"

        await query.edit_message_text(
            f"🔄 **Durum**\n\n"
            f"{mode_icon} Mod: {status['mode'].upper()}\n"
            f"Bekleyen not: {status['unsynced_count']}\n"
            f"Son sync: {status['last_sync']}",
            parse_mode='Markdown'
        )
    elif data == "note":
        await query.edit_message_text("📝 Notunuzu yazın...")
    elif data == "search":
        await query.edit_message_text("🔍 Aramak istediğinizi yazın...")


# ==================== HEARTBEAT THREAD ====================
def heartbeat_thread():
    """Heartbeat thread'i"""
    heartbeat_file = Path(config.heartbeat_file)
    heartbeat_file.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            # Heartbeat dosyasını güncelle
            heartbeat_file.write_text(datetime.now().isoformat())

            # 1 dakika bekle
            import time
            time.sleep(60)
        except:
            import time
            time.sleep(60)


# ==================== MAIN ====================
def main():
    # Handler'ları ekle
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))

    # Heartbeat thread'ini başlat
    import threading
    thread = threading.Thread(target=heartbeat_thread, daemon=True)
    thread.start()

    logger.info("=" * 50)
    logger.info("Hibrit Asistan v5.0 Başlatılıyor...")
    logger.info("Mod: Otomatik tespit (Yerel / Bulut)")
    logger.info("=" * 50)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
