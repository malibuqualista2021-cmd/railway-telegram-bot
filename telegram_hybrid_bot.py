#!/usr/bin/env python3
"""
Telegram Asistan - Railway Cloud Bot
PC kapalıyken Railway'de çalışır, notları depolar
PC açılınca yerel sistem senkronize olur

Environment Variables (Railway'de ayarlayın):
- TELEGRAM_TOKEN: Telegram bot token
- GROQ_API_KEY: Groq API key
- RAILWAY_VOLUME_URL: Persistent storage path (opsiyonel)
"""
import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from groq import Groq

# Logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==================== CONFIG ====================
class Config:
    """Environment variable based config"""

    # Required
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    groq_key: str = os.getenv("GROQ_API_KEY", "")

    # Optional - Railway storage
    # Railway volume: /data/storage
    storage_path: str = os.getenv("RAILWAY_VOLUME_URL", "/data/storage")

    # Yerel sistem kontrolü için
    # Bu Railway'de çalışmayacak ama senkronizasyon için gerekli
    local_webhook_url: str = os.getenv("LOCAL_WEBHOOK_URL", "")

    def validate(self) -> bool:
        """Validate config"""
        if not self.telegram_token:
            logger.error("TELEGRAM_TOKEN gerekli!")
            return False
        if not self.groq_key:
            logger.error("GROQ_API_KEY gerekli!")
            return False
        return True


config = Config()


# ==================== STORAGE ====================
class RailwayStorage:
    """
    Railway persistent storage
    Volume mount: /data/storage
    """

    def __init__(self, storage_path: str = "/data/storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.notes_file = self.storage_path / "notes.json"
        self.sync_file = self.storage_path / "sync_status.json"

        self.notes = self._load_notes()

    def _load_notes(self) -> List[Dict]:
        """Notları yükle"""
        if self.notes_file.exists():
            try:
                return json.loads(self.notes_file.read_text(encoding='utf-8'))
            except Exception as e:
                logger.error(f"Not yükleme hatası: {e}")
        return []

    def _save_notes(self):
        """Notları kaydet"""
        try:
            self.notes_file.write_text(
                json.dumps(self.notes, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Not kaydetme hatası: {e}")

    def add_note(self, user_id: int, text: str) -> str:
        """Not ekle"""
        note = {
            "id": f"railway_{user_id}_{datetime.now().timestamp()}",
            "user_id": user_id,
            "text": text,
            "created": datetime.now().isoformat(),
            "synced": False  # Yerel sistemle senkronize edilmedi
        }
        self.notes.append(note)
        self._save_notes()

        logger.info(f"Not eklendi: {note['id'][:20]}...")
        return note["id"]

    def get_notes(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Kullanıcı notlarını getir"""
        user_notes = [n for n in self.notes if n["user_id"] == user_id]
        return user_notes[-limit:]

    def get_pending(self, user_id: int) -> List[Dict]:
        """Senkronize edilmemiş notları getir"""
        return [n for n in self.notes if n["user_id"] == user_id and not n.get("synced", False)]

    def mark_synced(self, note_ids: List[str]):
        """Notları senkronize edildi olarak işaretle"""
        count = 0
        for note in self.notes:
            if note["id"] in note_ids and not note.get("synced", False):
                note["synced"] = True
                count += 1
        if count > 0:
            self._save_notes()
            logger.info(f"{count} not senkronize edildi olarak işaretlendi")

    def search(self, user_id: int, query: str) -> List[Dict]:
        """Arama"""
        query_lower = query.lower()
        results = []
        for note in self.notes:
            if note["user_id"] == user_id and query_lower in note["text"].lower():
                results.append(note)
        return results[-10:]  # Son 10 sonuç

    def get_stats(self) -> Dict:
        """İstatistikler"""
        total = len(self.notes)
        pending = sum(1 for n in self.notes if not n.get("synced", False))

        return {
            "total_notes": total,
            "pending_sync": pending,
            "storage_path": str(self.storage_path)
        }


# ==================== GROQ AGENT ====================
class GroqAgent:
    """Groq (Llama 3.3) - Bulut AI"""

    SYSTEM = """Sen yardımcı bir Türkçe asistanısın.
Kısa, öz ve dostça yanıtlar ver.
Emoji kullanabilirsin."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def chat(self, text: str, context: List[Dict] = None) -> Optional[str]:
        """Sohbet"""
        messages = [{"role": "system", "content": self.SYSTEM}]

        if context:
            for msg in context[-3:]:  # Son 3 mesaj
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
            return None


# ==================== RAILWAY BOT ====================
class RailwayBot:
    """Railway için Telegram botu"""

    def __init__(self):
        self.storage = RailwayStorage(config.storage_path)
        self.groq = GroqAgent(config.groq_key)

        # Kullanıcı bağlamı (kısa süreli memory)
        self.contexts: Dict[int, List[Dict]] = {}

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Başlangıç komutu"""
        user_id = update.effective_user.id
        stats = self.storage.get_stats()

        keyboard = [
            [
                InlineKeyboardButton("📝 Notlarım", callback_data=f"notes_{user_id}"),
                InlineKeyboardButton("🔍 Ara", callback_data=f"search_{user_id}")
            ],
            [
                InlineKeyboardButton("📊 Durum", callback_data=f"status_{user_id}"),
                InlineKeyboardButton("🔄 Bekleyen", callback_data=f"pending_{user_id}")
            ]
        ]

        reply = f"""🚂 **Railway Bot - 24/7 Aktif**

Merhaba {update.effective_user.first_name}!

**Özellikler:**
• Notlarınız Railway bulutta saklanır
• PC açılınca otomatik senkronize olur
• AI asistan (Llama 3.3) her zaman hazır

**Durum:**
📝 Toplam: {stats['total_notes']} not
⏳ Bekleyen: {stats['pending_sync']} not

**Kullanım:**
• Mesaj yazın → Not olarak kaydedilir
• Soru sorun → AI yanıt verir"""

        await update.message.reply_text(
            reply,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Mesaj işle"""
        user_id = update.effective_user.id
        text = update.message.text

        # Typing göster
        await update.message.chat.send_action("typing")

        # Soru mu, not mu?
        is_question = any(w in text.lower() for w in [
            "?", "nedir", "nasıl", "kim", "nerede", "kaç", "neden",
            "yapılır", "ederim", "eder misin", "bilir misin"
        ])

        if is_question:
            # Arama yap + AI yanıt
            await self._handle_question(update, user_id, text)
        else:
            # Not kaydet
            note_id = self.storage.add_note(user_id, text)

            # Kısa AI yanıt
            ai_response = self.groq.chat(f"Kullanıcı şunu not aldı: '{text}'. Kısa teyit mesajı ver.")
            response = ai_response or f"✅ Not kaydedildi"

            await update.message.reply_text(response)

    async def _handle_question(self, update: Update, user_id: int, query: str):
        """Soru işle"""
        # Önce notlarda ara
        results = self.storage.search(user_id, query)

        if results:
            # Sonuç bulundu
            reply = f"🔍 **Bulunanlar ({len(results)} adet):**\n\n"
            for note in results[-5:]:
                reply += f"• {note['text'][:80]}...\n"

            # AI ile özet
            await update.message.reply_text(reply, parse_mode='Markdown')
        else:
            # Sonuç yok, AI yanıt ver
            ai_response = self.groq.chat(query)
            if ai_response:
                await update.message.reply_text(f"🤖 **AI:**\n\n{ai_response}", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Üzgünüm, bir sorun oluştu.")

    async def show_notes(self, update: Update, user_id: int):
        """Notları göster"""
        notes = self.storage.get_notes(user_id)

        if not notes:
            await update.message.reply_text("📭 Henüz notunuz yok.")
            return

        reply = f"📝 **Notlarınız ({len(notes)} adet):**\n\n"
        for note in notes[-10:]:
            created = note['created'][:16].replace('T', ' ')
            reply += f"• {created}: {note['text'][:60]}...\n"

        await update.message.reply_text(reply, parse_mode='Markdown')

    async def show_pending(self, update: Update, user_id: int):
        """Bekleyen notları göster"""
        pending = self.storage.get_pending(user_id)

        if not pending:
            await update.message.reply_text("✅ Bekleyen not yok. Her şey senkronize!")
            return

        reply = f"⏳ **Senkronize bekleyen ({len(pending)} adet):**\n\n"
        for note in pending[-10:]:
            reply += f"• {note['created'][:10]}: {note['text'][:50]}...\n"

        reply += f"\n🔄 PC açılınca otomatik senkronize edilecek."
        await update.message.reply_text(reply, parse_mode='Markdown')

    async def show_status(self, update: Update):
        """Durum göster"""
        stats = self.storage.get_stats()

        reply = f"""📊 **Sistem Durumu**

🚂 Platform: Railway Cloud
📝 Toplam Not: {stats['total_notes']}
⏳ Bekleyen: {stats['pending_sync']}
💾 Depolama: {stats['storage_path']}

🤖 AI Model: Llama 3.3 70B (Groq)"""

        await update.message.reply_text(reply, parse_mode='Markdown')

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Buton callback"""
        query = update.callback_query
        await query.answer()

        data = query.data
        parts = data.split('_')
        action = parts[0]
        user_id = int(parts[1]) if len(parts) > 1 else 0

        if action == "notes":
            await self.show_notes(update, user_id)
        elif action == "search":
            await query.edit_message_text("🔍 Aramak istediğinizi yazın...")
        elif action == "status":
            await self.show_status(update)
        elif action == "pending":
            await self.show_pending(update, user_id)


# ==================== MAIN ====================
def main():
    """Bot başlat"""

    # Config kontrol
    if not config.validate():
        logger.error("Config hatası! Environment variables kontrol edin.")
        sys.exit(1)

    # Storage dizini kontrol
    logger.info(f"Storage path: {config.storage_path}")

    # Bot oluştur
    bot = RailwayBot()

    # Application
    app = Application.builder().token(config.telegram_token).build()

    # Handler'lar
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    app.add_handler(CallbackQueryHandler(bot.button_callback))

    # Başlat
    logger.info("=" * 50)
    logger.info("Railway Bot Başlatılıyor...")
    logger.info(f"Storage: {config.storage_path}")
    logger.info("AI: Groq Llama 3.3")
    logger.info("=" * 50)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
