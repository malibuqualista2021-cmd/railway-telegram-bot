#!/usr/bin/env python3
"""
Cloud Bot - PC Kapalıyken Çalışan Bot
Groq (Llama 3.3) üzerinden basit işlevler sağlar

Özellikler:
- Not alma ve bulut depolama
- Basit sorgulama
- PC açıldığında senkronizasyon için bekler
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Logging
logging.basicConfig(
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Yapılandırma
TELEGRAM_TOKEN = os.getenv("CLOUD_TOKEN", "8449158473:AAG-3HbGmY2740CdrAnS1SAzw4Hnyp3DAB0")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Cloud depolama (JSON bin olarak)
# Gerçek uygulamada: S3, MongoDB, Firebase, Supabase vb.
# Bu örnekte basit JSON dosyası (VPS'te)

CLOUD_STORAGE_PATH = Path("/tmp/cloud_bot_storage")
CLOUD_STORAGE_PATH.mkdir(exist_ok=True)

NOTES_FILE = CLOUD_STORAGE_PATH / "pending_notes.json"
SYNC_LOCK_FILE = CLOUD_STORAGE_PATH / "sync_lock.txt"


class CloudStorage:
    """Bulut depolama yöneticisi"""

    def __init__(self):
        self.notes_file = NOTES_FILE
        self.notes = self._load_notes()

    def _load_notes(self) -> List[Dict]:
        if self.notes_file.exists():
            try:
                return json.loads(self.notes_file.read_text(encoding='utf-8'))
            except:
                return []
        return []

    def _save_notes(self):
        self.notes_file.write_text(
            json.dumps(self.notes, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )

    def add_note(self, user_id: int, text: str) -> str:
        """Not ekle"""
        note = {
            "id": f"cloud_{user_id}_{datetime.now().timestamp()}",
            "user_id": user_id,
            "text": text,
            "created": datetime.now().isoformat(),
            "synced": False  # Yerel sistemle senkronize edilmedi
        }
        self.notes.append(note)
        self._save_notes()
        return note["id"]

    def get_pending_notes(self, user_id: int) -> List[Dict]:
        """Senkronize edilmemiş notları getir"""
        return [n for n in self.notes if n["user_id"] == user_id and not n.get("synced", False)]

    def mark_synced(self, note_ids: List[str]):
        """Notları senkronize edildi olarak işaretle"""
        for note in self.notes:
            if note["id"] in note_ids:
                note["synced"] = True
        self._save_notes()

    def search_notes(self, user_id: int, query: str) -> List[Dict]:
        """Notlarda arama"""
        query_lower = query.lower()
        results = []
        for note in self.notes:
            if note["user_id"] == user_id and query_lower in note["text"].lower():
                results.append(note)
        return results

    def is_local_active(self) -> bool:
        """Yerel sistem aktif mi kontrol et"""
        # Gerçek uygulamada: Yerel sistem'e ping atar
        # Bu örnekte: lock dosyasına bakar
        if SYNC_LOCK_FILE.exists():
            try:
                content = SYNC_LOCK_FILE.read_text()
                # Son 5 dakika içinde güncellendi mi?
                last_update = datetime.fromisoformat(content.strip())
                if (datetime.now() - last_update).total_seconds() < 300:
                    return True
            except:
                pass
        return False


class GroqClient:
    """Groq (Llama 3.3) istemcisi"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.3-70b-versatile"

    def chat(self, messages: List[Dict]) -> str:
        """Sohbet"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq hatası: {e}")

        return "Üzgünüm, bir hata oluştu."


class CloudBot:
    """Cloud bot - PC kapalıyken çalışan bot"""

    def __init__(self):
        self.storage = CloudStorage()
        self.groq = GroqClient(GROQ_API_KEY)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Başlangıç"""
        user_id = update.effective_user.id
        local_active = self.storage.is_local_active()

        status = "🟢 Yerel sistem AKTIF" if local_active else "🔵 Bulut modu (PC kapalı)"

        keyboard = [
            [
                {"text": "📝 Not Al", "callback_data": "note"},
                {"text": "🔍 Ara", "callback_data": "search"}
            ],
            [
                {"text": "📋 Bekleyenler", "callback_data": "pending"},
                {"text": "🔄 Senkronizasyon", "callback_data": "sync"}
            ]
        ]

        reply_text = f"""🤖 **Cloud Bot v1.0**

{status}

**Özellikler:**
• Notlarınız bulutta saklanır
• PC açıldığında otomatik senkronize olur
• 7/24 erişim

**Komutlar:**
• Mesaj yazın → Not olarak kaydedilir
• /pending → Bekleyen notlar
• /sync → Senkronizasyon durumunu"""

        await update.message.reply_text(
            reply_text,
            reply_markup={"inline_keyboard": keyboard},
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Mesaj işle"""
        user_id = update.effective_user.id
        text = update.message.text

        local_active = self.storage.is_local_active()

        if local_active:
            # Yerel sistem aktif, bildirim gönder
            await update.message.reply_text(
                "✅ Yerel sistem aktif - bu not otomatik senkronize edilecek"
            )

        # Notu kaydet
        note_id = self.storage.add_note(user_id, text)

        # Basit yanıt
        if "?" in text or "nedir" in text.lower() or "nasıl" in text.lower():
            # Soru varsa, Groq ile yanıt ver
            messages = [
                {"role": "system", "content": "Sen yardımcı bir asistansın. Kısa ve öz yanıtlar."},
                {"role": "user", "content": text}
            ]
            response = self.groq.chat(messages)
            await update.message.reply_text(response)
        else:
            # Normal not
            await update.message.reply_text(
                f"📝 Not kaydedildi (ID: {note_id[-8:]})\n"
                f"Durum: {'Yerel sisteme iletilecek' if not local_active else 'Yerel sistem aktif'}"
            )

    async def pending_notes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Bekleyen notlar"""
        user_id = update.effective_user.id
        pending = self.storage.get_pending_notes(user_id)

        if not pending:
            await update.message.reply_text("📭 Bekleyen not yok")
            return

        reply = f"📋 **Bekleyen Notlar ({len(pending)} adet)**\n\n"
        for note in pending[-10:]:  # Son 10
            reply += f"• {note['created'][:16]}: {note['text'][:50]}...\n"

        if len(pending) > 10:
            reply += f"\n... ve {len(pending) - 10} not daha"

        await update.message.reply_text(reply, parse_mode='Markdown')

    async def sync_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Senkronizasyon durumu"""
        user_id = update.effective_user.id
        pending = self.storage.get_pending_notes(user_id)
        local_active = self.storage.is_local_active()

        if local_active:
            status = "🟢 Yerel sistem AKTIF"
            detail = f"Senkronize edilmemiş: {len(pending)} not"
        else:
            status = "🔴 Yerel sistem KAPALI"
            detail = f"Bekleyen: {len(pending)} not (PC açılınca aktarılacak)"

        await update.message.reply_text(
            f"🔄 **Senkronizasyon Durumu**\n\n{status}\n{detail}",
            parse_mode='Markdown'
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Buton callback"""
        query = update.callback_query
        await query.answer()

        data = query.data
        user_id = query.from_user.id

        if data == "note":
            await query.edit_message_text("📝 Notunuzu yazın...")
        elif data == "search":
            await query.edit_message_text("🔍 Aramak istediğiniz kelimeyi yazın...")
        elif data == "pending":
            await self.pending_notes(update, context)
        elif data == "sync":
            await self.sync_status(update, context)


# ==================== MAIN ====================
def main():
    """Bot başlat"""
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    bot = CloudBot()

    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("pending", bot.pending_notes))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    app.add_handler(CallbackQueryHandler(bot.button_callback))

    logger.info("Cloud Bot başlatılıyor...")
    logger.info("PC kapalıyken çalışır, notları bulutta saklar")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
