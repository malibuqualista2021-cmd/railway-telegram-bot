#!/usr/bin/env python3
"""
Telegram Asistan - Railway Cloud Bot + Sync Bridge
PC kapalıyken Railway'de çalışır, notları depolar
Sync API ile yerel PC ile senkronize olur

Environment Variables:
- TELEGRAM_TOKEN: Telegram bot token
- GROQ_API_KEY: Groq API key
- SYNC_TOKEN: Senkronizasyon için güvenlik token'ı
- RAILWAY_VOLUME_URL: Persistent storage path
"""
import os
import sys
import json
import logging
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
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


# ==================== CONFIG ====================
class Config:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    groq_key: str = os.getenv("GROQ_API_KEY", "")
    sync_token: str = os.getenv("SYNC_TOKEN", "default-sync-token-change-me")
    storage_path: str = os.getenv("RAILWAY_VOLUME_URL", "/data/storage")
    port: int = int(os.getenv("PORT", "5000"))

    def validate(self) -> bool:
        if not self.telegram_token or not self.groq_key:
            return False
        return True


config = Config()


# ==================== STORAGE ====================
class RailwayStorage:
    """Railway persistent storage"""

    def __init__(self, storage_path: str = "/data/storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.notes_file = self.storage_path / "notes.json"
        self.sync_file = self.storage_path / "sync_log.json"

        self.notes = self._load_notes()
        self.sync_log = self._load_sync_log()
        self.lock = threading.Lock()

    def _load_notes(self) -> List[Dict]:
        if self.notes_file.exists():
            try:
                return json.loads(self.notes_file.read_text(encoding='utf-8'))
            except Exception as e:
                logger.error(f"Not yükleme hatası: {e}")
        return []

    def _save_notes(self):
        try:
            with self.lock:
                self.notes_file.write_text(
                    json.dumps(self.notes, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8'
                )
        except Exception as e:
            logger.error(f"Not kaydetme hatası: {e}")

    def _load_sync_log(self) -> Dict:
        if self.sync_file.exists():
            try:
                return json.loads(self.sync_file.read_text(encoding='utf-8'))
            except:
                pass
        return {"last_sync": None, "synced_notes": []}

    def _save_sync_log(self):
        try:
            self.sync_file.write_text(
                json.dumps(self.sync_log, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Sync log kaydetme hatası: {e}")

    def add_note(self, user_id: int, text: str, source: str = "railway") -> str:
        with self.lock:
            note = {
                "id": f"{source}_{user_id}_{datetime.now().timestamp()}",
                "user_id": user_id,
                "text": text,
                "created": datetime.now().isoformat(),
                "source": source,
                "synced_to_local": False
            }
            self.notes.append(note)
            self._save_notes()
            logger.info(f"Not eklendi: {note['id'][:20]}... (kaynak: {source})")
            return note["id"]

    def get_notes(self, user_id: int, limit: int = 50) -> List[Dict]:
        user_notes = [n for n in self.notes if n["user_id"] == user_id]
        return user_notes[-limit:]

    def get_pending_sync(self) -> List[Dict]:
        """Yerel'e senkronize edilmemiş notlar"""
        return [n for n in self.notes if not n.get("synced_to_local", False)]

    def mark_synced_to_local(self, note_ids: List[str]):
        with self.lock:
            count = 0
        for note in self.notes:
            if note["id"] in note_ids and not note.get("synced_to_local", False):
                note["synced_to_local"] = True
                count += 1
        if count > 0:
            self._save_notes()
            self.sync_log["last_sync"] = datetime.now().isoformat()
            self._save_sync_log()
            logger.info(f"{count} not yerel'e senkronize edildi olarak işaretlendi")

    def add_from_local(self, notes: List[Dict]) -> int:
        """Yerelden gelen notları ekle"""
        with self.lock:
            added = 0
            for note in notes:
                if not any(n.get("id") == note.get("id") for n in self.notes):
                    note["synced_from"] = "local"
                    note["synced_to_local"] = True  # Yerel'den geldi, zaten orada
                    self.notes.append(note)
                    added += 1
            if added > 0:
                self._save_notes()
                logger.info(f"{added} not yerelden senkronize edildi")
            return added

    def search(self, user_id: int, query: str) -> List[Dict]:
        query_lower = query.lower()
        results = []
        for note in self.notes:
            if note["user_id"] == user_id and query_lower in note["text"].lower():
                results.append(note)
        return results[-10:]

    def get_stats(self) -> Dict:
        total = len(self.notes)
        pending = len([n for n in self.notes if not n.get("synced_to_local", False)])

        return {
            "total_notes": total,
            "pending_sync": pending,
            "last_sync": self.sync_log.get("last_sync"),
            "storage_path": str(self.storage_path)
        }


# Global storage instance
storage = None


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
        messages = [{"role": "system", "content": self.SYSTEM}]

        if context:
            for msg in context[-3:]:
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


# ==================== SYNC API (Flask) ====================
sync_app = Flask(__name__)
CORS(sync_app)


@sync_app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "railway-sync-bridge",
        "timestamp": datetime.now().isoformat()
    })


@sync_app.route("/sync/from-local", methods=["POST"])
def from_local():
    """Yerel PC'den not al"""
    token = request.headers.get("X-Sync-Token")
    if token != config.sync_token:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.json
        notes = data.get("notes", [])

        added = storage.add_from_local(notes)

        return jsonify({
            "status": "ok",
            "added": added,
            "total_notes": len(storage.notes)
        })
    except Exception as e:
        logger.error(f"Sync from local error: {e}")
        return jsonify({"error": str(e)}), 500


@sync_app.route("/sync/to-local", methods=["GET"])
def to_local():
    """Yerel PC'ye not ver"""
    token = request.headers.get("X-Sync-Token")
    if token != config.sync_token:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = request.args.get("user_id", type=int)

        # Bekleyen notları getir
        pending = storage.get_pending_sync()
        if user_id:
            pending = [n for n in pending if n.get("user_id") == user_id]

        return jsonify({
            "status": "ok",
            "notes": pending,
            "count": len(pending)
        })
    except Exception as e:
        logger.error(f"Sync to local error: {e}")
        return jsonify({"error": str(e)}), 500


@sync_app.route("/sync/mark-local-synced", methods=["POST"])
def mark_local_synced():
    """Notları senkronize edildi olarak işaretle"""
    token = request.headers.get("X-Sync-Token")
    if token != config.sync_token:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.json
        note_ids = data.get("note_ids", [])

        storage.mark_synced_to_local(note_ids)

        return jsonify({
            "status": "ok",
            "marked": len(note_ids)
        })
    except Exception as e:
        logger.error(f"Mark synced error: {e}")
        return jsonify({"error": str(e)}), 500


@sync_app.route("/sync/all", methods=["GET"])
def get_all():
    """Tüm notları getir"""
    token = request.headers.get("X-Sync-Token")
    if token != config.sync_token:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = request.args.get("user_id", type=int)

    notes = storage.notes.copy()
    if user_id:
        notes = [n for n in notes if n.get("user_id") == user_id]

    return jsonify({
        "status": "ok",
        "notes": notes,
        "count": len(notes)
    })


def run_flask():
    """Flask API'yi ayrı thread'de çalıştır"""
    logger.info(f"Sync API başlatılıyor (port {config.port})")
    try:
        sync_app.run(host="0.0.0.0", port=config.port, use_reloader=False, threaded=True)
        logger.info("Sync API başarıyla başlatıldı")
    except Exception as e:
        logger.error(f"Sync API başlatma hatası: {e}")


# ==================== TELEGRAM BOT ====================
class RailwayBot:
    """Railway Telegram Bot"""

    def __init__(self):
        self.groq = GroqAgent(config.groq_key)
        self.contexts: Dict[int, List[Dict]] = {}

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        stats = storage.get_stats()

        keyboard = [
            [InlineKeyboardButton("📝 Notlarım", callback_data=f"notes_{user_id}"),
             InlineKeyboardButton("🔍 Ara", callback_data=f"search_{user_id}")],
            [InlineKeyboardButton("📊 Durum", callback_data=f"status_{user_id}"),
             InlineKeyboardButton("🔄 Sync", callback_data=f"sync_{user_id}")]
        ]

        reply = f"""🚂 **Railway Bot - 24/7 Aktif**

Merhaba {update.effective_user.first_name}!

**Özellikler:**
• Notlar Railway bulutta saklanır
• PC açılınca otomatik senkronize olur
• AI asistan (Llama 3.3) her zaman hazır

**Durum:**
📝 Toplam: {stats['total_notes']} not
⏳ Bekleyen sync: {stats['pending_sync']} not
🔄 Son sync: {stats['last_sync'] or 'Henüz'}"""

        await update.message.reply_text(
            reply,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        text = update.message.text

        await update.message.chat.send_action("typing")

        is_question = any(w in text.lower() for w in [
            "?", "nedir", "nasıl", "kim", "nerede", "kaç", "neden"
        ])

        if is_question:
            await self._handle_question(update, user_id, text)
        else:
            storage.add_note(user_id, text, source="railway")
            ai_response = self.groq.chat(f"Kullanıcı not aldı: '{text}'. Kısa teyit.")
            response = ai_response or "✅ Not kaydedildi (Railway)"
            await update.message.reply_text(response)

    async def _handle_question(self, update: Update, user_id: int, query: str):
        results = storage.search(user_id, query)

        if results:
            reply = f"🔍 **Bulunanlar ({len(results)}):**\n\n"
            for note in results[-5:]:
                reply += f"• {note['text'][:80]}...\n"
            await update.message.reply_text(reply, parse_mode='Markdown')
        else:
            ai_response = self.groq.chat(query)
            if ai_response:
                await update.message.reply_text(f"🤖 **AI:**\n\n{ai_response}", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Bir sorun oluştu.")

    async def show_notes(self, update: Update, user_id: int):
        notes = storage.get_notes(user_id)
        if not notes:
            await update.message.reply_text("📭 Henüz not yok.")
            return

        reply = f"📝 **Notlarınız ({len(notes)}):**\n\n"
        for note in notes[-10:]:
            created = note['created'][:16].replace('T', ' ')
            source = note.get('source', 'railway')
            reply += f"• [{source}] {created}: {note['text'][:60]}...\n"

        await update.message.reply_text(reply, parse_mode='Markdown')

    async def show_status(self, update: Update, user_id: int):
        stats = storage.get_stats()
        reply = f"""📊 **Durum**

🚂 Platform: Railway Cloud
📝 Toplam: {stats['total_notes']} not
⏳ Sync bekleyen: {stats['pending_sync']}
🔄 Son sync: {stats['last_sync'] or 'Henüz'}"""
        await update.message.reply_text(reply, parse_mode='Markdown')

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            await self.show_status(update, user_id)
        elif action == "sync":
            stats = storage.get_stats()
            reply = f"🔄 **Senkronizasyon**\n\nBekleyen: {stats['pending_sync']} not\nSon sync: {stats['last_sync'] or 'Henüz'}"
            await query.edit_message_text(reply, parse_mode='Markdown')


# ==================== MAIN ====================
def main():
    global storage

    if not config.validate():
        logger.error("Config hatası!")
        sys.exit(1)

    storage = RailwayStorage(config.storage_path)

    # Flask API'yi ayrı thread'de başlat
    flask_thread = threading.Thread(target=run_flask, daemon=False)
    flask_thread.start()
    logger.info("Sync API thread başlatıldı")
    # Flask'ın başlaması için bekle
    import time
    time.sleep(2)

    # Telegram bot
    bot = RailwayBot()
    app = Application.builder().token(config.telegram_token).build()

    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    app.add_handler(CallbackQueryHandler(bot.button_callback))

    logger.info("=" * 50)
    logger.info("Railway Bot + Sync Bridge Başlatılıyor...")
    logger.info(f"Storage: {config.storage_path}")
    logger.info(f"Sync API: Port {config.port}")
    logger.info("AI: Groq Llama 3.3")
    logger.info("=" * 50)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
