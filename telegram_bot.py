#!/usr/bin/env python3
"""
Malibu Telegram Bot - Stable Version
Compatible with python-telegram-bot>=21.0
"""
import os
import sys
import asyncio
import logging
import signal
import threading
import time
from datetime import datetime, timedelta, timezone

os.environ['PYTHONUNBUFFERED'] = '1'

import httpx
from flask import Flask, jsonify
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, filters
)
from telegram.error import TelegramError, TimedOut, RetryAfter, Conflict, NetworkError

# ==================== LOGGING ====================
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
log = logging.getLogger("MalibuBot")
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("telegram").setLevel(logging.WARNING)

# ==================== CONFIG ====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
ADMIN_ID = os.getenv("ADMIN_ID", "")
SHEETS_WEBHOOK = os.getenv("SHEETS_WEBHOOK", "")
WEBSITE_URL = os.getenv("WEBSITE_URL", "https://harmonikprzmalibu.netlify.app")
PORT = int(os.getenv("PORT", "8080"))

PAYMENT_ADDRESS = "TKUvYuzdZvkq6ksgPxfDRsUQE4vYjnEcnL"
TRADINGVIEW, TXID = range(2)

PLANS = {
    "plan_monthly_30": {"name": "Aylık", "price": "$30", "days": 30},
    "plan_quarterly_79": {"name": "3 Aylık", "price": "$79", "days": 90},
    "plan_yearly_269": {"name": "Yıllık", "price": "$269", "days": 365},
    "trial": {"name": "7 Günlük Deneme", "price": "Ücretsiz", "days": 7}
}

REJECTION_REASONS = {
    "duplicate": "🔄 Mükerrer Deneme Kaydı",
    "invalid_txid": "💳 Geçersiz TXID / Ödeme",
    "pending": "⏳ Ödeme Beklemede / Onaylanmadı",
    "invalid_user": "👤 Geçersiz TradingView Adı",
    "other": "❓ Diğer Sebep"
}

# ==================== STATE ====================
START_TIME = datetime.now(timezone.utc)
BOT_STATUS = {"running": False, "errors": 0, "restarts": 0}
pending_requests = {}
last_user_message = {}
SHUTDOWN = threading.Event()

# ==================== FLASK ====================
app = Flask(__name__)

@app.route("/")
@app.route("/health")
def health():
    uptime = int((datetime.now(timezone.utc) - START_TIME).total_seconds())
    return jsonify({"status": "ok", "uptime": uptime, "bot": BOT_STATUS}), 200

@app.route("/ping")
def ping():
    return "pong", 200

# ==================== SHEETS ====================
async def save_to_sheets(data: dict) -> bool:
    if not SHEETS_WEBHOOK:
        return False
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.post(SHEETS_WEBHOOK, json=data)
            return response.status_code == 200
    except Exception as e:
        log.error(f"Sheets error: {e}")
    return False

async def get_expired_users() -> list:
    if not SHEETS_WEBHOOK:
        return []
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(f"{SHEETS_WEBHOOK}?action=expired")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        log.error(f"Get expired error: {e}")
    return []

def calculate_end_date(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%d.%m.%Y")

# ==================== HANDLERS ====================
async def cmd_start(update: Update, context):
    user = update.effective_user
    args = context.args or []
    plan_key = args[0] if args else None

    if plan_key and plan_key in PLANS:
        plan = PLANS[plan_key]
        context.user_data['plan_key'] = plan_key
        context.user_data['plan'] = plan
        await update.message.reply_text(
            f"🌴 *Malibu PRZ Suite*\n\n✅ *{plan['name']}* seçildi!\n\n📝 TradingView kullanıcı adınızı yazın:",
            parse_mode="Markdown"
        )
        return TRADINGVIEW
    else:
        keyboard = [
            [InlineKeyboardButton("💳 Aylık - $30", callback_data="plan_monthly_30")],
            [InlineKeyboardButton("⭐ 3 Aylık - $79", callback_data="plan_quarterly_79")],
            [InlineKeyboardButton("👑 Yıllık - $269", callback_data="plan_yearly_269")],
            [InlineKeyboardButton("🆓 7 Gün Deneme", callback_data="trial")]
        ]
        await update.message.reply_text(
            f"Merhaba {user.first_name}! 👋\n\n🌴 *Malibu PRZ Suite*\n\n📊 Bir plan seçin:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        return ConversationHandler.END

async def plan_selected(update: Update, context):
    query = update.callback_query
    await query.answer()
    plan_key = query.data
    if plan_key not in PLANS:
        return ConversationHandler.END
    plan = PLANS[plan_key]
    context.user_data['plan_key'] = plan_key
    context.user_data['plan'] = plan
    await query.message.reply_text(
        f"✅ *{plan['name']}* seçildi!\n\n📝 TradingView kullanıcı adınızı yazın:",
        parse_mode="Markdown"
    )
    return TRADINGVIEW

async def receive_tradingview(update: Update, context):
    user = update.effective_user
    tv_username = update.message.text.strip()
    context.user_data['tradingview'] = tv_username
    plan_key = context.user_data.get('plan_key', '')

    if plan_key == "trial":
        await save_request(user, context, txid="DENEME")
        await update.message.reply_text(
            f"✅ *Deneme talebiniz alındı!*\n\n📺 TradingView: `{tv_username}`\n⏱️ 7 gün\n\nTeşekkürler! 🙏",
            parse_mode="Markdown"
        )
        return ConversationHandler.END
    else:
        plan = context.user_data.get('plan', {})
        await update.message.reply_text(
            f"📺 TradingView: `{tv_username}`\n\n💰 *Ödeme:*\n`{PAYMENT_ADDRESS}`\n\nTutar: *{plan.get('price', '?')}*\n\n⚠️ TXID gönderin:",
            parse_mode="Markdown"
        )
        return TXID

async def receive_txid(update: Update, context):
    user = update.effective_user
    txid = update.message.text.strip()
    context.user_data['txid'] = txid
    await save_request(user, context, txid=txid)
    plan = context.user_data.get('plan', {})
    await update.message.reply_text(
        f"✅ *Ödeme talebiniz alındı!*\n\n📋 TXID: `{txid}`\n📊 Plan: {plan.get('name', '?')}\n\nTeşekkürler! 🙏",
        parse_mode="Markdown"
    )
    return ConversationHandler.END

async def save_request(user, context, txid: str):
    plan = context.user_data.get('plan', {})
    tv_username = context.user_data.get('tradingview', '')
    now = datetime.now(timezone.utc)
    data = {
        'tarih': now.strftime("%d.%m.%Y %H:%M"),
        'telegram_id': str(user.id),
        'telegram_username': user.username or "Yok",
        'telegram_name': user.first_name or "",
        'txid': txid,
        'plan': plan.get('name', ''),
        'tradingview': tv_username,
        'baslangic_tarihi': now.strftime("%d.%m.%Y"),
        'bitis_tarihi': calculate_end_date(plan.get('days', 30)),
        'durum': 'Beklemede 🟡'
    }
    await save_to_sheets(data)

    if ADMIN_ID:
        try:
            keyboard = [[
                InlineKeyboardButton("✅ Onayla", callback_data=f"approve_{user.id}"),
                InlineKeyboardButton("❌ Reddet", callback_data=f"reject_{user.id}")
            ]]
            pending_requests[str(user.id)] = data
            is_trial = "🆓 DENEME" if txid == "DENEME" else "💰 ÖDEME"
            await context.bot.send_message(
                chat_id=int(ADMIN_ID),
                text=f"{is_trial} *Yeni Talep*\n\n👤 {user.first_name}\n📺 `{tv_username}`\n📋 TXID: `{txid}`",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            log.error(f"Admin bildirim hatası: {e}")

async def admin_callback(update: Update, context):
    query = update.callback_query
    await query.answer()
    if str(query.from_user.id) != str(ADMIN_ID):
        return
    data_parts = query.data.split("_")
    action = data_parts[0]

    if action == "approve":
        user_id = data_parts[1]
        user_data = pending_requests.pop(user_id, {})
        await query.message.edit_text(f"✅ *Onaylandı*\n📺 {user_data.get('tradingview', '?')}", parse_mode="Markdown")
        try:
            await context.bot.send_message(chat_id=int(user_id), text="🎉 *Erişiminiz aktifleştirildi!*", parse_mode="Markdown")
        except:
            pass
    elif action == "reject":
        user_id = data_parts[1]
        keyboard = [[InlineKeyboardButton(v, callback_data=f"rejectreason_{user_id}_{k}")] for k, v in REJECTION_REASONS.items()]
        await query.message.reply_text("❌ *Red Sebebi Seçin*", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    elif action == "rejectreason":
        user_id = data_parts[1]
        reason_key = data_parts[2]
        pending_requests.pop(user_id, {})
        reason_text = REJECTION_REASONS.get(reason_key, "Belirtilmedi")
        await query.message.edit_text(f"❌ *Reddedildi*\nSebep: {reason_text}", parse_mode="Markdown")
        try:
            await context.bot.send_message(chat_id=int(user_id), text=f"❌ *Talebiniz Reddedildi*\nSebep: {reason_text}", parse_mode="Markdown")
        except:
            pass

async def cmd_cancel(update: Update, context):
    await update.message.reply_text("İşlem iptal edildi. /start yazın.")
    return ConversationHandler.END

async def cmd_help(update: Update, context):
    await update.message.reply_text("📚 /start - Başla\n/help - Yardım")

async def cmd_status(update: Update, context):
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    uptime = int((datetime.now(timezone.utc) - START_TIME).total_seconds())
    await update.message.reply_text(f"📊 *Bot Durumu*\n✅ Çalışıyor\n⏱️ {uptime//3600}s {(uptime%3600)//60}dk", parse_mode="Markdown")

async def handle_user_message(update: Update, context):
    user = update.effective_user
    if str(user.id) == str(ADMIN_ID):
        return
    last_user_message[str(ADMIN_ID)] = {'user_id': str(user.id), 'user_name': user.first_name or "Kullanıcı"}
    if ADMIN_ID:
        try:
            await context.bot.send_message(chat_id=int(ADMIN_ID), text=f"💬 *Mesaj*\n👤 {user.first_name}\n📝 {update.message.text}", parse_mode="Markdown")
        except:
            pass
    await update.message.reply_text("📨 Mesajınız iletildi!")

# ==================== BOT ENGINE ====================
async def run_bot():
    log.info("Bot başlatılıyor...")
    application = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start), CallbackQueryHandler(plan_selected, pattern="^(plan_|trial)")],
        states={
            TRADINGVIEW: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_tradingview)],
            TXID: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_txid)]
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_message=False, per_chat=True, per_user=True
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CallbackQueryHandler(admin_callback, pattern="^(approve_|reject|rejectreason)"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))

    await application.initialize()

    log.info("🔄 Telegram oturumu temizleniyor...")
    for attempt in range(5):
        try:
            await application.bot.delete_webhook(drop_pending_updates=True)
            await asyncio.sleep(1)
            await application.bot.get_updates(offset=-1, timeout=1)
            log.info("✅ Oturum temizlendi")
            break
        except Conflict:
            log.warning(f"⚠️ Conflict ({attempt+1}/5)")
            await asyncio.sleep(5 * (attempt + 1))
        except Exception as e:
            log.warning(f"⚠️ Hata: {e}")
            await asyncio.sleep(2)

    await application.start()
    BOT_STATUS["running"] = True
    log.info("✅ Bot başlatıldı - polling modunda")

    offset = None
    while not SHUTDOWN.is_set():
        try:
            updates = await application.bot.get_updates(offset=offset, timeout=30, allowed_updates=Update.ALL_TYPES)
            for upd in updates:
                offset = upd.update_id + 1
                await application.process_update(upd)
        except TimedOut:
            continue
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 1)
        except Conflict:
            log.error("🔴 CONFLICT - Başka bot çalışıyor!")
            BOT_STATUS["running"] = False
            SHUTDOWN.set()
            break
        except (NetworkError, TelegramError) as e:
            log.warning(f"⚠️ Ağ hatası: {e}")
            await asyncio.sleep(5)
        except Exception as e:
            BOT_STATUS["errors"] += 1
            log.error(f"❌ Hata: {e}")
            await asyncio.sleep(5)

    await application.stop()
    await application.shutdown()

def bot_thread():
    while not SHUTDOWN.is_set():
        BOT_STATUS["restarts"] += 1
        log.info(f"🚀 Bot başlatılıyor (#{BOT_STATUS['restarts']})")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_bot())
        except Exception as e:
            log.error(f"Bot çöktü: {e}")
            BOT_STATUS["running"] = False
        finally:
            loop.close()
        if not SHUTDOWN.is_set():
            time.sleep(3)

def signal_handler(signum, frame):
    log.info("⚠️ Kapatma sinyali...")
    SHUTDOWN.set()
    time.sleep(2)
    sys.exit(0)

def main():
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if not BOT_TOKEN:
        log.error("❌ BOT_TOKEN bulunamadı!")
        app.run(host="0.0.0.0", port=PORT)
        return

    log.info("=" * 50)
    log.info("🌴 Malibu Telegram Bot")
    log.info(f"🔌 Port: {PORT}")
    log.info("=" * 50)

    threading.Thread(target=bot_thread, daemon=False).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
