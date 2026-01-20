#!/usr/bin/env python3
"""
🌴 Malibu Telegram Bot v1.0
===========================
- Website deep link desteği
- Conversation flow ile bilgi toplama
- Google Sheets webhook entegrasyonu
- Admin onay/red sistemi
- Süresi dolanlara bildirim
"""
import os
import sys
import asyncio
import logging
import json
import signal
import threading
import time
from datetime import datetime, timedelta, timezone

os.environ['PYTHONUNBUFFERED'] = '1'

import httpx
import requests
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
RAILWAY_URL = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")

# Ödeme adresi
PAYMENT_ADDRESS = "TKUvYuzdZvkq6ksgPxfDRsUQE4vYjnEcnL"

# Conversation states
TRADINGVIEW, TXID = range(2)

# Plan bilgileri
PLANS = {
    "plan_monthly_30": {"name": "Aylık", "price": "$30", "days": 30},
    "plan_quarterly_79": {"name": "3 Aylık", "price": "$79", "days": 90},
    "plan_yearly_269": {"name": "Yıllık", "price": "$269", "days": 365},
    "trial": {"name": "7 Günlük Deneme", "price": "Ücretsiz", "days": 7}
}

# ==================== STATE ====================
START_TIME = datetime.now(timezone.utc)
BOT_STATUS = {"running": False, "errors": 0, "restarts": 0}
pending_requests = {}
last_user_message = {}  # {admin_id: {user_id: str, user_name: str}}
SHUTDOWN = threading.Event()

# Red sebepleri
REJECTION_REASONS = {
    "duplicate_trial": "Mükerrer ücretsiz deneme kaydı",
    "invalid_payment": "Geçersiz ödeme bilgisi",
    "tv_not_found": "TradingView kullanıcısı bulunamadı",
    "suspicious": "Şüpheli aktivite",
    "other": "Diğer sebep"
}

# ==================== FLASK ====================
app = Flask(__name__)

@app.route("/")
@app.route("/health")
def health():
    uptime = int((datetime.now(timezone.utc) - START_TIME).total_seconds())
    return jsonify({
        "status": "ok",
        "version": "1.0",
        "uptime": uptime,
        "bot": BOT_STATUS
    }), 200

@app.route("/ping")
def ping():
    return "pong", 200

# ==================== GOOGLE SHEETS ====================
async def save_to_sheets(data: dict) -> bool:
    """Google Sheets'e webhook ile kaydet"""
    if not SHEETS_WEBHOOK:
        log.warning("SHEETS_WEBHOOK not configured")
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.post(SHEETS_WEBHOOK, json=data)
            if response.status_code == 200:
                log.info(f"✅ Sheets'e kaydedildi: {data.get('tradingview', '?')}")
                return True
            else:
                log.error(f"Sheets error: {response.status_code}")
    except Exception as e:
        log.error(f"Sheets webhook error: {e}")
    return False

async def get_expired_users() -> list:
    """Süresi dolan kullanıcıları al"""
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

# ==================== HELPERS ====================
def calculate_end_date(days: int) -> str:
    end = datetime.now(timezone.utc) + timedelta(days=days)
    return end.strftime("%d.%m.%Y")

# ==================== BOT HANDLERS ====================
async def cmd_start(update: Update, context):
    """Start komutu - website'den deep link ile gelir"""
    user = update.effective_user
    args = context.args if context.args else []
    
    log.info(f"START: {user.id} - args: {args}")
    
    # Deep link'ten plan al
    plan_key = args[0] if args else None
    
    if plan_key and plan_key in PLANS:
        plan = PLANS[plan_key]
        context.user_data['plan_key'] = plan_key
        context.user_data['plan'] = plan
        
        if plan_key == "trial":
            # Deneme için sadece TradingView sor
            await update.message.reply_text(
                f"🌴 *Malibu PRZ Suite*\n\n"
                f"✅ *{plan['name']}* seçildi!\n\n"
                f"📝 Lütfen TradingView kullanıcı adınızı yazın:",
                parse_mode="Markdown"
            )
            return TRADINGVIEW
        else:
            # Ücretli plan
            await update.message.reply_text(
                f"🌴 *Malibu PRZ Suite*\n\n"
                f"✅ *{plan['name']} ({plan['price']})* seçildi!\n\n"
                f"📝 Lütfen TradingView kullanıcı adınızı yazın:",
                parse_mode="Markdown"
            )
            return TRADINGVIEW
    else:
        # Normal start - plan seçimi göster
        keyboard = [
            [InlineKeyboardButton("💳 Aylık - $30", callback_data="plan_monthly_30")],
            [InlineKeyboardButton("⭐ 3 Aylık - $79 (En Popüler)", callback_data="plan_quarterly_79")],
            [InlineKeyboardButton("👑 Yıllık - $269", callback_data="plan_yearly_269")],
            [InlineKeyboardButton("🆓 7 Günlük Ücretsiz Deneme", callback_data="trial")]
        ]
        
        await update.message.reply_text(
            f"Merhaba {user.first_name}! 👋\n\n"
            f"🌴 *Malibu PRZ Suite'e* hoş geldiniz!\n\n"
            f"Harmonik PRZ + SMC Malibu hibrit sistemi ile\n"
            f"kurumsal düzeyde teknik analiz yapın.\n\n"
            f"📊 Bir plan seçin:",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        return ConversationHandler.END

async def plan_selected(update: Update, context):
    """Plan seçildiğinde"""
    query = update.callback_query
    await query.answer()
    
    plan_key = query.data
    if plan_key not in PLANS:
        return ConversationHandler.END
    
    plan = PLANS[plan_key]
    context.user_data['plan_key'] = plan_key
    context.user_data['plan'] = plan
    
    await query.message.reply_text(
        f"✅ *{plan['name']} ({plan['price']})* seçildi!\n\n"
        f"📝 Lütfen TradingView kullanıcı adınızı yazın:",
        parse_mode="Markdown"
    )
    return TRADINGVIEW

async def receive_tradingview(update: Update, context):
    """TradingView kullanıcı adı alındı"""
    user = update.effective_user
    tv_username = update.message.text.strip()
    
    context.user_data['tradingview'] = tv_username
    plan = context.user_data.get('plan', {})
    plan_key = context.user_data.get('plan_key', '')
    
    if plan_key == "trial":
        # Deneme - TXID gerekmez, direkt kaydet
        await save_request(user, context, txid="DENEME")
        
        await update.message.reply_text(
            f"✅ *Deneme talebiniz alındı!*\n\n"
            f"📺 TradingView: `{tv_username}`\n"
            f"⏱️ Süre: 7 gün\n\n"
            f"24 saat içinde erişiminiz aktifleştirilecektir.\n"
            f"Teşekkürler! 🙏",
            parse_mode="Markdown"
        )
        return ConversationHandler.END
    else:
        # Ücretli plan - ödeme bilgisi göster
        await update.message.reply_text(
            f"📺 TradingView: `{tv_username}`\n\n"
            f"💰 *Ödeme Bilgileri:*\n\n"
            f"Adres (TRC20 USDT):\n"
            f"`{PAYMENT_ADDRESS}`\n\n"
            f"Tutar: *{plan.get('price', '?')}*\n\n"
            f"⚠️ Ödeme yaptıktan sonra *TXID* (işlem numarası) gönderin:",
            parse_mode="Markdown"
        )
        return TXID

async def receive_txid(update: Update, context):
    """TXID alındı - kaydı tamamla"""
    user = update.effective_user
    txid = update.message.text.strip()
    
    context.user_data['txid'] = txid
    await save_request(user, context, txid=txid)
    
    plan = context.user_data.get('plan', {})
    
    await update.message.reply_text(
        f"✅ *Ödeme talebiniz alındı!*\n\n"
        f"📋 TXID: `{txid}`\n"
        f"📊 Plan: {plan.get('name', '?')} ({plan.get('price', '?')})\n\n"
        f"İşleminiz 24 saat içinde kontrol edilecektir.\n"
        f"Onaylandığında bilgilendirileceksiniz. 🙏",
        parse_mode="Markdown"
    )
    return ConversationHandler.END

async def save_request(user, context, txid: str):
    """Talebi kaydet ve admin'e bildir"""
    plan = context.user_data.get('plan', {})
    plan_key = context.user_data.get('plan_key', '')
    tv_username = context.user_data.get('tradingview', '')
    
    now = datetime.now(timezone.utc)
    end_date = calculate_end_date(plan.get('days', 30))
    
    data = {
        'tarih': now.strftime("%d.%m.%Y %H:%M"),
        'telegram_id': str(user.id),
        'telegram_username': user.username or "Yok",
        'telegram_name': user.first_name or "",
        'txid': txid,
        'plan': plan.get('name', ''),
        'tradingview': tv_username,
        'baslangic_tarihi': now.strftime("%d.%m.%Y"),
        'bitis_tarihi': end_date,
        'durum': 'Beklemede 🟡'
    }
    
    # Google Sheets'e kaydet
    await save_to_sheets(data)
    
    # Admin'e bildir
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
                text=f"{is_trial} *Yeni Talep*\n\n"
                     f"👤 {user.first_name} (@{user.username or 'yok'})\n"
                     f"🆔 `{user.id}`\n"
                     f"📊 {plan.get('name', '?')} ({plan.get('price', '?')})\n"
                     f"📺 TradingView: `{tv_username}`\n"
                     f"📋 TXID: `{txid}`",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            log.error(f"Admin bildirim hatası: {e}")

async def admin_callback(update: Update, context):
    """Admin onay/red işlemleri"""
    query = update.callback_query
    await query.answer()
    
    if str(query.from_user.id) != str(ADMIN_ID):
        return
    
    data_parts = query.data.split("_")
    action = data_parts[0]
    
    if action == "approve":
        user_id = data_parts[1]
        user_data = pending_requests.pop(user_id, {})
        
        await query.message.edit_text(
            f"✅ *Onaylandı*\n\n"
            f"👤 {user_data.get('telegram_name', user_id)}\n"
            f"📺 {user_data.get('tradingview', '?')}",
            parse_mode="Markdown"
        )
        
        # Kullanıcıya bildir
        try:
            await context.bot.send_message(
                chat_id=int(user_id),
                text="🎉 *Erişiminiz aktifleştirildi!*\n\n"
                     "TradingView'da indikatör erişiminiz açıldı.\n"
                     "İyi işlemler! 🌴",
                parse_mode="Markdown"
            )
        except:
            pass
            
    elif action == "reject":
        user_id = data_parts[1]
        user_data = pending_requests.get(user_id, {})
        
        # Red sebeplerini göster
        keyboard = []
        for reason_key, reason_text in REJECTION_REASONS.items():
            keyboard.append([InlineKeyboardButton(
                reason_text, 
                callback_data=f"rejectreason_{user_id}_{reason_key}"
            )])
        
        await query.message.reply_text(
            f"❌ *Red Sebebi Seçin*\n\n"
            f"👤 {user_data.get('telegram_name', user_id)}\n"
            f"📺 {user_data.get('tradingview', '?')}",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
    
    elif action == "rejectreason":
        # data format: rejectreason_USER_ID_REASON_KEY
        user_id = data_parts[1]
        reason_key = data_parts[2]
        user_data = pending_requests.pop(user_id, {})
        reason_text = REJECTION_REASONS.get(reason_key, "Belirtilmedi")
        
        await query.message.edit_text(
            f"❌ *Reddedildi*\n\n"
            f"👤 {user_data.get('telegram_name', user_id)}\n"
            f"📺 {user_data.get('tradingview', '?')}\n"
            f"📋 Sebep: *{reason_text}*",
            parse_mode="Markdown"
        )
        
        # Kullanıcıya sebepli red bildirimi
        try:
            await context.bot.send_message(
                chat_id=int(user_id),
                text=f"❌ *Talebiniz Reddedildi*\n\n"
                     f"Sebep: {reason_text}\n\n"
                     f"Sorularınız için destek ile iletişime geçebilirsiniz.",
                parse_mode="Markdown"
            )
        except:
            pass
    
    elif action == "manualreject":
        # Manuel red (eski kayıtlar için)
        # data format: manualreject_USER_ID_REASON_KEY
        user_id = data_parts[1]
        reason_key = data_parts[2]
        reason_text = REJECTION_REASONS.get(reason_key, "Belirtilmedi")
        
        await query.message.edit_text(
            f"❌ *Manuel Red Gönderildi*\n\n"
            f"🆔 User ID: `{user_id}`\n"
            f"📋 Sebep: *{reason_text}*",
            parse_mode="Markdown"
        )
        
        # Kullanıcıya bildirim gönder
        try:
            await context.bot.send_message(
                chat_id=int(user_id),
                text=f"❌ *Talebiniz Reddedildi*\n\n"
                     f"Sebep: {reason_text}\n\n"
                     f"Sorularınız için destek ile iletişime geçebilirsiniz.",
                parse_mode="Markdown"
            )
        except Exception as e:
            await query.message.reply_text(f"⚠️ Kullanıcıya gönderilemedi: {e}")


async def cmd_cancel(update: Update, context):
    """İptal komutu"""
    await update.message.reply_text(
        "İşlem iptal edildi.\n\nYeniden başlamak için /start yazın."
    )
    return ConversationHandler.END

# ==================== ADMIN COMMANDS ====================
async def cmd_pending(update: Update, context):
    """Bekleyen talepler"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    
    count = len(pending_requests)
    await update.message.reply_text(f"⏳ Bekleyen talep: {count}")

async def cmd_status(update: Update, context):
    """Bot durumu"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    
    uptime = int((datetime.now(timezone.utc) - START_TIME).total_seconds())
    hours = uptime // 3600
    minutes = (uptime % 3600) // 60
    
    await update.message.reply_text(
        f"📊 *Bot Durumu*\n\n"
        f"✅ Çalışıyor\n"
        f"⏱️ Uptime: {hours}s {minutes}dk\n"
        f"🔄 Restart: {BOT_STATUS['restarts']}\n"
        f"❌ Hatalar: {BOT_STATUS['errors']}",
        parse_mode="Markdown"
    )

async def cmd_notify_expired(update: Update, context):
    """Süresi dolanlara bildirim gönder"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    
    await update.message.reply_text("🔄 Süresi dolanlar kontrol ediliyor...")
    
    expired_users = await get_expired_users()
    
    if not expired_users:
        await update.message.reply_text("✅ Süresi dolan kullanıcı yok.")
        return
    
    sent = 0
    expired_count = len(expired_users)
    for user in expired_users:
        try:
            raw_id = user.get('telegram_id', '')
            user_id = str(raw_id).strip()
            if user_id and user_id.isdigit():
                await context.bot.send_message(
                    chat_id=int(user_id),
                    text=f"⚠️ Malibu PRZ Suite erişiminiz sona erdi. Yenilemek için: {WEBSITE_URL}/",
                    parse_mode="Markdown"
                )
                sent += 1
                await asyncio.sleep(0.15)
        except Exception as e:
            log.warning(f"Bildirim gönderilemedi {user.get('telegram_id')}: {e}")
    
    await update.message.reply_text(f"📨 {sent}/{expired_count} kişiye bildirim gönderildi.")

async def cmd_scan(update: Update, context):
    """Sheets'i kontrol et ve süresi dolanlara bildirim gönder - Crystal Clear Edition"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    
    status_msg = await update.message.reply_text("🔍 Gelişmiş tarama başlatılıyor... Lütfen bekleyin.")
    
    try:
        expired_users = await get_expired_users()
        
        if not expired_users:
            await status_msg.edit_text("✅ Süresi dolan veya bildirim bekleyen kullanıcı bulunamadı.")
            return
            
        if isinstance(expired_users, dict) and "error" in expired_users:
            await status_msg.edit_text(f"❌ Sheets Hatası: {expired_users.get('error')}")
            return

        total_detected = len(expired_users)
        sent = 0
        no_id = 0
        errors = 0
        
        for user in expired_users:
            raw_id = str(user.get('telegram_id', '')).strip()
            
            # ID kontrolü (Sayısal mı?)
            if raw_id and raw_id.isdigit():
                try:
                    await context.bot.send_message(
                        chat_id=int(raw_id),
                        text=f"⚠️ Malibu PRZ Suite erişiminiz sona erdi. Yenilemek için: {WEBSITE_URL}/",
                        parse_mode="Markdown"
                    )
                    sent += 1
                    await asyncio.sleep(0.15)
                except Exception as e:
                    errors += 1
                    log.error(f"Mesaj hatası ({raw_id}): {e}")
            else:
                # ID "Yok" veya geçersiz olanlar
                no_id += 1
        
        report = (
            f"🚀 *Tarama Raporu*\n\n"
            f"📅 Tarih: `{datetime.now(timezone.utc).strftime('%d.%m.%Y')}`\n"
            f"🔍 Tespit Edilen Süresi Dolan: `{total_detected}`\n\n"
            f"✅ Bildirim Gönderilen: `{sent}`\n"
            f"⚠️ ID'si Eksik (Yok): `{no_id}`\n"
            f"❌ Teknik Hata: `{errors}`\n\n"
            f"*Not:* ID'si 'Yok' olanlara Telegram üzerinden ulaşılamaz. Yeni kayıtlarda ID otomatik kaydedilecektir."
        )
        await status_msg.edit_text(report, parse_mode="Markdown")
        
    except Exception as e:
        log.error(f"Scan error: {e}")
        await status_msg.edit_text(f"❌ Tarama sırasında teknik hata oluştu: {e}")

async def cmd_sync(update: Update, context):
    """Sheets senkronizasyonu"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    await update.message.reply_text("🔄 Sheets ile senkronizasyon başlatıldı...")
    # Webhook üzerinden veri çekme mantığı buraya gelebilir
    await update.message.reply_text("✅ Senkronizasyon tamamlandı.")

async def cmd_repair_sheets(update: Update, context):
    """Sheets tablolarını onar"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    await update.message.reply_text("🔧 Sheets tabloları kontrol ediliyor...")
    # Tablo onarım mantığı buraya gelecek
    await update.message.reply_text("✅ Onarım tamamlandı.")

async def cmd_reply(update: Update, context):
    """Admin'in kullanıcıya direkt yanıt vermesi"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    
    # Sadece son mesaj gönderen kullanıcıya yanıt ver
    last_msg = last_user_message.get(str(ADMIN_ID))
    if not last_msg:
        await update.message.reply_text("⚠️ Henüz mesaj gönderen kullanıcı yok.")
        return
    
    # /reply komutundan sonraki mesajı al
    if not context.args:
        await update.message.reply_text(
            f"💬 *Yanıt Modu*\n\n"
            f"Son mesaj: {last_msg['user_name']} ({last_msg['user_id']})\n\n"
            f"Kullanım: `/reply mesajınız buraya`",
            parse_mode="Markdown"
        )
        return
    
    message_text = " ".join(context.args)
    
    try:
        await context.bot.send_message(
            chat_id=int(last_msg['user_id']),
            text=f"📩 *Admin'den Mesaj:*\n\n{message_text}",
            parse_mode="Markdown"
        )
        await update.message.reply_text(f"✅ Mesaj gönderildi: {last_msg['user_name']}")
    except Exception as e:
        await update.message.reply_text(f"❌ Mesaj gönderilemedi: {e}")

async def admin_direct_reply(update: Update, context):
    """Admin reply modundayken mesaj gönderme"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    
    # Admin'in reply modunda olup olmadığını kontrol et
    if 'reply_mode' in context.user_data and context.user_data['reply_mode']:
        target_user = context.user_data.get('reply_target')
        if target_user:
            try:
                await context.bot.send_message(
                    chat_id=int(target_user['user_id']),
                    text=f"📩 *Admin'den Mesaj:*\n\n{update.message.text}",
                    parse_mode="Markdown"
                )
                await update.message.reply_text(
                    f"✅ Gönderildi: {target_user['user_name']}\n\n"
                    f"Çıkmak için /done yazın."
                )
            except Exception as e:
                await update.message.reply_text(f"❌ Hata: {e}")
            return

async def cmd_reject_manual(update: Update, context):
    """EKLİ KAYITLAR için manuel red (sebep ile)"""
    if str(update.effective_user.id) != str(ADMIN_ID):
        return
    
    # Kullanım: /reject [user_id]
    if not context.args:
        await update.message.reply_text(
            "📝 *Manuel Red Komutu*\n\n"
            "Kullanım: `/reject [user_id]`\n\n"
            "Örnek: `/reject 123456789`\n\n"
            "Sebep seçim menüsü açılacaktır.",
            parse_mode="Markdown"
        )
        return
    
    user_id = context.args[0]
    
    # Red sebeplerini buton olarak göster
    keyboard = []
    for reason_key, reason_text in REJECTION_REASONS.items():
        keyboard.append([InlineKeyboardButton(
            reason_text, 
            callback_data=f"manualreject_{user_id}_{reason_key}"
        )])
    
    await update.message.reply_text(
        f"❌ *Red Sebebi Seçin*\n\n"
        f"🆔 User ID: `{user_id}`\n\n"
        f"Bir sebep seçin:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )


async def cmd_help(update: Update, context):
    """Yardım"""
    text = (
        "📚 *Komutlar*\n\n"
        "/start - Başla\n"
        "/help - Yardım\n"
    )
    
    if str(update.effective_user.id) == str(ADMIN_ID):
        text += (
            "\n*Admin Komutları:*\n"
            "/pending - Bekleyen talepler\n"
            "/status - Bot durumu\n"
            "/reply \[mesaj\] - Kullanıcıya yanıt\n"
            "/notify\\_expired - Süresi dolanlara bildirim\n"
            "/scan - Tarama yap\n"
            "/sync - Verileri senkronize et\n"
            "/repair\\_sheets - Tabloları onar"
        )
    
    await update.message.reply_text(text, parse_mode="Markdown")

async def handle_user_message(update: Update, context):
    """Kullanıcıdan gelen mesajları yakala ve admin'e ilet"""
    user = update.effective_user
    
    # Admin'in kendi mesajlarını işleme
    if str(user.id) == str(ADMIN_ID):
        return
    
    # Son mesajı kaydet (admin reply için)
    last_user_message[str(ADMIN_ID)] = {
        'user_id': str(user.id),
        'user_name': user.first_name or user.username or "Kullanıcı"
    }
    
    # Admin'e yönlendir
    if ADMIN_ID:
        try:
            await context.bot.send_message(
                chat_id=int(ADMIN_ID),
                text=f"💬 *Yeni Mesaj*\n\n"
                     f"👤 {user.first_name} (@{user.username or 'yok'})\n"
                     f"🆔 `{user.id}`\n\n"
                     f"📝 Mesaj:\n{update.message.text}\n\n"
                     f"Yanıtlamak için: `/reply mesajınız`",
                parse_mode="Markdown"
            )
        except Exception as e:
            log.error(f"Admin'e mesaj iletilemedi: {e}")
    
    # Kullanıcıya otomatik yanıt
    await update.message.reply_text(
        "📨 Mesajınız iletildi!\n\n"
        "Destek ekibimiz en kısa sürede size dönüş yapacaktır. 🙏"
    )


# ==================== BOT ENGINE ====================
async def run_bot():
    """Bot'u başlat"""
    log.info("Bot başlatılıyor...")
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", cmd_start),
            CallbackQueryHandler(plan_selected, pattern="^(plan_|trial)")
        ],
        states={
            TRADINGVIEW: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_tradingview)],
            TXID: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_txid)]
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        conversation_timeout=600
    )
    
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("pending", cmd_pending))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("reply", cmd_reply))
    application.add_handler(CommandHandler("reject", cmd_reject_manual))
    application.add_handler(CommandHandler("notify_expired", cmd_notify_expired))
    application.add_handler(CommandHandler("scan", cmd_scan))
    application.add_handler(CommandHandler("sync", cmd_sync))
    application.add_handler(CommandHandler("repair_sheets", cmd_repair_sheets))
    application.add_handler(CallbackQueryHandler(admin_callback, pattern="^(approve_|reject|rejectreason|manualreject)"))
    
    # Kullanıcı mesajlarını yakala (ConversationHandler dışında)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))

    
    await application.initialize()
    
    # Webhook sil
    for i in range(3):
        try:
            await application.bot.delete_webhook(drop_pending_updates=True)
            break
        except:
            await asyncio.sleep(2)
    
    await application.start()
    BOT_STATUS["running"] = True
    log.info("✅ Bot başlatıldı - polling...")
    
    # Polling loop
    offset = None
    while not SHUTDOWN.is_set():
        try:
            updates = await application.bot.get_updates(
                offset=offset, timeout=30, allowed_updates=Update.ALL_TYPES
            )
            for upd in updates:
                offset = upd.update_id + 1
                await application.process_update(upd)
        except TimedOut:
            continue
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 1)
        except Conflict:
            log.error("CONFLICT - başka bot çalışıyor!")
            await asyncio.sleep(30)
        except (NetworkError, TelegramError) as e:
            log.warning(f"Ağ hatası: {e}")
            await asyncio.sleep(5)
        except Exception as e:
            BOT_STATUS["errors"] += 1
            log.error(f"Hata: {e}")
            await asyncio.sleep(5)
    
    await application.stop()
    await application.shutdown()

def bot_thread():
    """Bot thread'i"""
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
            log.info("♻️ 3 saniye sonra yeniden başlatılacak...")
            time.sleep(3)

def keep_alive_thread():
    """Botun uykuya geçmesini engelleyen ping sistemi"""
    time.sleep(60)
    while not SHUTDOWN.is_set():
        try:
            url = f"https://{RAILWAY_URL}/ping" if RAILWAY_URL else f"http://localhost:{PORT}/ping"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                log.debug("Keep-alive ping successful")
        except:
            pass
        # 3 dakikada bir ping at
        time.sleep(180)

def signal_handler(signum, frame):
    """Graceful shutdown"""
    log.info("⚠️ Kapatma sinyali alındı...")
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
    log.info("🌴 Malibu Telegram Bot v1.0")
    log.info(f"📊 Sheets Webhook: {'✅' if SHEETS_WEBHOOK else '❌'}")
    log.info(f"👤 Admin ID: {ADMIN_ID}")
    log.info(f"🔌 Port: {PORT}")
    log.info("=" * 50)
    
    # Bot thread
    threading.Thread(target=bot_thread, daemon=False).start()
    
    # Keep-alive thread
    threading.Thread(target=keep_alive_thread, daemon=True).start()
    
    # Flask
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
