#!/usr/bin/env python3
"""
MallibuSupportbot v3.0 - Minimal & Reliable
Designed specifically for Railway deployment.
All unnecessary complexity removed.
"""
import os
import sys
import json
import logging
import asyncio
from datetime import datetime

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from groq import Groq

# ==================== LOGGING ====================
logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
def get_env(key: str, default: str = "") -> str:
    """Get environment variable with fallbacks"""
    return os.getenv(key, default)

# Support multiple env var names for compatibility
TELEGRAM_TOKEN = get_env("TELEGRAM_TOKEN") or get_env("MBOT_TKN") or get_env("BOT_TOKEN")
GROQ_API_KEY = get_env("GROQ_API_KEY") or get_env("GROQ_KEY")
ADMIN_CHAT_ID = get_env("ADMIN_CHAT_ID") or get_env("MCHAT_ID")
WEBHOOK_DOMAIN = get_env("RAILWAY_PUBLIC_DOMAIN") or get_env("RAILWAY_STATIC_URL", "").replace("https://", "").replace("http://", "").rstrip("/")
PORT = int(get_env("PORT", "8080"))

# Validate required config
if not TELEGRAM_TOKEN:
    logger.error("❌ TELEGRAM_TOKEN or MBOT_TKN not set!")
    sys.exit(1)

if not GROQ_API_KEY:
    logger.warning("⚠️  GROQ_API_KEY not set - AI features disabled")

logger.info(f"✓ Token loaded: {TELEGRAM_TOKEN[:10]}...")
logger.info(f"✓ Webhook domain: {WEBHOOK_DOMAIN or 'NOT SET'}")
logger.info(f"✓ Port: {PORT}")

# ==================== FLASK APP ====================
flask_app = Flask(__name__)
CORS(flask_app)

# Global Telegram app reference
telegram_app = None

@flask_app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Railway"""
    return jsonify({
        "status": "ok",
        "bot": "MallibuSupportbot",
        "version": "3.0",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

@flask_app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify({"message": "MallibuSupportbot is running!"}), 200

@flask_app.route("/telegram-webhook", methods=["POST"])
def telegram_webhook():
    """Handle Telegram webhook updates"""
    global telegram_app
    try:
        if telegram_app is None:
            logger.error("Telegram app not initialized!")
            return "Not ready", 503
        
        data = request.get_json(force=True)
        update = Update.de_json(data, telegram_app.bot)
        
        # Run async update processing
        asyncio.run(telegram_app.process_update(update))
        
        return "OK", 200
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "Error", 500

# ==================== AI HELPER ====================
def get_ai_response(user_message: str) -> str:
    """Get AI response from Groq"""
    if not GROQ_API_KEY:
        return "AI devre dışı. Groq API key ayarlanmamış."
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Sen Malibu TradingView indikatörlerinin destek asistanısın. Türkçe yanıt ver. Kısa ve öz ol."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI error: {e}")
        return f"AI yanıtı alınamadı. Lütfen daha sonra tekrar deneyin."

# ==================== BOT HANDLERS ====================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user = update.effective_user
    logger.info(f"/start from {user.id} ({user.first_name})")
    
    # Check for deep link parameters (from website buttons)
    if context.args:
        plan = context.args[0]
        logger.info(f"Deep link: {plan}")
        
        if plan.startswith("plan_"):
            plan_names = {
                "plan_monthly_30": "Aylık ($30)",
                "plan_quarterly_79": "3 Aylık ($79)",
                "plan_yearly_269": "Yıllık ($269)"
            }
            plan_name = plan_names.get(plan, plan)
            
            await update.message.reply_text(
                f"🎉 **{plan_name}** planını seçtiniz!\n\n"
                f"💰 Ödeme için:\n"
                f"`TKUvYuzdZvkq6ksgPxfDRsUQE4vYjnEcnL`\n\n"
                f"⚠️ Sadece **TRC20 USDT** gönderin.\n"
                f"📱 Ödeme sonrası ekran görüntüsünü buraya gönderin.",
                parse_mode="Markdown"
            )
            
            # Notify admin
            if ADMIN_CHAT_ID:
                try:
                    await context.bot.send_message(
                        chat_id=int(ADMIN_CHAT_ID),
                        text=f"🆕 Yeni müşteri!\nUser: {user.first_name} ({user.id})\nPlan: {plan_name}"
                    )
                except Exception as e:
                    logger.error(f"Admin notify error: {e}")
            return
    
    # Default welcome message
    keyboard = [
        [InlineKeyboardButton("📊 Aylık - $30", callback_data="plan_monthly_30")],
        [InlineKeyboardButton("💎 3 Aylık - $79", callback_data="plan_quarterly_79")],
        [InlineKeyboardButton("🏆 Yıllık - $269", callback_data="plan_yearly_269")],
        [InlineKeyboardButton("🆓 Deneme Talebi", callback_data="trial_request")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"Merhaba {user.first_name}! 👋\n\n"
        f"Ben **Malibu PRZ Suite** destek botuyum.\n"
        f"Harmonic PRZ + SMC Malibu indikatörlerine hoş geldiniz!\n\n"
        f"🔽 Bir plan seçin veya soru sorun:",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular messages"""
    user = update.effective_user
    text = update.message.text
    logger.info(f"Message from {user.id}: {text[:50]}...")
    
    # Get AI response
    response = get_ai_response(text)
    await update.message.reply_text(response)
    
    # Forward to admin
    if ADMIN_CHAT_ID:
        try:
            await context.bot.send_message(
                chat_id=int(ADMIN_CHAT_ID),
                text=f"📨 {user.first_name} ({user.id}):\n{text}"
            )
        except Exception as e:
            logger.error(f"Admin forward error: {e}")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user = update.effective_user
    data = query.data
    logger.info(f"Callback from {user.id}: {data}")
    
    if data.startswith("plan_"):
        plan_names = {
            "plan_monthly_30": ("Aylık", "$30"),
            "plan_quarterly_79": ("3 Aylık", "$79"),
            "plan_yearly_269": ("Yıllık", "$269")
        }
        name, price = plan_names.get(data, ("Bilinmeyen", "?"))
        
        await query.message.reply_text(
            f"✅ **{name} ({price})** planını seçtiniz!\n\n"
            f"💳 Ödeme adresi (TRC20 USDT):\n"
            f"`TKUvYuzdZvkq6ksgPxfDRsUQE4vYjnEcnL`\n\n"
            f"📸 Ödeme sonrası ekran görüntüsünü buraya gönderin.",
            parse_mode="Markdown"
        )
        
        # Notify admin
        if ADMIN_CHAT_ID:
            try:
                await context.bot.send_message(
                    chat_id=int(ADMIN_CHAT_ID),
                    text=f"🆕 Plan seçimi!\nUser: {user.first_name} ({user.id})\nPlan: {name} ({price})"
                )
            except Exception as e:
                logger.error(f"Admin notify error: {e}")
    
    elif data == "trial_request":
        await query.message.reply_text(
            "📝 **Deneme Talebi**\n\n"
            "Lütfen TradingView kullanıcı adınızı yazın.\n"
            "24 saat içinde 7 günlük deneme erişiminiz aktif edilecektir.",
            parse_mode="Markdown"
        )
        
        if ADMIN_CHAT_ID:
            try:
                await context.bot.send_message(
                    chat_id=int(ADMIN_CHAT_ID),
                    text=f"🆓 Deneme talebi!\nUser: {user.first_name} ({user.id})"
                )
            except Exception as e:
                logger.error(f"Admin notify error: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages (payment screenshots)"""
    user = update.effective_user
    logger.info(f"Photo from {user.id}")
    
    await update.message.reply_text(
        "📸 Ödeme ekran görüntüsü alındı!\n"
        "İşleminiz en kısa sürede kontrol edilecektir.\n"
        "Teşekkürler! 🙏"
    )
    
    # Forward to admin
    if ADMIN_CHAT_ID:
        try:
            await context.bot.send_message(
                chat_id=int(ADMIN_CHAT_ID),
                text=f"💰 Ödeme SS!\nUser: {user.first_name} ({user.id})"
            )
            await update.message.forward(chat_id=int(ADMIN_CHAT_ID))
        except Exception as e:
            logger.error(f"Admin forward error: {e}")

# ==================== MAIN ====================
def main():
    global telegram_app
    import threading
    
    logger.info("=" * 50)
    logger.info("🚀 MallibuSupportbot v3.1.2-FINAL-BUILD-FORCE-1 Starting...")
    logger.info("=" * 50)
    
    # Build Telegram application
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    telegram_app.add_handler(CallbackQueryHandler(handle_callback))
    
    if WEBHOOK_DOMAIN:
        # ==================== WEBHOOK MODE ====================
        logger.info(f"📡 Running in WEBHOOK mode")
        webhook_url = f"https://{WEBHOOK_DOMAIN}/telegram-webhook"
        
        async def setup_webhook():
            await telegram_app.initialize()
            await telegram_app.start()
            await telegram_app.bot.delete_webhook(drop_pending_updates=True)
            await telegram_app.bot.set_webhook(
                url=webhook_url,
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            logger.info(f"✓ Webhook set: {webhook_url}")
        
        asyncio.run(setup_webhook())
        
        logger.info(f"🌐 Starting Flask on port {PORT}...")
        flask_app.run(host="0.0.0.0", port=PORT, use_reloader=False, threaded=True)
        
    else:
        # ==================== POLLING MODE (FIXED ARCHITECTURE) ====================
        # Flask is MAIN process (for health checks)
        # Telegram polling runs in BACKGROUND thread
        logger.info("📡 Running in POLLING mode")
        logger.info(f"✓ Starting health server on port {PORT}")
        
        # Initialize Telegram app
        async def init_telegram():
            await telegram_app.initialize()
            # Clear webhook to use polling
            await telegram_app.bot.delete_webhook(drop_pending_updates=True)
            await telegram_app.start()
            logger.info("✓ Telegram app initialized")
        
        asyncio.run(init_telegram())
        
        # Run Telegram polling in background thread
        def run_telegram_polling():
            """Background thread for Telegram polling"""
            logger.info("📡 Starting Telegram polling thread...")
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Polling loop
                while True:
                    try:
                        updates = loop.run_until_complete(
                            telegram_app.bot.get_updates(
                                timeout=30,
                                allowed_updates=Update.ALL_TYPES
                            )
                        )
                        for update in updates:
                            loop.run_until_complete(telegram_app.process_update(update))
                            # Acknowledge the update
                            loop.run_until_complete(
                                telegram_app.bot.get_updates(
                                    offset=update.update_id + 1,
                                    timeout=0
                                )
                            )
                    except Exception as e:
                        logger.error(f"Polling error: {e}")
                        import time
                        time.sleep(5)
            except Exception as e:
                logger.error(f"Polling thread crashed: {e}")
        
        # Start polling thread
        polling_thread = threading.Thread(target=run_telegram_polling, daemon=True)
        polling_thread.start()
        logger.info("✓ Telegram polling thread started")
        
        # Flask as MAIN process (blocking)
        logger.info(f"🌐 Flask server ready on port {PORT}")
        flask_app.run(host="0.0.0.0", port=PORT, use_reloader=False, threaded=True)

if __name__ == "__main__":
    main()

