#!/usr/bin/env python3
"""
Telegram Bot Asistan v3
- Sesli Mesaj (Whisper)
- Hafıza Sistemi (ChromaDB)
- Hatırlatıcı Sistemi (APScheduler)
- Multi-Agent (Groq + Claude opsiyonu)
"""

import os
import logging
import json
import subprocess
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import pytz
import shutil

import chromadb
from chromadb.config import Settings
# from apscheduler.schedulers.asyncio import AsyncIOScheduler  # Kullanılmıyor
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from groq import Groq

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# API Keys
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8449158473:AAG-3HbGmY2740CdrAnS1SAzw4Hnyp3DAB0")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
TIMEZONE = "Europe/Istanbul"


class ReminderSystem:
    """Hatırlatıcı yönetim sistemi."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path.home() / "asistant_reminders"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.file_path = self.data_dir / "reminders.json"
        self.reminders: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        """Hatırlatıcıları dosyadan yükler."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save(self):
        """Hatırlatıcıları dosyaya kaydeder."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.reminders, f, ensure_ascii=False, indent=2, default=str)

    def add(self, user_id: int, mesaj: str, zaman: datetime, tekrar: str = None) -> str:
        """Yeni hatırlatıcı ekler."""
        reminder_id = f"rem_{user_id}_{datetime.now().timestamp()}"

        reminder = {
            "id": reminder_id,
            "user_id": user_id,
            "mesaj": mesaj,
            "zaman": zaman.isoformat(),
            "tekrar": tekrar,  # None, "daily", "weekly", "monthly"
            "created_at": datetime.now().isoformat(),
            "gonderildi": False
        }

        self.reminders.append(reminder)
        self._save()
        return reminder_id

    def remove(self, reminder_id: str, user_id: int) -> bool:
        """Hatırlatıcı siler."""
        for i, rem in enumerate(self.reminders):
            if rem["id"] == reminder_id and rem["user_id"] == user_id:
                self.reminders.pop(i)
                self._save()
                return True
        return False

    def list_user(self, user_id: int) -> List[Dict]:
        """Kullanıcının aktif hatırlatıcılarını listeler."""
        aktif = [r for r in self.reminders
                 if r["user_id"] == user_id and not r.get("gonderildi", False)]
        # Zaman göre sırala
        aktif.sort(key=lambda x: x["zaman"])
        return aktif

    def get_due(self) -> List[Dict]:
        """Vadesi gelen hatırlatıcıları döndürür."""
        now = datetime.now(pytz.timezone(TIMEZONE))
        due = []

        for rem in self.reminders:
            if rem.get("gonderildi", False):
                continue

            rem_zaman = datetime.fromisoformat(rem["zaman"])

            if rem_zaman <= now:
                due.append(rem)

        return due

    def mark_sent(self, reminder_id: str):
        """Hatırlatıcıyı gönderildi olarak işaretler."""
        for rem in self.reminders:
            if rem["id"] == reminder_id:
                rem["gonderildi"] = True

                # Tekrarlı hatırlatıcıysa yeni tane oluştur
                if rem.get("tekrar"):
                    self._create_repeating(rem)

                self._save()
                break

    def _create_repeating(self, original: Dict):
        """Tekrarlı hatırlatıcı için yeni kayıt oluşturur."""
        original_zaman = datetime.fromisoformat(original["zaman"])

        if original["tekrar"] == "daily":
            yeni_zaman = original_zaman + timedelta(days=1)
        elif original["tekrar"] == "weekly":
            yeni_zaman = original_zaman + timedelta(weeks=1)
        elif original["tekrar"] == "monthly":
            # Basit monthly: 30 gün ekle
            yeni_zaman = original_zaman + timedelta(days=30)
        else:
            return

        new_reminder = {
            "id": f"rem_{original['user_id']}_{datetime.now().timestamp()}",
            "user_id": original["user_id"],
            "mesaj": original["mesaj"],
            "zaman": yeni_zaman.isoformat(),
            "tekrar": original["tekrar"],
            "created_at": datetime.now().isoformat(),
            "gonderildi": False
        }

        self.reminders.append(new_reminder)


class MemorySystem:
    """ChromaDB tabanlı hafıza sistemi."""

    def __init__(self, persist_dir: str = None):
        if persist_dir is None:
            persist_dir = Path.home() / "asistant_memory"
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            self.notlar_collection = self.client.get_or_create_collection(
                name="notlar",
                metadata={"description": "Kullanıcı notları"}
            )
            self.sohbet_collection = self.client.get_or_create_collection(
                name="sohbet",
                metadata={"description": "Sohbet geçmişi"}
            )
            self.available = True
        except Exception as e:
            logger.warning(f"ChromaDB hatasi: {e}")
            self.available = False

    def not_ekle(self, user_id: int, metin: str, metadata: dict = None):
        if not self.available:
            return
        not_id = f"not_{user_id}_{datetime.now().timestamp()}"
        meta = {"user_id": str(user_id), "tarih": datetime.now().isoformat(), **(metadata or {})}
        self.notlar_collection.add(ids=[not_id], documents=[metin], metadatas=[meta])
        return not_id

    def not_ara(self, user_id: int, sorgu: str, n: int = 5) -> List[dict]:
        if not self.available:
            return []
        results = self.notlar_collection.query(query_texts=[sorgu], n_results=n, where={"user_id": str(user_id)})
        sonuc_liste = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                sonuc_liste.append({'icerik': doc, 'metadata': results['metadatas'][0][i] if results['metadatas'] else {}})
        return sonuc_liste

    def sohbet_ekle(self, user_id: int, rol: str, metin: str):
        if not self.available:
            return
        mesaj_id = f"msg_{user_id}_{datetime.now().timestamp()}"
        self.sohbet_collection.add(
            ids=[mesaj_id],
            documents=[metin],
            metadatas={"user_id": str(user_id), "rol": rol, "tarih": datetime.now().isoformat()}
        )

    def sohbet_gecmisi(self, user_id: int, limit: int = 10) -> List[dict]:
        if not self.available:
            return []
        results = self.sohbet_collection.get(where={"user_id": str(user_id)}, limit=limit * 2, include=["documents", "metadatas"])
        mesajlar = []
        if results['documents']:
            for doc, meta in zip(results['documents'], results['metadatas']):
                mesajlar.append({'rol': meta.get('rol', 'user'), 'icerik': doc})
        return mesajlar[-limit:]


class SpeechToText:
    """Sesli mesajı metne çevirir (Whisper)."""

    def __init__(self):
        self.available = False
        try:
            import whisper
            self.model = whisper.load_model("base")
            self.available = True
            logger.info("Whisper model yuklendi")
        except ImportError:
            logger.warning("whisper yuklu degil")
        except Exception as e:
            logger.warning(f"Whisper yuklenemedi: {e}")

    def transcribe(self, audio_path: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            import whisper
            result = self.model.transcribe(audio_path, language="tr")
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription hatasi: {e}")
            return None


class TelegramAsistan:
    """Multi-agent Telegram asistanı."""

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.3-70b-versatile"

        # Modüller
        self.reminders = ReminderSystem()
        self.memory = MemorySystem()
        self.stt = SpeechToText()
        self.notlar_dizini = Path.home() / "telegram_asistant_notlar"
        self.notlar_dizini.mkdir(exist_ok=True)

    def groq_cagri(self, prompt: str, system: str = None, json_cikti: bool = False) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 500,
        }

        if json_cikti:
            params["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            return f"API Hatası: {str(e)}"

    def zamani_parse_et(self, metin: str, user_id: int) -> Optional[datetime]:
        """Metinden zaman çıkarır."""
        now = datetime.now(pytz.timezone(TIMEZONE))

        system_prompt = f"""Sen bir zaman çıkarıcısın. Şu an: {now.strftime('%d.%m.%Y %H:%M')}.

Verilen metinden datetime formatında zaman çıkar.
JSON formatı: {{"zaman": "DD.MM.YYYY HH:MM", "tekrar": "none|daily|weekly|monthly|null"}}

Kurallar:
- "yarın" = yarın saat 09:00
- "haftaya" = 7 gün sonra 09:00
- "her gün" = tekrar: "daily"
- "her hafta" = tekrar: "weekly"
- saat belirtilmemişse 09:00 varsayılan

Sadece JSON döndür."""

        try:
            response = self.groq_cagri(metin, system=system_prompt, json_cikti=True)
            result = json.loads(response)
            zaman_str = result.get("zaman", "")
            tekrar = result.get("tekrar")

            if zaman_str:
                try:
                    dt = datetime.strptime(zaman_str, "%d.%m.%Y %H:%M")
                    # Timezone ekle
                    tz = pytz.timezone(TIMEZONE)
                    dt = tz.localize(dt)
                    return {"zaman": dt, "tekrar": tekrar if tekrar != "none" else None}
                except:
                    pass
        except:
            pass

        return None

    def kategorile(self, metin: str) -> dict:
        """Metni kategorilere ayırır."""
        system_prompt = """Sen bir sesli asistan kategorize edicisin.
Gelen metni analiz et ve JSON formatında döndür.

Kategoriler:
- "otomasyon": Terminal komutu çalıştırma istekleri
- "bilgi": Not kaydetme istekleri
- "bilgi_ara": Notlarda arama istekleri
- "iletisim": E-posta/mesaj istekleri
- "hatirlatma": Hatırlatıcı ekleme ("x zamanında hatırlat", "yarın reminds")
- "hatirlatma_liste": Hatırlatıcıları listeleme
- "sohbet": Genel sohbet, sorular

JSON formatı: {"kategori": "...", "icerik": "...", "aciklama": "..."}

Sadece JSON döndür."""

        try:
            response = self.groq_cagri(metin, system=system_prompt, json_cikti=True)
            return json.loads(response)
        except:
            return {"kategori": "sohbet", "icerik": metin, "aciklama": "Genel sohbet"}

    def otomasyon_isle(self, metin: str, user_id: int) -> str:
        guvenli_komutlar = ["dir", "ls", "pwd", "date", "time", "whoami", "hostname", "echo", "cat", "head", "tail", "wc", "grep", "find", "tree"]
        komut = metin.strip().split()[-1] if metin.strip() else ""

        if not any(komut.startswith(k) or k in komut for k in guvenli_komutlar):
            return "[GUVENLIK] Bu komut calistirilamaz."

        try:
            result = subprocess.run(komut, shell=True, capture_output=True, text=True, encoding='utf-8', timeout=10)
            cikti = result.stdout or result.stderr or "[+] Basarili"
            return f"[CMD]\n```\n{cikti[:500]}\n```"
        except Exception as e:
            return f"[ERROR] {str(e)}"

    def bilgi_isle(self, metin: str, user_id: int) -> str:
        tarih = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dosya = self.notlar_dizini / f"user_{user_id}_not_{tarih}.md"

        with open(dosya, "w", encoding="utf-8") as f:
            f.write(f"# Not - {tarih}\n\n{metin}\n\n*Kayit: {datetime.now()}*")

        self.memory.not_ekle(user_id, metin, {"dosya": dosya.name})
        return f"[NOT] Kaydedildi: `{dosya.name}`"

    def bilgi_ara_isle(self, metin: str, user_id: int) -> str:
        sonuclar = self.memory.not_ara(user_id, metin, n=5)

        if not sonuclar:
            return "[ARAMA] Sonuc yok. Ilk notunu eklemek ister misin?"

        cevap = "[ARAMA] Bulunanlar:\n\n"
        for i, sonuc in enumerate(sonuclar[:3], 1):
            cevap += f"{i}. {sonuc['icerik'][:100]}...\n"
        return cevap

    def iletisim_isle(self, metin: str, user_id: int) -> str:
        tarih = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dosya = self.notlar_dizini / f"user_{user_id}_mesaj_{tarih}.md"

        system = "Verilen metinden alıcı ve konu çıkar. JSON: {alici: '...', konu: '...'}"
        try:
            resp = self.groq_cagri(metin, system=system, json_cikti=True)
            veri = json.loads(resp)
            alici = veri.get("alici", "[Alıcı]")
            konu = veri.get("konu", "[Konu]")
        except:
            alici, konu = "[Alıcı]", "[Konu]"

        with open(dosya, "w", encoding="utf-8") as f:
            f.write(f"# Mesaj Taslagi\n\n**Konu:** {konu}\n**Alici:** {alici}\n\n---\n\n{metin}")

        return f"[MAIL] Taslak: `{dosya.name}`\nKonu: {konu}\nAlici: {alici}"

    def hatirlatma_isle(self, metin: str, user_id: int) -> str:
        """Hatırlatıcı ekler."""
        zaman_data = self.zamani_parse_et(metin, user_id)

        if not zaman_data:
            return "[HATIRLATMA] Zaman anlasilmadi. Orn: \"Yarin saat 10'da toplantıyı hatirlat\""

        zaman = zaman_data["zaman"]
        tekrar = zaman_data.get("tekrar")

        # Mesajdan zaman ifadelerini temizle
        mesaj = metin
        for kelime in ["yarın", "haftaya", "saat", "dakika", "gun", "hafta", "her gun", "her hafta"]:
            mesaj = mesaj.replace(kelime, "")

        mesaj = mesaj.strip()

        self.reminders.add(user_id, mesaj, zaman, tekrar)

        tekrar_str = f" (Tekrar: {tekrar})" if tekrar else ""
        return f"[HATIRLATMA] Ayarlandı!\n\n⏰ {zaman.strftime('%d.%m.%Y %H:%M')}{tekrar_str}\n📝 {mesaj}"

    def hatirlatma_liste_isle(self, metin: str, user_id: int) -> str:
        """Hatırlatıcıları listeler."""
        liste = self.reminders.list_user(user_id)

        if not liste:
            return "[HATIRLATMA] Aktif hatırlatıcın yok.\n\nYeni eklemek için: \"Yarin saat 10'da toplantıyı hatirlat\""

        cevap = "[HATIRLATMA] Aktif hatırlatıcıların:\n\n"
        for i, rem in enumerate(liste, 1):
            zaman = datetime.fromisoformat(rem["zaman"])
            tekrar = f" [{rem.get('tekrar')}]" if rem.get('tekrar') else ""
            cevap += f"{i}. {zaman.strftime('%d.%m.%H:%M')}{tekrar} - {rem['mesaj'][:40]}...\n"

        return cevap

    def sohbet_isle(self, metin: str, user_id: int) -> str:
        self.memory.sohbet_ekle(user_id, "user", metin)
        gecmis = self.memory.sohbet_gecmisi(user_id, limit=6)

        system = "Sen yardımcı bir Türkçe asistansın. Kısa ve öz cevap ver."
        messages = [{"role": "system", "content": system}]

        for msg in gecmis:
            messages.append({"role": msg['rol'], "content": msg['icerik']})

        messages.append({"role": "user", "content": metin})

        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=300)
            cevap = response.choices[0].message.content
            self.memory.sohbet_ekle(user_id, "assistant", cevap)
            return cevap
        except Exception as e:
            return f"Hata: {str(e)}"

    def isle(self, metin: str, user_id: int) -> str:
        sonuc = self.kategorile(metin)
        kategori = sonuc.get("kategori", "sohbet")
        icerik = sonuc.get("icerik", metin)

        if kategori == "otomasyon":
            return self.otomasyon_isle(icerik, user_id)
        elif kategori == "bilgi":
            return self.bilgi_isle(icerik, user_id)
        elif kategori == "bilgi_ara":
            return self.bilgi_ara_isle(icerik, user_id)
        elif kategori == "iletisim":
            return self.iletisim_isle(icerik, user_id)
        elif kategori == "hatirlatma":
            return self.hatirlatma_isle(icerik, user_id)
        elif kategori == "hatirlatma_liste":
            return self.hatirlatma_liste_isle(icerik, user_id)
        else:
            return self.sohbet_isle(metin, user_id)


# Asistan instance
asistan = TelegramAsistan()


# Telegram Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📝 Not Al", callback_data="not"),
         InlineKeyboardButton("🔍 Ara", callback_data="ara")],
        [InlineKeyboardButton("⏰ Hatirlatici", callback_data="hatirlat"),
         InlineKeyboardButton("📋 Listem", callback_data="liste")],
        [InlineKeyboardButton("🖥️ Komut", callback_data="komut"),
         InlineKeyboardButton("📧 Mesaj", callback_data="mesaj")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "👋 AI Asistan v3\n\n"
        "📋 *Yeniler:*\n"
        "• ⏰ Hatırlatıcı sistemi\n"
        "• 🔍 Gelişmiş arama\n"
        "• 🎤 Sesli mesaj\n"
        "• 🧠 Hafıza sistemi\n\n"
        "Asistanın seni bekliyor!",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )


async def hatirlatici_kontrol(context: ContextTypes.DEFAULT_TYPE):
    """Periyodik hatırlatıcı kontrolü - job queue için."""
    due = asistan.reminders.get_due()

    for rem in due:
        try:
            await context.bot.send_message(
                chat_id=rem["user_id"],
                text=f"🔔 *HATIRLATMA*\n\n{rem['mesaj']}",
                parse_mode="Markdown"
            )
            asistan.reminders.mark_sent(rem["id"])
        except Exception as e:
            logger.error(f"Hatirlatici gonderilemedi: {e}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Kullanım:*\\n\\n"
        "• 📝 *Not al* - Hafızaya kaydeder\\n"
        "• 🔍 *Ara* - Notlarda arama\\n"
        "• ⏰ *Hatırlatıcı* - \"Yarin saat 10'da toplantıyı hatirlat\"\\n"
        "• 📋 *Listem* - Hatırlatıcılarımı göster\\n"
        "• 🎤 *Sesli* - Metne cevirir\\n"
        "• 🖥️ *Komut* - Terminal calistirir",
        parse_mode="Markdown"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    metin = update.message.text

    await update.message.chat.send_action("typing")

    try:
        sonuc = asistan.isle(metin, user_id)
        await update.message.reply_text(sonuc, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"[ERROR] {str(e)}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("typing")

    try:
        voice_file = await update.message.voice.get_file()
        audio_path = f"temp_voice_{update.effective_user.id}.ogg"
        await voice_file.download_to_drive(audio_path)

        if asistan.stt.available:
            await update.message.reply_text("[SOUND] Metne cevriliyor...")
            metin = asistan.stt.transcribe(audio_path)

            if metin:
                os.remove(audio_path)
                sonuc = asistan.isle(metin, update.effective_user.id)
                await update.message.reply_text(f"🎤 \"{metin}\"\n\n{sonuc}", parse_mode="Markdown")
            else:
                await update.message.reply_text("[ERROR] Metne cevirilemedi")
        else:
            await update.message.reply_text("[SOUND] Whisper yuklu degil. `pip install openai-whisper`")

    except Exception as e:
        await update.message.reply_text(f"[ERROR] {str(e)}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    if data == "not":
        await query.edit_message_text("📝 Notunuzu yazin.")
    elif data == "ara":
        await query.edit_message_text("🔍 Aramak istediginizi yazin.")
    elif data == "hatirlat":
        await query.edit_message_text("⏰ Hatirlatici ornegin: \"Yarin saat 10'da toplantıyı hatirlat\"")
    elif data == "liste":
        user_id = query.from_user.id
        cevap = asistan.hatirlatma_liste_isle("", user_id)
        await query.edit_message_text(cevap, parse_mode="Markdown")
    elif data == "komut":
        await query.edit_message_text("🖥️ Komut yazin (dir, ls, date, vb.)")
    elif data == "mesaj":
        await query.edit_message_text("📧 Mesajınızı yazın.")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(CallbackQueryHandler(button_callback))

    # Hatırlatıcı job - Telegram bot job queue kullan
    job_queue = app.job_queue

    async def reminder_job(context):
        await hatirlatici_kontrol(context)

    if job_queue:
        job_queue.run_repeating(reminder_job, interval=60, first=10)

    print("[+] Telegram bot v3 baslatiliyor...")
    print(f"[+] Hafiza: {asistan.memory.persist_dir}")
    print(f"[+] Hatirlatıcılar: {asistan.reminders.data_dir}")
    print(f"[+] STT: {'Aktif' if asistan.stt.available else 'Pasif'}")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
