#!/usr/bin/env python3
"""
Telegram Bot Asistan v4.1 - Ekonomik Multi-Agent
- Llama 3.3 (Groq) - Ana Agent
- Gemini Pro - Arşivci (Haftalık özet)
- ChromaDB - Günlük hafıza
- edge-tts - Ücretsiz TTS
- Whisper - STT
- GÜVENLİ Terminal - shell=False + beyaz liste
"""

import os
import sys
import json
import asyncio
import logging
import shlex
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pytz

# ChromaDB
import chromadb
from chromadb.config import Settings

# Telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

# Groq
from groq import Groq

# Google Gemini (opsiyonel - arşivci için)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Logging
logging.basicConfig(
    format='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('asistant_v4.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================
@dataclass
class Config:
    telegram_token: str
    groq_key: str
    gemini_key: str = ""

    timezone: str = "Europe/Istanbul"
    memory_path: str = None
    reminder_path: str = None
    notes_path: str = None
    archive_path: str = None

    groq_model: str = "llama-3.3-70b-versatile"
    groq_flash_model: str = "llama-3.1-8b-instant"

    # Güvenli terminal config
    safe_commands: Dict[str, List[str]] = None

    def __post_init__(self):
        if not self.memory_path:
            self.memory_path = str(Path.home() / "asistant_memory_v4")
        if not self.reminder_path:
            self.reminder_path = str(Path.home() / "asistant_reminders_v4")
        if not self.notes_path:
            self.notes_path = str(Path.home() / "asistant_notes_v4")
        if not self.archive_path:
            self.archive_path = str(Path.home() / "asistant_archives_v4")

        if self.safe_commands is None:
            # Komut => izin verilen parametreler (None = parametresiz)
            self.safe_commands = {
                'dir': None,
                'ls': ['-la', '-a', '-l'],
                'pwd': None,
                'date': None,
                'time': None,
                'whoami': None,
                'hostname': None,
                'echo': None,
                'cat': None,
                'head': ['-n'],
                'tail': ['-n'],
                'wc': None,
                'tree': None,
                'find': ['.', '-name', '-type'],
            }


config = Config(
    telegram_token=os.getenv("TELEGRAM_TOKEN", "8449158473:AAG-3HbGmY2740CdrAnS1SAzw4Hnyp3DAB0"),
    groq_key=os.getenv("GROQ_API_KEY", "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"),
)


# ==================== SECURE TERMINAL ====================
class SecureTerminal:
    """
    GÜVENLİ Terminal - shell=False + beyaz liste
    Komut enjeksiyonuna karşı korumalı
    """

    # Beyaz liste - sadece bu komutlar çalıştırılabilir
    ALLOWED_COMMANDS = {
        'dir': None,           # Parametresiz
        'ls': ['-la', '-a', '-l', '-h'],  # İzin verilen parametreler
        'pwd': None,
        'date': None,
        'time': None,
        'whoami': None,
        'hostname': None,
        'echo': None,
        'cat': None,
        'head': ['-n', '10'],
        'tail': ['-n', '10'],
        'wc': None,
        'tree': None,
        'find': ['.'],          # Sabit dizin
    }

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.allowed = self.ALLOWED_COMMANDS

    def is_safe(self, command: str) -> bool:
        """Komut güvenli mi kontrol et"""
        parts = command.strip().split()

        if not parts:
            return False

        cmd = parts[0]

        # Beyaz listede mi?
        if cmd not in self.allowed:
            return False

        # Parametre kontrolü
        allowed_params = self.allowed[cmd]

        if allowed_params is None:
            # Parametre kabul edilmiyor
            return len(parts) == 1

        # Bazı parametreler izin verilmiş
        # Basit kontrol: parametrelerin izin verilenlerle başlıyor mu?
        for param in parts[1:]:
            if not any(param.startswith(allowed) for allowed in allowed_params):
                return False

        return True

    def execute(self, command: str) -> Dict[str, Any]:
        """
        Güvenli komut çalıştır
        shell=False kullanır - komut enjeksiyonuna karşı korumalı
        """
        try:
            # Komutu analiz et
            parts = shlex.split(command.strip())

            if not parts:
                return {"success": False, "error": "Boş komut"}

            cmd = parts[0]

            # Güvenlik kontrolü
            if not self.is_safe(command):
                return {
                    "success": False,
                    "error": f"❌ Yasaklı komut: {cmd}\n\nİzin verilen: {', '.join(list(self.allowed.keys())[:5])}..."
                }

            # Parametreleri filtrele
            allowed_params = self.allowed[cmd]

            if allowed_params is None:
                args = []
            else:
                args = [p for p in parts[1:] if any(p.startswith(a) for a in allowed_params)]

            # GÜVENLİ ÇALIŞTIRMA - shell=False
            result = subprocess.run(
                [cmd] + args,
                shell=False,           # ← GÜVENLİK İÇİN KRİTİK
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=self.timeout,
                cwd=str(Path.home())   # Ana dizinde kapat
            )

            output = result.stdout or result.stderr

            return {
                "success": True,
                "output": output.strip() or "✅ Başarılı (çıktı yok)",
                "command": cmd,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "⏱️ Komut zaman aşımına uğradı"}
        except FileNotFoundError:
            return {"success": False, "error": f"❌ Komut bulunamadı: {cmd}"}
        except Exception as e:
            return {"success": False, "error": f"❌ Hata: {str(e)}"}


# ==================== GEMINI ARCHIVIST ====================
class GeminiArchivist:
    """
    Gemini Pro - Arşivci Agent
    Hahaftalık ChromaDB özetleme ve arşivleme
    """

    def __init__(self, api_key: str, archive_path: str):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(exist_ok=True)

        if not GEMINI_AVAILABLE or not api_key:
            logger.warning("Gemini arşivci aktif değil (API key yok veya yüklü değil)")
            self.available = False
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')  # 1M context
            self.available = True
            logger.info("Gemini Pro Arşivci aktif")
        except Exception as e:
            logger.error(f"Gemini başlatma hatası: {e}")
            self.available = False

    async def create_weekly_summary(self, chroma_client, user_id: int = None) -> Dict:
        """
        ChromaDB'den haftalık notları alıp Gemini Pro ile özetle

        Returns:
            {
                "week": "2025-W02",
                "note_count": 15,
                "summary": "...",
                "categories": {...},
                "archive_file": "path/to/summary.json"
            }
        """
        if not self.available:
            return {"error": "Gemini arşivci aktif değil"}

        try:
            # 1. ChromaDB'den geçen haftanın verilerini al
            collection = chroma_client.get_collection("notes")

            # Geçen haftanın tarih aralığı
            now = datetime.now(pytz.timezone(config.timezone))
            week_start = now - timedelta(days=now.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

            # Tüm notları al (user_id varsa filtrele)
            where_clause = {"user_id": str(user_id)} if user_id else None

            results = collection.get(
                where=where_clause,
                limit=1000,
                include=["documents", "metadatas"]
            )

            # Haftalık filtre
            weekly_notes = []
            for doc, meta in zip(results['documents'], results['metadatas']):
                created = meta.get('created', '')
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created)
                        if created_dt >= week_start:
                            weekly_notes.append({
                                'content': doc,
                                'metadata': meta,
                                'created': created_dt
                            })
                    except:
                        weekly_notes.append({'content': doc, 'metadata': meta, 'created': None})

            if not weekly_notes:
                return {"week": now.strftime("%Y-W%U"), "note_count": 0, "summary": "Bu hafta not yok."}

            # 2. Gemini'ya gönder ve özetle
            notes_text = "\n\n".join([
                f"[{n['created'].strftime('%d.%m %H:%M') if n['created'] else '?'}] {n['content']}"
                for n in weekly_notes[:100]  # Max 100 not
            ])

            prompt = f"""Sen bir arşivcisin. Aşağıdaki notları haftalık özetle.

HAFTALIK NOTLAR:
{notes_text}

JSON formatında döndür:
{{
    "summary": "Haftalık özet (2-3 cümle)",
    "key_topics": ["konu1", "konu2", "konu3"],
    "action_items": ["yapılacak1", "yapılacak2"],
    "mood": "pozitif/nötr/negatif",
    "productivity_score": 7
}}

Sadece JSON döndür."""

            response = self.model.generate_content(prompt)
            summary_text = response.text

            # JSON'ı çıkar
            try:
                # Markdown code block içinde olabilir
                if "```json" in summary_text:
                    summary_text = summary_text.split("```json")[1].split("```")[0].strip()
                elif "```" in summary_text:
                    summary_text = summary_text.split("```")[1].split("```")[0].strip()

                summary_data = json.loads(summary_text)
            except:
                # JSON parse başarısızsa varsayılan
                summary_data = {
                    "summary": summary_text[:500],
                    "key_topics": [],
                    "action_items": [],
                    "mood": "nötr",
                    "productivity_score": 5
                }

            # 3. Arşiv dosyasına kaydet
            week_id = now.strftime("%Y-W%U")
            archive_file = self.archive_path / f"week_{week_id}_user_{user_id if user_id else 'all'}.json"

            archive_data = {
                "week": week_id,
                "week_start": week_start.isoformat(),
                "week_end": now.isoformat(),
                "user_id": user_id,
                "note_count": len(weekly_notes),
                "summary": summary_data,
                "raw_notes": weekly_notes[:20]  # İlk 20 not
            }

            archive_file.write_text(
                json.dumps(archive_data, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )

            logger.info(f"Haftalık özet oluşturuldu: {archive_file.name}")

            return {
                "week": week_id,
                "note_count": len(weekly_notes),
                "summary": summary_data.get("summary", ""),
                "key_topics": summary_data.get("key_topics", []),
                "action_items": summary_data.get("action_items", []),
                "mood": summary_data.get("mood", "nötr"),
                "productivity_score": summary_data.get("productivity_score", 5),
                "archive_file": str(archive_file)
            }

        except Exception as e:
            logger.error(f"Haftalık özet hatası: {e}")
            return {"error": str(e)}

    async def get_insights(self, weeks: int = 4) -> Dict:
        """
        Son haftaların analizini yap
        """
        if not self.available:
            return {"error": "Gemini arşivci aktif değil"}

        try:
            # Arşiv dosyalarını bul
            archive_files = sorted(self.archive_path.glob("week_*.json"))[-weeks:]

            if not archive_files:
                return {"error": "Yeterli arşiv yok"}

            # Arşivleri oku
            archives_data = []
            for f in archive_files:
                try:
                    data = json.loads(f.read_text(encoding='utf-8'))
                    archives_data.append(data)
                except:
                    continue

            if not archives_data:
                return {"error": "Arşivler okunamadı"}

            # Gemini'ya analiz et
            archives_summary = json.dumps(archives_data, ensure_ascii=False, indent=2)

            prompt = f"""Son {weeks} haftanın arşiv verileri aşağıda. Analiz et:

{archives_summary[:10000]}

JSON formatında döndür:
{{
    "trend": "iyileşiyor/bozuluyor/stabil",
    "top_topics": ["konu1", "konu2"],
    "avg_productivity": 7.5,
    "recommendations": ["öneri1", "öneri2"]
}}

Sadece JSON döndür."""

            response = self.model.generate_content(prompt)
            insight_text = response.text

            if "```json" in insight_text:
                insight_text = insight_text.split("```json")[1].split("```")[0].strip()
            elif "```" in insight_text:
                insight_text = insight_text.split("```")[1].split("```")[0].strip()

            return json.loads(insight_text)

        except Exception as e:
            logger.error(f"İçgörü hatası: {e}")
            return {"error": str(e)}


# ==================== AGENT SYSTEM ====================
class LLamaAgent:
    """Llama 3.3 Agent - Groq üzerinden"""

    SYSTEM_PROMPT = """Sen AI Asistan Ajanısın.

ARAÇLAR:
- memory_search: Notlarda arama
- memory_save: Not kaydetme
- terminal_run: Komut çalıştırma (sadece güvenli olanlar)
- reminder_add: Hatırlatıcı ekleme
- reminder_list: Hatırlatıcı listeleme
- chat: Genel sohbet

JSON formatı döndür:
{"tool": "tool_name", "params": {"key": "value"}}"""

    FLASH_SYSTEM = """Kategorize et:
- "otomasyon": Komut çalıştırma
- "bilgi": Not kaydetme
- "bilgi_ara": Notlarda arama
- "iletisim": Mesaj taslağı
- "hatirlatma": Hatırlatıcı ekle
- "hatirlatma_liste": Listele
- "sohbet": Genel sohbet

JSON: {"kategori": "...", "icerik": "..."}"""

    def __init__(self, api_key: str, model: str, flash_model: str):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.flash_model = flash_model

    def call(self, prompt: str, use_flash: bool = False, json_mode: bool = False) -> Dict:
        messages = [
            {"role": "system", "content": self.FLASH_SYSTEM if use_flash else self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        params = {
            "model": self.flash_model if use_flash else self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 300 if use_flash else 500,
        }

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
            if json_mode:
                return json.loads(content)
            return {"raw": content}
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def categorize(self, text: str) -> Dict:
        """Hızlı kategorilendirme"""
        try:
            result = self.call(text, use_flash=True, json_mode=True)
            return result
        except:
            # Fallback
            text_lower = text.lower()
            if any(k in text_lower for k in ["çalıştır", "komut", "run"]):
                return {"kategori": "otomasyon", "icerik": text}
            elif any(k in text_lower for k in ["not al", "kaydet"]):
                return {"kategori": "bilgi", "icerik": text}
            elif any(k in text_lower for k in ["ne not", "bul", "ara"]):
                return {"kategori": "bilgi_ara", "icerik": text}
            elif any(k in text_lower for k in ["mail", "mesaj at"]):
                return {"kategori": "iletisim", "icerik": text}
            elif any(k in text_lower for k in ["hatırlat", "yarın", "haftaya"]):
                return {"kategori": "hatirlatma", "icerik": text}
            else:
                return {"kategori": "sohbet", "icerik": text}


# ==================== MEMORY SYSTEM ====================
class MemorySystem:
    """ChromaDB ile vektör hafıza"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False)
            )
            self.notes = self.client.get_or_create_collection("notes")
            self.chats = self.client.get_or_create_collection("chats")
            self.available = True
            logger.info("ChromaDB aktif")
        except Exception as e:
            logger.error(f"ChromaDB hatası: {e}")
            self.available = False

    @property
    def chroma_client(self):
        """Dış erişim için"""
        return self.client if self.available else None

    def save_note(self, user_id: int, text: str, meta: Dict = None) -> bool:
        if not self.available:
            return False
        try:
            doc_id = f"note_{user_id}_{datetime.now().timestamp()}"
            metadata = {"user_id": str(user_id), "created": datetime.now().isoformat()}
            if meta:
                metadata.update(meta)
            self.notes.add(ids=[doc_id], documents=[text], metadatas=[metadata])
            return True
        except Exception as e:
            logger.error(f"Not kayıt hatası: {e}")
            return False

    def search(self, user_id: int, query: str, n: int = 5) -> List[Dict]:
        if not self.available:
            return []
        try:
            results = self.notes.query(
                query_texts=[query],
                n_results=n,
                where={"user_id": str(user_id)}
            )
            items = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    items.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    })
            return items
        except Exception as e:
            logger.error(f"Arama hatası: {e}")
            return []

    def add_chat(self, user_id: int, role: str, text: str):
        if not self.available:
            return
        try:
            msg_id = f"chat_{user_id}_{datetime.now().timestamp()}"
            self.chats.add(
                ids=[msg_id],
                documents=[text],
                metadatas={"user_id": str(user_id), "role": role, "created": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Sohbet kayıt hatası: {e}")

    def get_context(self, user_id: int, limit: int = 10) -> List[Dict]:
        if not self.available:
            return []
        try:
            results = self.chats.get(
                where={"user_id": str(user_id)},
                limit=limit * 2,
                include=["documents", "metadatas"]
            )
            messages = []
            if results['documents']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    messages.append({'role': meta.get('role', 'user'), 'content': doc})
            return messages[-limit:]
        except Exception as e:
            logger.error(f"Context alma hatası: {e}")
            return []


# ==================== REMINDER SYSTEM ====================
class ReminderSystem:
    """Hatırlatıcı yönetimi"""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        self.file = self.path / "reminders.json"
        self.reminders: List[Dict] = self._load()
        self.tz = pytz.timezone(config.timezone)

    def _load(self) -> List[Dict]:
        if self.file.exists():
            try:
                return json.loads(self.file.read_text(encoding='utf-8'))
            except:
                return []
        return []

    def _save(self):
        try:
            self.file.write_text(
                json.dumps(self.reminders, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Hatırlatıcı kayıt hatası: {e}")

    def add(self, user_id: int, message: str, when: datetime, repeat: str = None) -> str:
        rem_id = f"rem_{user_id}_{datetime.now().timestamp()}"
        reminder = {
            "id": rem_id,
            "user_id": user_id,
            "message": message,
            "when": when.isoformat(),
            "repeat": repeat,
            "sent": False,
            "created": datetime.now().isoformat()
        }
        self.reminders.append(reminder)
        self._save()
        return rem_id

    def list_user(self, user_id: int) -> List[Dict]:
        return [r for r in self.reminders if r['user_id'] == user_id and not r.get('sent', False)]

    def get_due(self) -> List[Dict]:
        now = datetime.now(self.tz)
        due = []
        for r in self.reminders:
            if r.get('sent', False):
                continue
            try:
                when = datetime.fromisoformat(r['when'])
                if when <= now:
                    due.append(r)
            except:
                continue
        return due

    def mark_sent(self, rem_id: str):
        for r in self.reminders:
            if r['id'] == rem_id:
                r['sent'] = True
                self._save()
                break


# ==================== TTS/STT ====================
class TTSEngine:
    def __init__(self):
        self.available = False
        try:
            import edge_tts
            self.edge_tts = edge_tts
            self.available = True
            self.voice = "tr-TR-AhmetNeural"
            logger.info("edge-tts TTS aktif")
        except ImportError:
            logger.warning("edge-tts yüklü değil")

    async def speak(self, text: str, output_path: str) -> bool:
        if not self.available:
            return False
        try:
            communicate = self.edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            return True
        except Exception as e:
            logger.error(f"TTS hatası: {e}")
            return False


class STTEngine:
    def __init__(self):
        self.available = False
        try:
            import whisper
            self.model = whisper.load_model("base")
            self.available = True
            logger.info("Whisper STT aktif")
        except Exception as e:
            logger.warning(f"Whisper yükleme hatası: {e}")

    def transcribe(self, audio_path: str) -> Optional[str]:
        if not self.available:
            return None
        try:
            result = self.model.transcribe(audio_path, language="tr")
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription hatası: {e}")
            return None


# ==================== MAIN ASSISTANT ====================
class AsistanV4:
    """Ana asistan - tüm modülleri bir araya getirir"""

    def __init__(self, config: Config):
        self.config = config

        # Modüller
        self.agent = LLamaAgent(config.groq_key, config.groq_model, config.groq_flash_model)
        self.memory = MemorySystem(config.memory_path)
        self.reminders = ReminderSystem(config.reminder_path)
        self.terminal = SecureTerminal()
        self.archivist = GeminiArchivist(config.gemini_key, config.archive_path)
        self.tts = TTSEngine()
        self.stt = STTEngine()

        # Notlar dizini
        self.notes_dir = Path(config.notes_path)
        self.notes_dir.mkdir(exist_ok=True)

    async def process(self, text: str, user_id: int) -> str:
        """Ana işlem pipeline"""
        try:
            result = self.agent.categorize(text)
            category = result.get("kategori", "sohbet")
            content = result.get("icerik", text)

            logger.info(f"User {user_id} | {category} | {text[:50]}...")

            if category == "otomasyon":
                return await self._handle_automation(content, user_id)
            elif category == "bilgi":
                return await self._handle_note(content, user_id)
            elif category == "bilgi_ara":
                return await self._handle_search(content, user_id)
            elif category == "iletisim":
                return await self._handle_email(content, user_id)
            elif category == "hatirlatma":
                return await self._handle_reminder_add(content, user_id)
            elif category == "hatirlatma_liste":
                return await self._handle_reminder_list(user_id)
            elif category == "ozet":
                return await self._handle_summary(user_id)
            else:
                return await self._handle_chat(text, user_id)

        except Exception as e:
            logger.error(f"İşlem hatası: {e}")
            return f"⚠️ Bir hata oluştu."

    async def _handle_automation(self, text: str, user_id: int) -> str:
        """GÜVENLİ Terminal - shell=False ile"""
        # Komutu çıkar
        parts = text.strip().split()
        cmd = parts[-1] if parts else ""

        result = self.terminal.execute(cmd)

        if result["success"]:
            return f"🖥️ **Çıktı:**\n```\n{result['output']}\n```"
        else:
            return result["error"]

    async def _handle_note(self, text: str, user_id: int) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.notes_dir / f"note_{user_id}_{timestamp}.md"
        filename.write_text(
            f"# Not - {timestamp}\n\n{text}\n\n*Kaydedilme: {datetime.now()}*",
            encoding='utf-8'
        )
        self.memory.save_note(user_id, text, {"file": filename.name})
        return f"📝 Not kaydedildi!\n\n`{filename.name}`"

    async def _handle_search(self, text: str, user_id: int) -> str:
        results = self.memory.search(user_id, text, n=5)
        if not results:
            return "🔍 Sonuç bulunamadı."
        response = "🔍 **Bulunan notlar:**\n\n"
        for i, r in enumerate(results[:3], 1):
            response += f"{i}. {r['content'][:80]}...\n"
        return response

    async def _handle_email(self, text: str, user_id: int) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.notes_dir / f"email_{user_id}_{timestamp}.md"
        filename.write_text(
            f"# E-posta Taslağı\n\n**Konu:** [Konu]\n**Alıcı:** [Alıcı]\n\n---\n\n{text}\n\n---\n*Taslak: {datetime.now()}*",
            encoding='utf-8'
        )
        return f"📧 Taslak hazır!\n\n`{filename.name}`"

    async def _handle_reminder_add(self, text: str, user_id: int) -> str:
        now = datetime.now(pytz.timezone(config.timezone))
        reminder_time = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
        message = text
        for word in ["hatırlat", "yarın", "saat", "haftaya"]:
            message = message.replace(word, "")
        message = message.strip() or "Hatırlatıcı"
        self.reminders.add(user_id, message, reminder_time)
        return f"⏰ Hatırlatıcı ayarlandı!\n\n**Zaman:** {reminder_time.strftime('%d.%m.%Y %H:%M')}\n**Mesaj:** {message}"

    async def _handle_reminder_list(self, user_id: int) -> str:
        reminders = self.reminders.list_user(user_id)
        if not reminders:
            return "📋 Aktif hatırlatıcın yok."
        response = "📋 **Hatırlatıcıların:**\n\n"
        for i, r in enumerate(reminders, 1):
            when = datetime.fromisoformat(r['when'])
            response += f"{i}. {when.strftime('%d.%m %H:%M')} - {r['message'][:40]}\n"
        return response

    async def _handle_summary(self, user_id: int) -> str:
        """Haftalık özet - Gemini Pro"""
        if not self.archivist.available:
            return "📊 Arşivci aktif değil. Gemini API key gerekli."

        summary = await self.archivist.create_weekly_summary(self.memory.chroma_client, user_id)

        if "error" in summary:
            return f"📊 Hata: {summary['error']}"

        response = f"📊 **Haftalık Özet ({summary.get('week', '?')}**\n\n"
        response += f"📝 {summary.get('note_count', 0)} not\n\n"
        response += f"**Özet:** {summary.get('summary', '')}\n\n"

        if summary.get('key_topics'):
            response += f"**Konular:** {', '.join(summary['key_topics'][:5])}\n\n"

        if summary.get('action_items'):
            response += f"**Yapılacaklar:**\n"
            for item in summary['action_items'][:3]:
                response += f"• {item}\n"

        response += f"\n📁 Arşiv: {Path(summary.get('archive_file', '')).name}"

        return response

    async def _handle_chat(self, text: str, user_id: int) -> str:
        self.memory.add_chat(user_id, "user", text)
        context = self.memory.get_context(user_id, limit=6)

        messages = [{"role": "system", "content": "Sen yardımcı bir Türkçe asistansın. Kısa ve öz cevap ver."}]
        for msg in context:
            messages.append({"role": msg['role'], "content": msg['content']})
        if not context or context[-1]['content'] != text:
            messages.append({"role": "user", "content": text})

        try:
            response = self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=messages,
                max_tokens=300
            )
            reply = response.choices[0].message.content
            self.memory.add_chat(user_id, "assistant", reply)
            return reply
        except Exception as e:
            logger.error(f"Sohbet hatası: {e}")
            return "Bir hata oluştu."

    async def check_reminders(self, bot) -> int:
        due = self.reminders.get_due()
        sent = 0
        for rem in due:
            try:
                await bot.send_message(
                    chat_id=rem['user_id'],
                    text=f"🔔 **HATIRLATMA**\n\n{rem['message']}",
                    parse_mode='Markdown'
                )
                self.reminders.mark_sent(rem['id'])
                sent += 1
                logger.info(f"Hatırlatıcı gönderildi: {rem['id']}")
            except Exception as e:
                logger.error(f"Hatırlatıcı gönderilemedi: {e}")
        return sent


# ==================== HANDLERS ====================
asistan = AsistanV4(config)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📝 Not", callback_data="note"),
         InlineKeyboardButton("🔍 Ara", callback_data="search")],
        [InlineKeyboardButton("⏰ Hatırlat", callback_data="remind"),
         InlineKeyboardButton("📋 Listem", callback_data="list")],
        [InlineKeyboardButton("📊 Özet", callback_data="summary")],
    ]
    await update.message.reply_text(
        "🤖 **Asistan v4.1**\n\n"
        "⚡ Llama 3.3 Agent\n"
        "🔒 Güvenli Terminal (shell=False)\n"
        "📊 Gemini Pro Arşivci\n"
        "🧠 ChromaDB Hafıza\n"
        "⏰ Hatırlatıcı Sistemi\n"
        "🎤 Sesli Mesaj\n\n"
        "Nasıl yardımcı olabilirim?",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("typing")
    user_id = update.effective_user.id
    text = update.message.text

    try:
        response = await asistan.process(text, user_id)
        await update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Mesaj işleme hatası: {e}")
        await update.message.reply_text("⚠️ Bir hata oluştu.")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("typing")
    user_id = update.effective_user.id

    try:
        voice = await update.message.voice.get_file()
        temp_path = f"voice_{user_id}.ogg"
        await voice.download_to_drive(temp_path)

        if asistan.stt.available:
            await update.message.reply_text("🎤 Metne çevriliyor...")
            text = asistan.stt.transcribe(temp_path)
            os.remove(temp_path)

            if text:
                response = await asistan.process(text, user_id)
                await update.message.reply_text(f"🎤 \"{text}\"\n\n{response}", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Metne çevrilemedi")
        else:
            await update.message.reply_text("🎤 Whisper yüklü değil.")
    except Exception as e:
        logger.error(f"Sesli mesaj hatası: {e}")
        await update.message.reply_text(f"❌ Hata: {e}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = query.from_user.id

    if data == "summary":
        response = await asistan._handle_summary(user_id)
        await query.edit_message_text(response, parse_mode='Markdown')
    elif data == "list":
        response = await asistan._handle_reminder_list(user_id)
        await query.edit_message_text(response, parse_mode='Markdown')
    else:
        prompts = {
            "note": "📝 Notunuzu yazın...",
            "search": "🔝 Aramak istediğinizi yazın...",
            "remind": "⏰ Örnek: \"Yarın saat 10'da toplantıyı hatırlat\""
        }
        await query.edit_message_text(prompts.get(data, "..."))


async def reminder_job(context):
    sent = await asistan.check_reminders(context.bot)
    if sent > 0:
        logger.info(f"{sent} hatırlatıcı gönderildi")


# ==================== MAIN ====================
def main():
    app = Application.builder().token(config.telegram_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(CallbackQueryHandler(button_callback))

    job_queue = app.job_queue
    if job_queue:
        job_queue.run_repeating(reminder_job, interval=60, first=10)

    logger.info("=" * 50)
    logger.info("Asistan v4.1 Başlatılıyor...")
    logger.info(f"Hafıza: {config.memory_path}")
    logger.info(f"Arşiv: {config.archive_path}")
    logger.info(f"Terminal: Güvenli (shell=False)")
    logger.info(f"Arşivci: {'Aktif' if asistan.archivist.available else 'Pasif'}")
    logger.info("=" * 50)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
