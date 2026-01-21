#!/usr/bin/env python3
"""
Telegram Bot Asistan v4.2 - TAM ÜCRETSİZ MİMARİ
- Groq (Llama 3.3) - Ana Agent - ÜCRETSİZ
- Ollama (GLM 4) - Yerel Arşivci - ÜCRETSİZ
- ChromaDB - Vektör Hafıza - ÜCRETSİZ
- Hierarchical RAG - 4 Seviye
- Otomatik Temizlik - Maintenance Jobs
- Güvenli Terminal (shell=False)
"""

import os
import sys
import json
import asyncio
import logging
import shlex
import subprocess
import requests
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
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

# Logging
logging.basicConfig(
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('asistant_v42.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==================== CONFIG ====================
@dataclass
class Config:
    telegram_token: str
    groq_key: str

    # Ollama config
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "glm4"  # veya "glm-4.7"

    # Yollar
    timezone: str = "Europe/Istanbul"
    hot_memory_path: str = None      # ChromaDB (7 gün)
    warm_archive_path: str = None    # JSON (30 gün)
    cold_archive_path: str = None    # JSON (aylık)
    deep_archive_path: str = None    # Topic-based (sonsuz)
    notes_path: str = None

    # LLM Ayarları
    groq_model: str = "llama-3.3-70b-versatile"
    groq_flash: str = "llama-3.1-8b-instant"

    # Retention politikası
    hot_days: int = 7
    warm_days: int = 30
    max_hot_notes: int = 500  # ChromaDB limiti

    # Güvenli terminal
    safe_commands: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.hot_memory_path:
            self.hot_memory_path = str(Path.home() / "asistant_v42_hot")
        if not self.warm_archive_path:
            self.warm_archive_path = str(Path.home() / "asistant_v42_warm")
        if not self.cold_archive_path:
            self.cold_archive_path = str(Path.home() / "asistant_v42_cold")
        if not self.deep_archive_path:
            self.deep_archive_path = str(Path.home() / "asistant_v42_deep")
        if not self.notes_path:
            self.notes_path = str(Path.home() / "asistant_v42_notes")

        # Güvenli komutlar
        self.safe_commands = {
            'dir': None, 'ls': ['-la', '-a', '-l', '-h'], 'pwd': None,
            'date': None, 'time': None, 'whoami': None, 'hostname': None,
            'echo': None, 'cat': None, 'head': ['-n'], 'tail': ['-n'],
            'wc': None, 'tree': None, 'find': ['.']
        }


config = Config(
    telegram_token=os.getenv("TELEGRAM_TOKEN", "8449158473:AAG-3HbGmY2740CdrAnS1SAzw4Hnyp3DAB0"),
    groq_key=os.getenv("GROQ_API_KEY", "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"),
)


# ==================== OLLAMA CLIENT (YEREL, ÜCRETSİZ) ====================
class OllamaClient:
    """
    Ollama - Yerel LLM, ücretsiz
    GLM 4 veya başka modeller çalıştırır
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.available = False

        # Ollama çalışıyor mu kontrol et
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                self.available = True
                logger.info(f"Ollama aktif: {model}")
        except:
            logger.warning("Ollama çalışmıyor. Yerel arşivci pasif.")

    def generate(self, prompt: str, system: str = None) -> str:
        """Ücretsiz yerel LLM çağrısı"""
        if not self.available:
            return None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt if not system else f"{system}\n\n{prompt}",
                    "stream": False,
                    "options": {"num_ctx": 4096}  # 4K context
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json().get("response", "")

        except Exception as e:
            logger.error(f"Ollama hatası: {e}")

        return None

    def is_available(self) -> bool:
        return self.available


# ==================== SECURE TERMINAL ====================
class SecureTerminal:
    """Güvenli terminal - shell=False + beyaz liste"""

    ALLOWED = {
        'dir': None, 'ls': ['-la', '-a', '-l', '-h'], 'pwd': None,
        'date': None, 'time': None, 'whoami': None, 'hostname': None,
        'echo': None, 'cat': None, 'head': ['-n'], 'tail': ['-n'],
        'wc': None, 'tree': None, 'find': ['.']
    }

    def __init__(self, safe_commands: Dict = None):
        self.allowed = safe_commands or self.ALLOWED

    def is_safe(self, command: str) -> bool:
        parts = shlex.split(command.strip())
        if not parts:
            return False
        cmd = parts[0]

        if cmd not in self.allowed:
            return False

        allowed_params = self.allowed[cmd]
        if allowed_params is None:
            return len(parts) == 1

        for param in parts[1:]:
            if not any(param.startswith(a) for a in allowed_params):
                return False
        return True

    def execute(self, command: str) -> Dict[str, Any]:
        try:
            parts = shlex.split(command.strip())
            if not parts or not self.is_safe(command):
                return {
                    "success": False,
                    "error": f"❌ Yasaklı: {parts[0] if parts else ''}\n\nİzin: {', '.join(list(self.allowed.keys())[:5])}..."
                }

            cmd = parts[0]
            allowed_params = self.allowed[cmd]
            args = [p for p in parts[1:] if allowed_params and any(p.startswith(a) for a in allowed_params)]

            result = subprocess.run(
                [cmd] + args,
                shell=False,  # GÜVENLİK
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10,
                cwd=str(Path.home())
            )

            return {
                "success": True,
                "output": (result.stdout or result.stderr).strip() or "✅ Tamam",
                "command": cmd
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "⏱️ Zaman aşımı"}
        except Exception as e:
            return {"success": False, "error": f"❌ Hata: {e}"}


# ==================== HIERARCHICAL MEMORY ====================
class HierarchicalMemory:
    """
    4 Seviyeli Hiyerarşik Hafıza:
    - L1: Sıcak (ChromaDB, 7 gün)
    - L2: Ilık (JSON günlük özetleri, 30 gün)
    - L3: Soğuk (JSON haftalık özetleri, 1 yıl)
    - L4: Arşiv (JSON aylık analiz, sonsuz)
    """

    def __init__(self, hot_path: str, warm_path: str, cold_path: str):
        self.hot_path = Path(hot_path)
        self.warm_path = Path(warm_path)
        self.cold_path = Path(cold_path)

        # Dizinleri oluştur
        for p in [self.hot_path, self.warm_path, self.cold_path]:
            p.mkdir(parents=True, exist_ok=True)

        # ChromaDB (Sıcak)
        try:
            self.chroma = chromadb.PersistentClient(
                path=str(self.hot_path),
                settings=Settings(anonymized_telemetry=False)
            )
            self.notes = self.chroma.get_or_create_collection("notes")
            self.summaries = self.chroma.get_or_create_collection("summaries")
            self.hot_available = True
            logger.info("ChromaDB (Sıcak hafıza) aktif")
        except Exception as e:
            logger.error(f"ChromaDB hatası: {e}")
            self.hot_available = False

        # Yerel LLM (Ollama)
        self.ollama = OllamaClient(config.ollama_base_url, config.ollama_model)

        # Indeks dosyaları
        self.warm_index = self.warm_path / "daily_summaries.json"
        self.cold_index = self.cold_path / "weekly_summaries.json"

        self._load_indices()

    def _load_indices(self):
        """İndeks dosyalarını yükle"""
        self.daily_summaries = {}
        self.weekly_summaries = {}

        if self.warm_index.exists():
            try:
                self.daily_summaries = json.loads(self.warm_index.read_text(encoding='utf-8'))
            except:
                pass

        if self.cold_index.exists():
            try:
                self.weekly_summaries = json.loads(self.cold_index.read_text(encoding='utf-8'))
            except:
                pass

    def _save_indices(self):
        """İndeks dosyalarını kaydet"""
        self.warm_index.write_text(
            json.dumps(self.daily_summaries, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )
        self.cold_index.write_text(
            json.dumps(self.weekly_summaries, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )

    # ==================== NOT YÖNETİMİ ====================
    def add_note(self, user_id: int, text: str, tags: List[str] = None) -> str:
        """Not ekle - Sıcak hafızaya"""
        if not self.hot_available:
            return None

        note_id = f"note_{user_id}_{datetime.now().timestamp()}"
        metadata = {
            "user_id": str(user_id),
            "created": datetime.now().isoformat(),
            "tags": tags or []
        }

        try:
            self.notes.add(
                ids=[note_id],
                documents=[text],
                metadatas=[metadata]
            )
            return note_id
        except Exception as e:
            logger.error(f"Not ekleme hatası: {e}")
            return None

    def search_hot(self, user_id: int, query: str, n: int = 5) -> List[Dict]:
        """Sıcak hafızada ara - ChromaDB vektör"""
        if not self.hot_available:
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
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'level': 'hot'
                    })
            return items
        except Exception as e:
            logger.error(f"Sıcak arama hatası: {e}")
            return []

    def search_warm(self, user_id: int, query: str, n: int = 3) -> List[Dict]:
        """Ilık hafızada ara - Günlük özetleri"""
        results = []
        query_lower = query.lower()

        for date, summary in self.daily_summaries.items():
            if summary.get('user_id') != user_id:
                continue

            # Basit metin arama
            summary_text = f"{summary.get('summary', '')} {' '.join(summary.get('topics', []))}"
            if query_lower in summary_text.lower():
                results.append({
                    'type': 'daily',
                    'date': date,
                    'summary': summary,
                    'level': 'warm'
                })

        return results[:n]

    def search_cold(self, user_id: int, query: str, n: int = 3) -> List[Dict]:
        """Soğuk hafızada ara - Haftalık özetleri"""
        results = []
        query_lower = query.lower()

        for week, summary in self.weekly_summaries.items():
            if summary.get('user_id') != user_id:
                continue

            summary_text = f"{summary.get('summary', '')} {' '.join(summary.get('topics', []))}"
            if query_lower in summary_text.lower():
                results.append({
                    'type': 'weekly',
                    'week': week,
                    'summary': summary,
                    'level': 'cold'
                })

        return results[:n]

    def search_deep(self, user_id: int, query: str, n: int = 3) -> List[Dict]:
        """Derin arşivde ara - Topic-based (GLM 4 özetleri)"""
        results = []
        query_lower = query.lower()

        deep_path = Path(self.cold_path).parent / "asistant_v42_deep"
        index_file = deep_path / "archive_index.json"

        if not index_file.exists():
            return results

        try:
            index = json.loads(index_file.read_text(encoding='utf-8'))

            for topic_id, topic in index.get("topics", {}).items():
                if query_lower in topic.get("name", "").lower():
                    results.append({
                        'type': 'topic',
                        'id': topic_id,
                        'name': topic.get("name", ""),
                        'summary': topic.get("summary", ""),
                        'level': 'deep'
                    })
                    continue

                summary = topic.get("summary", "")
                keywords = topic.get("keywords", [])

                if query_lower in summary.lower() or any(query_lower in kw.lower() for kw in keywords):
                    results.append({
                        'type': 'topic',
                        'id': topic_id,
                        'name': topic.get("name", ""),
                        'summary': summary,
                        'level': 'deep'
                    })

                if len(results) >= n:
                    break
        except Exception as e:
            logger.error(f"Derin arama hatası: {e}")

        return results

    # ==================== HIERARCHICAL RETRIEVAL ====================
    async def retrieve(self, user_id: int, query: str) -> Tuple[List[Dict], str]:
        """
        Geri çağırma protokolü - Seviyeli arama

        Returns:
            (results, level) - results ve en düşük seviye
        """
        logger.info(f"Retrieval: user={user_id}, query='{query[:30]}...'")

        # Seviye 1: Sıcak hafıza (ChromaDB - son 7 gün)
        hot_results = self.search_hot(user_id, query, n=5)
        if hot_results:
            logger.info(f"  -> Sıcak hafızada {len(hot_results)} sonuç")
            return hot_results, 'hot'

        # Seviye 2: Ilık hafıza (Günlük özetler)
        warm_results = self.search_warm(user_id, query, n=3)
        if warm_results:
            logger.info(f"  -> Ilık hafızada {len(warm_results)} sonuç")
            return warm_results, 'warm'

        # Seviye 3: Soğuk hafıza (Haftalık özetler)
        cold_results = self.search_cold(user_id, query, n=3)
        if cold_results:
            logger.info(f"  -> Soğuk hafızada {len(cold_results)} sonuç")
            return cold_results, 'cold'

        # Seviye 4: Derin arşiv (Topic-based - GLM 4 özetleri)
        deep_results = self.search_deep(user_id, query, n=3)
        if deep_results:
            logger.info(f"  -> Derin arşivde {len(deep_results)} sonuç")
            return deep_results, 'deep'

        logger.info("  -> Sonuç bulunamadı")
        return [], 'none'
        if warm_results:
            logger.info(f"  -> Ilık hafızada {len(warm_results)} sonuç")
            return warm_results, 'warm'

        # Seviye 3: Soğuk hafıza (Haftalık özetler)
        cold_results = self.search_cold(user_id, query, n=3)
        if cold_results:
            logger.info(f"  -> Soğuk hafızada {len(cold_results)} sonuç")
            return cold_results, 'cold'

        logger.info("  -> Sonuç bulunamadı")
        return [], 'none'

    async def retrieve_with_expansion(self, user_id: int, query: str) -> List[Dict]:
        """
        Genişletilmiş geri çağırma - Özetlerden detayları getir
        """
        results, level = await self.retrieve(user_id, query)

        if not results:
            return []

        # Ilık/Soğuk seviyedeyse, detayları genişlet
        if level in ['warm', 'cold']:
            expanded = []
            for r in results:
                if r['type'] == 'daily':
                    # O günün tüm notlarını yükle
                    day_notes = self._load_day_notes(r['date'], user_id)
                    expanded.extend(day_notes)
                elif r['type'] == 'weekly':
                    # O haftanın günlük özetlerini yükle
                    daily_summaries = r['summary'].get('daily_summaries', {})
                    for date, ds in daily_summaries.items():
                        day_notes = self._load_day_notes(date, user_id)
                        expanded.extend(day_notes)

                if expanded:
                    return expanded[:10]  # Max 10 detay

            return results

        return results

    def _load_day_notes(self, date: str, user_id: int) -> List[Dict]:
        """Belirli bir günün notlarını yükle"""
        daily_file = self.warm_path / f"day_{date}_user_{user_id}.json"

        if daily_file.exists():
            try:
                data = json.loads(daily_file.read_text(encoding='utf-8'))
                return data.get('notes', [])
            except:
                return []
        return []

    # ==================== ARŞİVLEME ====================
    async def create_daily_summary(self, user_id: int) -> Dict:
        """Günlük özet oluştur"""
        now = datetime.now(pytz.timezone(config.timezone))
        date_str = now.strftime("%Y-%m-%d")

        # O günün notlarını al
        day_start = now.replace(hour=0, minute=0, second=0)
        day_end = day_start + timedelta(days=1)

        # ChromaDB'den günün notlarını al
        if not self.hot_available:
            return {"error": "ChromaDB yok"}

        try:
            results = self.notes.get(
                where={"user_id": str(user_id)},
                limit=1000,
                include=["documents", "metadatas"]
            )

            # Günün notlarını filtrele
            day_notes = []
            for doc, meta in zip(results['documents'], results['metadatas']):
                created = meta.get('created', '')
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created)
                        if day_start <= created_dt < day_end:
                            day_notes.append({
                                'content': doc,
                                'created': created_dt
                            })
                    except:
                        pass

            if not day_notes:
                return {"date": date_str, "count": 0, "summary": "Not yok"}

            # Özet oluştur
            # 1. Ollama ile (ücretsiz)
            notes_text = "\n".join([f"- [{n['created'].strftime('%H:%M')}] {n['content']}" for n in day_notes])

            prompt = f"""Bu günün notları:
{notes_text}

JSON formatında özetle:
{{
    "summary": "Kısa özet (1-2 cümle)",
    "topics": ["konu1", "konu2"],
    "count": {len(day_notes)},
    "mood": "pozitif/nötr/negatif"
}}"""

            summary_text = self.ollama.generate(prompt)

            if summary_text:
                try:
                    if "```json" in summary_text:
                        summary_text = summary_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in summary_text:
                        summary_text = summary_text.split("```")[1].split("```")[0].strip()
                    summary_data = json.loads(summary_text)
                except:
                    summary_data = {"summary": summary_text[:500], "topics": [], "mood": "nötr"}
            else:
                summary_data = {"summary": "Özet oluşturulamadı", "topics": [], "mood": "nötr"}

            # Özeti kaydet
            daily_summary = {
                "date": date_str,
                "user_id": user_id,
                "note_count": len(day_notes),
                "summary": summary_data.get("summary", ""),
                "topics": summary_data.get("topics", []),
                "mood": summary_data.get("mood", "nötr"),
                "created": now.isoformat()
            }

            self.daily_summaries[date_str] = daily_summary
            self._save_indices()

            # Detaylı dosyaya kaydet
            daily_file = self.warm_path / f"day_{date_str}_user_{user_id}.json"
            daily_file.write_text(
                json.dumps({
                    "date": date_str,
                    "user_id": user_id,
                    "notes": day_notes,
                    "summary": daily_summary
                }, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )

            logger.info(f"Günlük özet oluşturuldu: {date_str} ({len(day_notes)} not)")

            return daily_summary

        except Exception as e:
            logger.error(f"Günlük özet hatası: {e}")
            return {"error": str(e)}

    async def create_weekly_summary(self, user_id: int) -> Dict:
        """Haftalık özet oluştur"""
        now = datetime.now(pytz.timezone(config.timezone))
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0)
        week_id = now.strftime("%Y-W%U")

        # Son 7 günlük günlük özetleri al
        daily_summaries_data = []
        for i in range(7):
            day = week_start + timedelta(days=i)
            date_str = day.strftime("%Y-%m-%d")
            if date_str in self.daily_summaries:
                daily_summaries_data.append(self.daily_summaries[date_str])

        if not daily_summaries_data:
            return {"week": week_id, "count": 0, "summary": "Bu hafta not yok"}

        # Haftalık özet oluştur
        summaries_text = "\n".join([
            f"{d['date']}: {d.get('summary', '')} ({d.get('note_count', 0)} not)"
            for d in daily_summaries_data
        ])

        prompt = f"""Son haftanın günlük özetleri:
{summaries_text}

JSON formatında haftalık özet:
{{
    "summary": "Haftanın genel özeti",
    "key_topics": ["konu1", "konu2"],
    "total_notes": 15,
    "productivity_score": 7,
    "trend": "artıyor/azalıyor/stabil"
}}"""

        summary_text = self.ollama.generate(prompt)

        if summary_text:
            try:
                if "```json" in summary_text:
                    summary_text = summary_text.split("```json")[1].split("```")[0].strip()
                elif "```" in summary_text:
                    summary_text = summary_text.split("```")[1].split("```")[0].strip()
                summary_data = json.loads(summary_text)
            except:
                summary_data = {
                    "summary": summary_text[:500],
                    "key_topics": [],
                    "trend": "stabil",
                    "productivity_score": 5
                }
        else:
            summary_data = {"summary": "Özet oluşturulamadı"}

        # Haftalık özeti kaydet
        weekly_summary = {
            "week": week_id,
            "user_id": user_id,
            "week_start": week_start.isoformat(),
            "daily_summaries": {d['date']: d for d in daily_summaries_data},
            "summary": summary_data,
            "created": now.isoformat()
        }

        self.weekly_summaries[week_id] = weekly_summary
        self._save_indices()

        # Detaylı dosya
        weekly_file = self.cold_path / f"week_{week_id}_user_{user_id}.json"
        weekly_file.write_text(
            json.dumps(weekly_summary, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )

        logger.info(f"Haftalık özet oluşturuldu: {week_id}")

        return weekly_summary

    # ==================== TEMİZLİK ====================
    async def cleanup_hot_to_warm(self, user_id: int) -> int:
        """7 günden eski notları Ilık arşive taşı"""
        if not self.hot_available:
            return 0

        cutoff = datetime.now(pytz.timezone(config.timezone)) - timedelta(days=config.hot_days)
        moved_count = 0

        try:
            results = self.notes.get(
                where={"user_id": str(user_id)},
                limit=10000,
                include=["documents", "metadatas", "ids"]
            )

            ids_to_delete = []
            for doc, meta, doc_id in zip(results['documents'], results['metadatas'], results['ids']):
                created = meta.get('created', '')
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created)
                        if created_dt < cutoff:
                            # Önce günlük özet oluştur
                            await self.create_daily_summary(user_id)
                            ids_to_delete.append(doc_id)
                            moved_count += 1
                    except:
                        pass

            # ChromaDB'den sil
            if ids_to_delete:
                self.notes.delete(ids=ids_to_delete)
                logger.info(f"{len(ids_to_delete)} not sıcaktan ılığa taşındı")

        except Exception as e:
            logger.error(f"Temizlik hatası: {e}")

        return moved_count

    async def cleanup_warm_to_cold(self, user_id: int) -> int:
        """30 günden eski günlükleri soğuk arşive taşı"""
        moved_count = 0
        cutoff = datetime.now() - timedelta(days=config.warm_days)

        for date_str, summary in list(self.daily_summaries.items()):
            if summary.get('user_id') != user_id:
                continue

            try:
                summary_date = datetime.fromisoformat(summary.get('created', ''))
                if summary_date < cutoff:
                    # Haftalık özet oluştur
                    await self.create_weekly_summary(user_id)

                    # Günlük özeti sil
                    del self.daily_summaries[date_str]
                    moved_count += 1

                    # Dosyayı da sil
                    daily_file = self.warm_path / f"day_{date_str}_user_{user_id}.json"
                    if daily_file.exists():
                        daily_file.unlink()
            except:
                pass

        if moved_count > 0:
            self._save_indices()
            logger.info(f"{moved_count} günlük ılıktan soğua taşındı")

        return moved_count

    # ==================== SOHBET ====================
    def add_chat(self, user_id: int, role: str, text: str):
        """Sohbet geçmişine ekle"""
        if not self.hot_available:
            return

        try:
            msg_id = f"chat_{user_id}_{datetime.now().timestamp()}"
            self.summaries.add(
                ids=[msg_id],
                documents=[text],
                metadatas={
                    "user_id": str(user_id),
                    "role": role,
                    "type": "chat",
                    "created": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Sohbet kayıt hatası: {e}")

    def get_chat_context(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Sohbet bağlamını getir"""
        if not self.hot_available:
            return []

        try:
            results = self.summaries.get(
                where={"user_id": str(user_id), "type": "chat"},
                limit=limit * 2,
                include=["documents", "metadatas"]
            )

            messages = []
            if results['documents']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    messages.append({
                        'role': meta.get('role', 'user'),
                        'content': doc
                    })
            return messages[-limit:]
        except Exception as e:
            logger.error(f"Context alma hatası: {e}")
            return []


# ==================== GROQ AGENT ====================
class GroqAgent:
    """Groq (Llama 3.3) - Ücretsiz Ana Agent"""

    SYSTEM = """Sen AI Asistan Ajanısın.

Kategoriler:
- "otomasyon": Komut çalıştırma
- "bilgi": Not kaydetme
- "bilgi_ara": Notlarda arama
- "iletisim": Mesaj taslağı
- "hatirlatma": Hatırlatıcı ekleme
- "hatirlatma_liste": Listeleme
- "ozet": Haftalık özet

JSON: {"kategori": "...", "icerik": "..."}"""

    def __init__(self, api_key: str, model: str, flash: str):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.flash = flash

    def categorize(self, text: str) -> Dict:
        """Kategorize et"""
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": text}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.flash,
                messages=messages,
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except:
            # Fallback
            text_lower = text.lower()
            if any(k in text_lower for k in ["çalıştır", "komut", "run"]):
                return {"kategori": "otomasyon", "icerik": text}
            elif any(k in text_lower for k in ["not al", "kaydet"]):
                return {"kategori": "bilgi", "icerik": text}
            elif any(k in text_lower for k in ["ne not", "bul", "ara", "hatırla"]):
                return {"kategori": "bilgi_ara", "icerik": text}
            elif any(k in text_lower for k in ["mail", "mesaj at"]):
                return {"kategori": "iletisim", "icerik": text}
            elif "hatırlat" in text_lower or "hatirlatıcı" in text_lower:
                return {"kategori": "hatirlatma_liste", "icerik": text}
            elif "ozet" in text_lower or "haftalık" in text_lower:
                return {"kategori": "ozet", "icerik": text}
            else:
                return {"kategori": "sohbet", "icerik": text}

    def chat(self, messages: List[Dict]) -> str:
        """Genel sohbet"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Sohbet hatası: {e}")
            return "Bir hata oluştu."


# ==================== TTS/STT ====================
class TTSEngine:
    def __init__(self):
        self.available = False
        try:
            import edge_tts
            self.edge_tts = edge_tts
            self.available = True
        except:
            pass


class STTEngine:
    def __init__(self):
        self.available = False
        try:
            import whisper
            self.model = whisper.load_model("base")
            self.available = True
        except:
            pass


# ==================== REMINDER ====================
class ReminderSystem:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        self.file = self.path / "reminders.json"
        self.tz = pytz.timezone(config.timezone)
        self.reminders = self._load()

    def _load(self):
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

    def add(self, user_id, message, when: datetime):
        rem_id = f"rem_{user_id}_{datetime.now().timestamp()}"
        self.reminders.append({
            "id": rem_id,
            "user_id": user_id,
            "message": message,
            "when": when.isoformat(),
            "sent": False
        })
        self._save()
        return rem_id

    def list_user(self, user_id):
        return [r for r in self.reminders if r['user_id'] == user_id and not r['sent']]

    def get_due(self):
        now = datetime.now(self.tz)
        return [r for r in self.reminders if not r['sent'] and datetime.fromisoformat(r['when']) <= now]

    def mark_sent(self, rem_id):
        for r in self.reminders:
            if r['id'] == rem_id:
                r['sent'] = True
                self._save()
                break


# ==================== MAIN ASSISTANT ====================
class AsistanV42:
    """Asistan v4.2 - Tam Ücretsiz Mimari"""

    def __init__(self, config: Config):
        self.config = config

        # Modüller
        self.memory = HierarchicalMemory(
            config.hot_memory_path,
            config.warm_archive_path,
            config.cold_archive_path
        )
        self.agent = GroqAgent(config.groq_key, config.groq_model, config.groq_flash)
        self.terminal = SecureTerminal(config.safe_commands)
        self.reminders = ReminderSystem(config.reminder_path if hasattr(config, 'reminder_path') else str(Path.home() / "asistant_v42_reminders"))
        self.tts = TTSEngine()
        self.stt = STTEngine()

        self.notes_dir = Path(config.notes_path)
        self.notes_dir.mkdir(exist_ok=True)

    async def process(self, text: str, user_id: int) -> str:
        """Ana işlem"""
        try:
            result = self.agent.categorize(text)
            category = result.get("kategori", "sohbet")
            content = result.get("icerik", text)

            logger.info(f"User {user_id} | {category} | {text[:40]}...")

            if category == "otomasyon":
                return await self._automation(content, user_id)
            elif category == "bilgi":
                return await self._note(content, user_id)
            elif category == "bilgi_ara":
                return await self._search(content, user_id)
            elif category == "iletisim":
                return await self._email(content, user_id)
            elif category == "hatirlatma":
                return await self._reminder_add(content, user_id)
            elif category == "hatirlatma_liste":
                return await self._reminder_list(user_id)
            elif category == "ozet":
                return await self._summary(user_id)
            else:
                return await self._chat(text, user_id)

        except Exception as e:
            logger.error(f"İşlem hatası: {e}")
            return f"⚠️ Hata: {str(e)[:50]}"

    async def _automation(self, text, user_id):
        result = self.terminal.execute(text.strip().split()[-1])
        if result.get("success"):
            return f"🖥️ **Çıktı:**\n```\n{result['output']}\n```"
        return result.get("error", "Hata")

    async def _note(self, text, user_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.notes_dir / f"note_{user_id}_{timestamp}.md"
        filename.write_text(f"# {timestamp}\n\n{text}\n\n*Kaydedilme: {datetime.now()}*", encoding='utf-8')
        self.memory.add_note(user_id, text)
        return f"📝 Kaydedildi: `{filename.name}`"

    async def _search(self, text, user_id):
        # Hierarchical retrieval
        results, level = await self.memory.retrieve(user_id, text)

        if not results:
            return "🔍 Sonuç bulunamadı."

        response = f"🔍 **Bulunanlar ({level.upper()}):**\n\n"
        for r in results[:5]:
            if 'content' in r:
                response += f"• {r['content'][:80]}...\n"
            elif 'summary' in r:
                summary = r.get('summary', {})
                response += f"📅 {r.get('date', r.get('week', '?'))}: {summary.get('summary', '')[:60]}...\n"

        return response

    async def _email(self, text, user_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.notes_dir / f"email_{user_id}_{timestamp}.md"
        filename.write_text(f"# E-posta Taslağı\n\n**Konu:** [Konu]\n**Alıcı:** [Alıcı]\n\n---\n\n{text}\n\n---", encoding='utf-8')
        return f"📧 Taslak: `{filename.name}`"

    async def _reminder_add(self, text, user_id):
        now = datetime.now(pytz.timezone(config.timezone))
        reminder_time = now.replace(hour=9, minute=0) + timedelta(days=1)
        message = text
        for w in ["hatırlat", "yarın", "saat"]:
            message = message.replace(w, "")
        message = message.strip() or "Hatırlatıcı"
        self.reminders.add(user_id, message, reminder_time)
        return f"⏰ Hatırlatıcı: {reminder_time.strftime('%d.%m.%H:%M')}\n\n📝 {message}"

    async def _reminder_list(self, user_id):
        reminders = self.reminders.list_user(user_id)
        if not reminders:
            return "📋 Aktif hatırlatıcın yok."
        response = "📋 **Hatırlatıcılar:**\n\n"
        for r in reminders[:10]:
            when = datetime.fromisoformat(r['when'])
            response += f"{when.strftime('%d.%m %H:%M')} - {r['message'][:40]}\n"
        return response

    async def _summary(self, user_id):
        summary = await self.memory.create_weekly_summary(user_id)
        if "error" in summary:
            return f"📊 Hata: {summary['error']}"

        s = summary.get('summary', {})
        response = f"📊 **Haftalık Özet ({summary.get('week', '?')}**\n\n"
        response += f"📝 {summary.get('total_notes', 0)} not\n"
        response += f"📋 Özet: {s.get('summary', '')}\n"
        if s.get('key_topics'):
            response += f"\n🏷️ Konular: {', '.join(s.get('key_topics', [])[:5])}"
        return response

    async def _chat(self, text, user_id):
        self.memory.add_chat(user_id, "user", text)
        context = self.memory.get_chat_context(user_id, limit=6)

        messages = [{"role": "system", "content": "Sen yardımcı Türkçe asistanısın. Kısa cevap ver."}]
        for m in context:
            messages.append({"role": m['role'], "content": m['content']})
        if not context or context[-1]['content'] != text:
            messages.append({"role": "user", "content": text})

        reply = self.agent.chat(messages)
        self.memory.add_chat(user_id, "assistant", reply)
        return reply

    async def check_reminders(self, bot) -> int:
        sent = 0
        for rem in self.reminders.get_due():
            try:
                await bot.send_message(
                    chat_id=rem['user_id'],
                    text=f"🔔 **HATIRLATMA**\n\n{rem['message']}",
                    parse_mode='Markdown'
                )
                self.reminders.mark_sent(rem['id'])
                sent += 1
            except:
                pass
        return sent

    async def maintenance_job(self, bot=None):
        """Bakım job'u - Temizlik ve arşivleme"""
        logger.info("=== Bakım job başladı ===")

        # 1. Sıcaktan ılığa taşı
        # 2. Ilıktan soğuga taşı

        # Tüm kullanıcılar için
        # Not: Şimdilik tek kullanıcı olduğu varsayımıyla
        user_id = 0  # Veya mevcut kullanıcıları takip et

        # Temizlik
        moved_hot = await self.memory.cleanup_hot_to_warm(user_id)
        moved_warm = await self.memory.cleanup_warm_to_cold(user_id)

        logger.info(f"=== Bakım tamam: {moved_hot} sıcak->ılık, {moved_warm} ılık->soğuk ===")


# ==================== TELEGRAM HANDLERS ====================
asistan = AsistanV42(config)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📝 Not", callback_data="note"),
         InlineKeyboardButton("🔍 Ara", callback_data="search")],
        [InlineKeyboardButton("⏰ Hatırlat", callback_data="remind"),
         InlineKeyboardButton("📋 Listem", callback_data="list")],
        [InlineKeyboardButton("📊 Özet", callback_data="summary")],
    ]
    await update.message.reply_text(
        "🤖 **Asistan v4.2** - Tam Ücretsiz\n\n"
        "⚡ Groq (Llama 3.3)\n"
        "🦙 Ollama (GLM 4)\n"
        "🧠 ChromaDB Hafıza\n"
        "📊 Hierarchical RAG\n"
        "🔒 Güvenli Terminal\n\n"
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
        await update.message.reply_text(f"⚠️ {str(e)[:100]}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("typing")
    user_id = update.effective_user.id

    try:
        voice = await update.message.voice.get_file()
        temp_path = f"voice_{user_id}.ogg"
        await voice.download_to_drive(temp_path)

        if asistan.stt.available:
            await update.message.reply_text("🎤 Çevriliyor...")
            text = asistan.stt.model.transcribe(temp_path, language="tr")
            os.remove(temp_path)

            if text:
                response = await asistan.process(text, user_id)
                await update.message.reply_text(f"🎤 \"{text}\"\n\n{response}", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Çevrilemedi")
        else:
            await update.message.reply_text("🎤 Whisper yüklü değil:\n`pip install openai-whisper`")
    except Exception as e:
        await update.message.reply_text(f"❌ {str(e)[:100]}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    user_id = query.from_user.id

    if data == "summary":
        response = await asistan._summary(user_id)
        await query.edit_message_text(response, parse_mode='Markdown')
    elif data == "list":
        response = await asistan._reminder_list(user_id)
        await query.edit_message_text(response, parse_mode='Markdown')
    else:
        prompts = {
            "note": "📝 Notunuzu yazın...",
            "search": "🔝 Aramak istediğiniz...",
            "remind": "⏰ Örnek: \"Yarın saat 10'da toplantıyı hatırlat\""
        }
        await query.edit_message_text(prompts.get(data, "..."))


async def reminder_job(context):
    sent = await asistan.check_reminders(context.bot)
    if sent > 0:
        logger.info(f"{sent} hatırlatıcı gönderildi")


async def maintenance_job(context):
    """Haftalık bakım job'u"""
    await asistan.maintenance_job()


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
        # Haftalık bakım - her gün 02:00 (otomatik temizlik)
        job_queue.run_daily(maintenance_job, time=time(2, 0))

    logger.info("=" * 50)
    logger.info("Asistan v4.2 Başlatılıyor...")
    logger.info(f"Sıcak: {config.hot_memory_path}")
    logger.info(f"Ilık: {config.warm_archive_path}")
    logger.info(f"Soğuk: {config.cold_archive_path}")
    logger.info(f"Ollama: {config.ollama_base_url} ({config.ollama_model})")
    logger.info("=" * 50)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
