#!/usr/bin/env python3
"""
Kalıcı Arşiv Hafıza Yönetim Sistemi
Tamamen yerel GLM 4 tabanlı semantik arşivleme

01: Yerel Arşivleme Protokolü - Semantik Özet
02: Bağlamsal Sıkıştırma - Topic Clustering
03: Hibrit Sorgulama - Multi-level Retrieval
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
import requests

# ChromaDB
import chromadb
from chromadb.config import Settings

logging.basicConfig(
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ==================== CONFIG ====================
@dataclass
class MemoryConfig:
    # Ollama (GLM 4)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "glm4"

    # Yollar
    hot_memory_path: str = None   # ChromaDB (7 gün)
    warm_archive_path: str = None # Günlük özetler (30 gün)
    cold_archive_path: str = None # Haftalık özetler (1 yıl)
    deep_archive_path: str = None # Topic-based arşiv (sonsuz)

    # Retention
    hot_days: int = 7
    warm_days: int = 30
    cold_months: int = 12

    # Arşivleme
    archive_time: str = "02:00"
    max_topics_per_week: int = 8
    min_notes_per_topic: int = 2

    def __post_init__(self):
        base = Path.home()
        if not self.hot_memory_path:
            self.hot_memory_path = str(base / "memory_hot")
        if not self.warm_archive_path:
            self.warm_archive_path = str(base / "memory_warm")
        if not self.cold_archive_path:
            self.cold_archive_path = str(base / "memory_cold")
        if not self.deep_archive_path:
            self.deep_archive_path = str(base / "memory_deep")


config = MemoryConfig()


# ==================== OLLAMA CLIENT ====================
class OllamaClient:
    """Yerel GLM 4 istemcisi - semantik işlemler için"""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if r.status_code == 200:
                logger.info(f"Ollama aktif: {self.model}")
                return True
        except:
            pass
        logger.warning("Ollama aktif değil")
        return False

    def generate(self, prompt: str, system: str = None, json_mode: bool = False) -> Optional[str]:
        """GLM 4 ile metin üret"""
        if not self.available:
            return None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_ctx": 8192}  # 8K context
                },
                timeout=180
            )

            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama hatası: {e}")

        return None

    def extract_json(self, text: str) -> Optional[Dict]:
        """Metinden JSON çıkar"""
        if not text:
            return None

        # JSON bloğunu bul
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except:
            return None


# ==================== SEMANTIC SUMMARIZER ====================
class SemanticSummarizer:
    """
    GLM 4 tabanlı semantik özetleyici
    - Haftalık notları konulara göre kümele
    - Her konu için özet üret
    - Semantik anahtar kelimeler çıkar
    """

    CLUSTERING_PROMPT = """Bu haftanın notlarını ana konulara göre kümele.

Notlar:
{notes}

JSON formatında yanıtla (sadece JSON):
{{
    "topics": [
        {{
            "name": "Konu adı (kısa)",
            "summary": "Bu konudaki notların 2-3 cümlelik özeti",
            "keywords": ["kelime1", "kelime2", "kelime3"],
            "sentiment": "pozitif/nötr/negatif",
            "importance": "yüksek/orta/düşük",
            "note_count": 5
        }}
    ],
    "week_summary": "Haftanın genel 1 cümlelik özeti"
}}"""

    TOPIC_SUMMARY_PROMPT = """Bu konuyla ilgili notları özetle:

Konu: {topic_name}
Notlar:
{notes}

JSON formatında:
{{
    "summary": "Detaylı özet (3-5 cümle)",
    "key_insights": ["içgörü1", "içgörü2"],
    "action_items": ["eylem1" varsa],
    "related_topics": ["ilişkili_konu"]
}}"""

    def __init__(self, ollama: OllamaClient):
        self.ollama = ollama

    async def cluster_weekly_notes(self, notes: List[Dict]) -> Optional[Dict]:
        """Haftalık notları konulara kümele"""
        if not notes:
            return None

        # Notları metne çevir
        notes_text = "\n".join([
            f"- [{n.get('created', '')}] {n.get('content', '')[:100]}"
            for n in notes[:50]  # Max 50 not
        ])

        prompt = self.CLUSTERING_PROMPT.format(notes=notes_text)
        result = self.ollama.generate(prompt, json_mode=True)

        return self.ollama.extract_json(result)

    async def summarize_topic(self, topic_name: str, notes: List[Dict]) -> Optional[Dict]:
        """Konu detaylı özeti"""
        notes_text = "\n".join([
            f"- {n.get('content', '')}"
            for n in notes
        ])

        prompt = self.TOPIC_SUMMARY_PROMPT.format(
            topic_name=topic_name,
            notes=notes_text
        )
        result = self.ollama.generate(prompt, json_mode=True)

        return self.ollama.extract_json(result)


# ==================== DEEP ARCHIVE ====================
class DeepArchive:
    """
    Kalıcı arşiv - Topic based storage
    - JSON: İndeksleme ve hızlı erişim
    - Markdown: Okunabilirlik
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.path / "archive_index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        if self.index_file.exists():
            try:
                return json.loads(self.index_file.read_text(encoding='utf-8'))
            except:
                pass
        return {"topics": {}, "stats": {"total_topics": 0, "total_notes": 0}}

    def _save_index(self):
        self.index_file.write_text(
            json.dumps(self.index, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    def save_topic(
        self,
        topic_id: str,
        topic_name: str,
        summary: Dict,
        notes: List[Dict],
        metadata: Dict
    ) -> str:
        """Konuyu arşivle"""
        topic_dir = self.path / topic_id
        topic_dir.mkdir(exist_ok=True)

        # JSON - Tam veri
        json_file = topic_dir / "data.json"
        topic_data = {
            "id": topic_id,
            "name": topic_name,
            "created": datetime.now().isoformat(),
            "summary": summary,
            "note_count": len(notes),
            "metadata": metadata
        }
        json_file.write_text(
            json.dumps(topic_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        # Markdown - Okunabilir
        md_file = topic_dir / "readme.md"
        md_content = self._generate_markdown(topic_name, summary, notes, metadata)
        md_file.write_text(md_content, encoding='utf-8')

        # İndeksi güncelle
        self.index["topics"][topic_id] = {
            "name": topic_name,
            "summary": summary.get("summary", ""),
            "keywords": summary.get("keywords", []),
            "created": datetime.now().isoformat(),
            "note_count": len(notes)
        }
        self.index["stats"]["total_topics"] = len(self.index["topics"])
        self.index["stats"]["total_notes"] += len(notes)
        self._save_index()

        return str(json_file)

    def _generate_markdown(
        self,
        topic_name: str,
        summary: Dict,
        notes: List[Dict],
        metadata: Dict
    ) -> str:
        """Markdown format oluştur"""
        lines = [
            f"# {topic_name}\n",
            f"**Oluşturulma:** {metadata.get('week', '?')}\n",
            f"**Not Sayısı:** {len(notes)}\n",
            f"**Duygu:** {summary.get('sentiment', '?')}\n",
            f"**Onem:** {summary.get('importance', '?')}\n",
            "\n## Özet\n",
            summary.get("summary", ""),
            "\n## Anahtar Kelimeler\n",
            ", ".join(summary.get("keywords", [])),
            "\n## Notlar\n"
        ]

        for i, note in enumerate(notes[:20], 1):
            lines.append(f"\n### Not {i}")
            lines.append(f"_Tarih: {note.get('created', '?')}_")
            lines.append(f"{note.get('content', '')}")

        if len(notes) > 20:
            lines.append(f"\n*... ve {len(notes) - 20} not daha*")

        return "\n".join(lines)

    def search_topics(self, query: str) -> List[Dict]:
        """Konularda arama"""
        results = []
        query_lower = query.lower()

        for topic_id, topic in self.index.get("topics", {}).items():
            # İsimde ara
            if query_lower in topic["name"].lower():
                results.append({"id": topic_id, **topic, "match_type": "name"})
                continue

            # Özetde ara
            if query_lower in topic.get("summary", "").lower():
                results.append({"id": topic_id, **topic, "match_type": "summary"})
                continue

            # Anahtar kelimelerde ara
            for kw in topic.get("keywords", []):
                if query_lower in kw.lower():
                    results.append({"id": topic_id, **topic, "match_type": "keyword"})
                    break

        return results

    def get_topic(self, topic_id: str) -> Optional[Dict]:
        """Konu detayını getir"""
        topic_file = self.path / topic_id / "data.json"
        if topic_file.exists():
            try:
                return json.loads(topic_file.read_text(encoding='utf-8'))
            except:
                pass
        return None


# ==================== PERMANENT MEMORY MANAGER ====================
class PermanentMemoryManager:
    """
    Kalıcı Hafıza Yöneticisi

    Hiyerarşi:
    - L1: Sıcak (ChromaDB, 7 gün) - Vektör arama
    - L2: Ilık (Günlük özet, 30 gün) - JSON indeks
    - L3: Soğuk (Haftalık özet, 12 ay) - JSON indeks
    - L4: Derin (Topic-based, sonsuz) - JSON + Markdown
    """

    def __init__(self, cfg: MemoryConfig):
        self.config = cfg

        # Yollar
        self.hot_path = Path(cfg.hot_memory_path)
        self.warm_path = Path(cfg.warm_archive_path)
        self.cold_path = Path(cfg.cold_archive_path)
        for p in [self.hot_path, self.warm_path, self.cold_path]:
            p.mkdir(parents=True, exist_ok=True)

        # ChromaDB
        try:
            self.chroma = chromadb.PersistentClient(
                path=str(self.hot_path),
                settings=Settings(anonymized_telemetry=False)
            )
            self.notes_collection = self.chroma.get_or_create_collection("notes")
            self.chroma_available = True
            logger.info("ChromaDB aktif")
        except Exception as e:
            logger.error(f"ChromaDB hatası: {e}")
            self.chroma_available = False

        # Modüller
        self.ollama = OllamaClient(cfg.ollama_base_url, cfg.ollama_model)
        self.summarizer = SemanticSummarizer(self.ollama)
        self.deep_archive = DeepArchive(cfg.deep_archive_path)

        # İndeksler
        self.daily_index = self.warm_path / "daily_index.json"
        self.weekly_index = self.cold_path / "weekly_index.json"
        self.daily_summaries = self._load_json(self.daily_index, {})
        self.weekly_summaries = self._load_json(self.weekly_index, {})

    def _load_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except:
                pass
        return default

    def _save_json(self, path: Path, data: Any):
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )

    # ==================== ARŞİVLEME PROTESOKOLÜ ====================
    async def weekly_archive_protocol(self, user_id: int) -> Dict:
        """
        Haftalık Arşivleme Protokolü - 02:00'de çalışır

        1. ChromaDB'den son 7 günün notlarını al
        2. GLM 4 ile semantik kümeleme
        3. Her konu için detaylı özet
        4. L4 Derin arşive kaydet
        5. L1'den temizle
        """
        logger.info(f"=== Haftalık arşivleme başladı: user={user_id} ===")

        week_id = datetime.now().strftime("%Y-W%U")
        week_start = datetime.now() - timedelta(days=7)

        # 1. Notları topla
        notes = await self._get_weekly_notes(user_id, week_start)

        if not notes:
            logger.info("Arşivlenecek not yok")
            return {"status": "no_notes", "archived": 0}

        logger.info(f"{len(notes)} not arşivlenecek")

        # 2. Kümeleme
        clustering_result = await self.summarizer.cluster_weekly_notes(notes)

        if not clustering_result:
            logger.warning("Kümeleme başarısız, yedek yöntem kullanılıyor")
            clustering_result = {"topics": [], "week_summary": "Özet oluşturulamadı"}

        # 3. Her konuyu arşivle
        archived_count = 0
        for topic in clustering_result.get("topics", []):
            topic_name = topic["name"]
            keywords = topic.get("keywords", [])

            # Konuya ait notları filtrele
            topic_notes = self._filter_notes_by_keywords(notes, topic_name, keywords)

            if len(topic_notes) < self.config.min_notes_per_topic:
                continue

            # Detaylı özet
            detailed_summary = await self.summarizer.summarize_topic(topic_name, topic_notes)

            # Arşivle
            topic_id = f"{week_id}_{self._slugify(topic_name)}"
            self.deep_archive.save_topic(
                topic_id=topic_id,
                topic_name=topic_name,
                summary=detailed_summary or topic,
                notes=topic_notes,
                metadata={"week": week_id, "user_id": user_id}
            )
            archived_count += len(topic_notes)

        # 4. ChromaDB'den temizle
        await self._purge_archived_notes(user_id, week_start)

        # 5. İndeksleri güncelle
        self.weekly_summaries[week_id] = {
            "user_id": user_id,
            "created": datetime.now().isoformat(),
            "summary": clustering_result.get("week_summary", ""),
            "topics": [t["name"] for t in clustering_result.get("topics", [])],
            "note_count": len(notes),
            "archived_count": archived_count
        }
        self._save_json(self.weekly_index, self.weekly_summaries)

        logger.info(f"=== Arşivleme tamamlandı: {archived_count}/{len(notes)} not ===")

        return {
            "status": "success",
            "total_notes": len(notes),
            "archived": archived_count,
            "topics": len(clustering_result.get("topics", []))
        }

    async def _get_weekly_notes(self, user_id: int, since: datetime) -> List[Dict]:
        """ChromaDB'den haftalık notları al"""
        if not self.chroma_available:
            return []

        try:
            results = self.notes_collection.get(
                where={"user_id": str(user_id)},
                limit=10000,
                include=["documents", "metadatas", "ids"]
            )

            notes = []
            cutoff = since

            for doc, meta, doc_id in zip(
                results['documents'],
                results['metadatas'],
                results['ids']
            ):
                created = meta.get('created', '')
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created)
                        if created_dt >= cutoff:
                            notes.append({
                                "id": doc_id,
                                "content": doc,
                                "created": created,
                                **meta
                            })
                    except:
                        pass

            return notes
        except Exception as e:
            logger.error(f"Not alma hatası: {e}")
            return []

    def _filter_notes_by_keywords(
        self,
        notes: List[Dict],
        topic_name: str,
        keywords: List[str]
    ) -> List[Dict]:
        """Anahtar kelimelere göre notları filtrele"""
        filtered = []
        topic_lower = topic_name.lower()
        keywords_lower = [k.lower() for k in keywords]

        for note in notes:
            content = note.get("content", "").lower()

            # Konu adı veya anahtar kelime eşleşmesi
            if topic_lower in content or any(kw in content for kw in keywords_lower):
                filtered.append(note)

        return filtered

    def _slugify(self, text: str) -> str:
        """Metni dosya adı için güvenli hale getir"""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text[:50]

    async def _purge_archived_notes(self, user_id: int, since: datetime):
        """Arşivlenen notları ChromaDB'den sil"""
        if not self.chroma_available:
            return

        try:
            results = self.notes_collection.get(
                where={"user_id": str(user_id)},
                limit=10000,
                include=["metadatas", "ids"]
            )

            ids_to_delete = []
            cutoff = since

            for meta, doc_id in zip(results['metadatas'], results['ids']):
                created = meta.get('created', '')
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created)
                        if created_dt < cutoff:
                            ids_to_delete.append(doc_id)
                    except:
                        pass

            if ids_to_delete:
                self.notes_collection.delete(ids=ids_to_delete)
                logger.info(f"{len(ids_to_delete)} not ChromaDB'den silindi")
        except Exception as e:
            logger.error(f"Temizleme hatası: {e}")

    # ==================== HİBRİT SORGULAMA ====================
    async def hybrid_search(self, user_id: int, query: str) -> Dict:
        """
        Hibrit Sorgulama Protokolü

        Akış:
        1. L1 Sıcak (ChromaDB vektör)
        2. L2 Ilık (Günlük indeks)
        3. L3 Soğuk (Haftalık indeks)
        4. L4 Derin (Topic-based arşiv)
        """
        logger.info(f"Hibrit arama: '{query[:50]}...'")

        results = {
            "query": query,
            "total": 0,
            "sources": {
                "hot": [],
                "warm": [],
                "cold": [],
                "deep": []
            }
        }

        # L1: Sıcak - Vektör arama
        hot_results = await self._search_hot(user_id, query)
        if hot_results:
            results["sources"]["hot"] = hot_results
            results["total"] += len(hot_results)
            logger.info(f"  L1 Sıcak: {len(hot_results)} sonuç")
            return results

        # L2: Ilık - Günlük özet
        warm_results = self._search_warm(user_id, query)
        if warm_results:
            results["sources"]["warm"] = warm_results
            results["total"] += len(warm_results)
            logger.info(f"  L2 Ilık: {len(warm_results)} sonuç")
            return results

        # L3: Soğuk - Haftalık özet
        cold_results = self._search_cold(user_id, query)
        if cold_results:
            results["sources"]["cold"] = cold_results
            results["total"] += len(cold_results)
            logger.info(f"  L3 Soğuk: {len(cold_results)} sonuç")
            return results

        # L4: Derin - Topic arşiv
        deep_results = self.deep_archive.search_topics(query)
        if deep_results:
            results["sources"]["deep"] = deep_results
            results["total"] += len(deep_results)
            logger.info(f"  L4 Derin: {len(deep_results)} sonuç")

        if results["total"] == 0:
            logger.info("  Sonuç bulunamadı")

        return results

    async def _search_hot(self, user_id: int, query: str, n: int = 5) -> List[Dict]:
        """ChromaDB vektör arama"""
        if not self.chroma_available:
            return []

        try:
            results = self.notes_collection.query(
                query_texts=[query],
                n_results=n,
                where={"user_id": str(user_id)}
            )

            items = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    items.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'level': 'hot'
                    })
            return items
        except:
            return []

    def _search_warm(self, user_id: int, query: str) -> List[Dict]:
        """Günlük özet arama"""
        results = []
        query_lower = query.lower()

        for date, summary in self.daily_summaries.items():
            if summary.get('user_id') != user_id:
                continue

            text = f"{summary.get('summary', '')} {' '.join(summary.get('topics', []))}"
            if query_lower in text.lower():
                results.append({
                    'type': 'daily',
                    'date': date,
                    'summary': summary,
                    'level': 'warm'
                })

        return results

    def _search_cold(self, user_id: int, query: str) -> List[Dict]:
        """Haftalık özet arama"""
        results = []
        query_lower = query.lower()

        for week, summary in self.weekly_summaries.items():
            if summary.get('user_id') != user_id:
                continue

            text = f"{summary.get('summary', '')} {' '.join(summary.get('topics', []))}"
            if query_lower in text.lower():
                results.append({
                    'type': 'weekly',
                    'week': week,
                    'summary': summary,
                    'level': 'cold'
                })

        return results

    # ==================== NOT EKLEME ====================
    def add_note(self, user_id: int, text: str, tags: List[str] = None) -> str:
        """Not ekle - Sıcak hafızaya"""
        if not self.chroma_available:
            return None

        note_id = f"note_{user_id}_{datetime.now().timestamp()}"
        metadata = {
            "user_id": str(user_id),
            "created": datetime.now().isoformat(),
            "tags": tags or []
        }

        try:
            self.notes_collection.add(
                ids=[note_id],
                documents=[text],
                metadatas=[metadata]
            )
            return note_id
        except Exception as e:
            logger.error(f"Not ekleme hatası: {e}")
            return None


# ==================== TEST ====================
async def main():
    """Test"""
    manager = PermanentMemoryManager(config)

    # Test notları ekle
    print("=== Test notları ekleniyor ===")
    for i in range(10):
        manager.add_note(
            user_id=123456,
            text=f"Test notu {i}: Bu bir örnek not içeriğidir",
            tags=["test"]
        )

    # Hibrit arama test
    print("\n=== Hibrit arama test ===")
    results = await manager.hybrid_search(123456, "test")
    print(f"Bulunan sonuç: {results['total']}")

    # Haftalık arşivleme test
    print("\n=== Haftalık arşivleme test ===")
    archive_result = await manager.weekly_archive_protocol(123456)
    print(f"Arşivleme sonucu: {archive_result}")


if __name__ == "__main__":
    asyncio.run(main())
