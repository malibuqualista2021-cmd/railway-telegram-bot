#!/usr/bin/env python3
"""
Senkronizasyon Modülü
Yerel sistem ile bulut bot arasında senkronizasyon sağlar

Çalışma mantığı:
1. Yerel sistem her 1 dakikada bir "heartbeat" gönderir
2. Bulut bot, yerel sistem aktifse notları yerel'e yönlendirir
3. Yerel sistem, buluttan bekleyen notları çeker
4. Notları ChromaDB'ye ekler
5. Bulutu günceller (sync edildi olarak işaretler)
"""

import os
import json
import asyncio
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


# ==================== YAPILANDIRMA ====================
# Bu ayarları ortak bir config dosyasından alınmalı

# Yerel sistem (PC)
LOCAL_STORAGE_URL = "http://localhost:8000"  # Yerel API endpoint
SYNC_HEARTBEAT_FILE = Path.home() / "asistant_v42_deep" / "sync_heartbeat.txt"

# Bulut sistem (VPS)
CLOUD_SYNC_URL = os.getenv("CLOUD_SYNC_URL", "")  # Bulut senkronizasyon URL'i
CLOUD_TOKEN = os.getenv("CLOUD_TOKEN", "")  # Bulut bot token

# Groq (fallback için)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# ==================== SENKRONİZASYON SERVİSİ ====================
class SyncService:
    """
    Yerel sistemde çalışan senkronizasyon servisi
    - Buluttan bekleyen notları çeker
    - Yerel ChromaDB'ye ekler
    - Bulutu günceller
    """

    def __init__(self):
        self.heartbeat_file = SYNC_HEARTBEAT_FILE
        self.sync_log_file = Path.home() / "asistant_v42_deep" / "sync_log.json"
        self.sync_log = self._load_sync_log()
        self.cloud_url = CLOUD_SYNC_URL
        self.running = False

        # Dizinleri oluştur
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_sync_log(self) -> Dict:
        if self.sync_log_file.exists():
            try:
                return json.loads(self.sync_log_file.read_text(encoding='utf-8'))
            except:
                pass
        return {"last_sync": None, "synced_count": 0, "pending": []}

    def _save_sync_log(self):
        self.sync_log_file.write_text(
            json.dumps(self.sync_log, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8'
        )

    async def start_heartbeat(self):
        """
        Her 1 dakikada bir heartbeat gönder
        Bu sayede bulut bot yerel sistemin aktif olduğunu anlar
        """
        self.running = True

        while self.running:
            try:
                # Heartbeat dosyasını güncelle
                self.heartbeat_file.write_text(datetime.now().isoformat())

                # Bulut senkronizasyonunu dene
                if self.cloud_url:
                    await self._sync_from_cloud()

                # 1 dakika bekle
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Heartbeat hatası: {e}")
                await asyncio.sleep(60)

    def stop_heartbeat(self):
        """Heartbeat'i durdur"""
        self.running = False

    async def _sync_from_cloud(self):
        """
        Buluttan bekleyen notları çek
        """
        try:
            # Bekleyen notları iste
            response = requests.get(
                f"{self.cloud_url}/sync/pending",
                headers={"X-Token": CLOUD_TOKEN},
                timeout=10
            )

            if response.status_code != 200:
                return

            data = response.json()
            pending_notes = data.get("notes", [])

            if not pending_notes:
                logger.debug("Senkronize edilecek not yok")
                return

            logger.info(f"{len(pending_notes)} not buluttan çekiliyor...")

            # Her notu yerel sisteme ekle
            # (Burada Ana Bot'un ChromaDB'sine eklemeliyiz)
            synced_ids = []

            for note in pending_notes:
                # Notu yerel sisteme ekle
                success = await self._add_to_local_memory(note)
                if success:
                    synced_ids.append(note["id"])

            # Bulutu güncelle
            if synced_ids:
                await self._mark_cloud_synced(synced_ids)

                # Log güncelle
                self.sync_log["last_sync"] = datetime.now().isoformat()
                self.sync_log["synced_count"] += len(synced_ids)
                self.sync_log["pending"] = [
                    n for n in self.sync_log.get("pending", [])
                    if n["id"] not in synced_ids
                ]
                self._save_sync_log()

                logger.info(f"{len(synced_ids)} not senkronize edildi")

        except Exception as e:
            logger.error(f"Senkronizasyon hatası: {e}")

    async def _add_to_local_memory(self, note: Dict) -> bool:
        """
        Notu yerel ChromaDB'ye ekle
        (Bu fonksiyon Ana Bot'un HierarchicalMemory'sini kullanmalı)
        """
        try:
            # Burada dinamik import ile ana botun memory sınıfını yükleyeceğiz
            # Döngüsel import sorunlarını önlemek için
            import sys
            sys.path.insert(0, str(Path.home()))

            from telegram_asistant_v42 import HierarchicalMemory, config

            if not hasattr(self, 'memory'):
                self.memory = HierarchicalMemory(
                    config.hot_memory_path,
                    config.warm_archive_path,
                    config.cold_archive_path
                )

            # Notu ekle
            note_id = self.memory.add_note(
                user_id=note["user_id"],
                text=note["text"],
                tags=["cloud_synced"]
            )

            return note_id is not None

        except Exception as e:
            logger.error(f"Yerel hafızaya ekleme hatası: {e}")
            return False

    async def _mark_cloud_synced(self, note_ids: List[str]):
        """
        Buluttaki notları senkronize edildi olarak işaretle
        """
        try:
            response = requests.post(
                f"{self.cloud_url}/sync/mark",
                headers={"X-Token": CLOUD_TOKEN, "Content-Type": "application/json"},
                json={"note_ids": note_ids},
                timeout=10
            )

            if response.status_code == 200:
                logger.debug("Bulut güncellendi")
        except Exception as e:
            logger.error(f"Bulut güncelleme hatası: {e}")

    async def get_pending_notes(self) -> List[Dict]:
        """Bekleyen notları getir (manuel kontrol için)"""
        try:
            response = requests.get(
                f"{self.cloud_url}/sync/pending",
                headers={"X-Token": CLOUD_TOKEN},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("notes", [])
        except:
            pass

        return []


# ==================== YEREL SENKRONİZASYON API'Sİ ====================
from flask import Flask, request, jsonify

class SyncAPI:
    """
    Yerel sistem için Flask API
    Bulut bot bu API'ye bağlanarak senkronizasyon yapar
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.sync_service = SyncService()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/health", methods=["GET"])
        def health():
            """Health check"""
            return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

        @self.app.route("/sync/pending", methods=["GET"])
        def get_pending():
            """Bekleyen notları buluta sun"""
            # Burada yerel ChromaDB'den senkronize edilmemiş notları çekmeli
            # Şimdilik örnek implementasyon
            return jsonify({
                "notes": self._get_pending_from_local()
            })

        @self.app.route("/sync/mark", methods=["POST"])
        def mark_synced():
            """Notları senkronize edildi olarak işaretle"""
            data = request.json
            note_ids = data.get("note_ids", [])

            # Yerel'de notları güncelle
            for note_id in note_ids:
                self._mark_synced_local(note_id)

            return jsonify({"status": "ok", "count": len(note_ids)})

    def _get_pending_from_local(self) -> List[Dict]:
        """Yerel sistemden bekleyen notları çıkar"""
        # Bu implementasyon ana botun yapısına göre güncellenmeli
        pending = self.sync_log.get("pending", [])

        # ChromaDB'den yeni notları da kontrol edebiliriz
        return pending

    def _mark_synced_local(self, note_id: str):
        """Notu yerelde sync edildi olarak işaretle"""
        # Implementasyon ana botun yapısına göre
        pass

    def run(self, port=8000):
        """API servisini başlat"""
        logger.info(f"Senkronizasyon API başlatılıyor (port {port})")
        self.app.run(host="127.0.0.1", port=port)


# ==================== ANA BOT'A ENTEGRASYON ====================
class HybridMemoryManager:
    """
    Hibrit hafıza yöneticisi
    Yerel bot'a eklenen senkronizasyon özellikleri
    """

    def __init__(self, base_memory):
        """
        Args:
            base_memory: Mevcut HierarchicalMemory örneği
        """
        self.base_memory = base_memory
        self.sync_service = SyncService()

    async def add_hybrid_note(self, user_id: int, text: str, source: str = "local") -> str:
        """
        Hibrit not ekleme

        Args:
            user_id: Kullanıcı ID
            text: Not içeriği
            source: 'local' veya 'cloud'

        Returns:
            Not ID
        """
        # Yerel hafızaya ekle
        note_id = self.base_memory.add_note(user_id, text, tags=[f"source_{source}"])

        if source == "cloud":
            # Buluttan gelen notu log'a ekle
            self.sync_service.sync_log["pending"].append({
                "id": note_id,
                "cloud_sync": True
            })
            self.sync_service._save_sync_log()

        return note_id

    async def sync_from_cloud(self):
        """Buluttan senkronizasyon başlat"""
        await self.sync_service._sync_from_cloud()

    async def start_sync_service(self):
        """Senkronizasyon servisini başlat"""
        # Heartbeat thread'ini başlat
        import threading

        def run_heartbeat():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.sync_service.start_heartbeat())

        heartbeat_thread = threading.Thread(target=run_heartbeat, daemon=True)
        heartbeat_thread.start()

        logger.info("Senkronizasyon servisi başlatıldı")


# ==================== BAĞIMSIZ ÇALIŞTIRMA ====================
async def main():
    """Test"""
    sync = SyncService()

    # Heartbeat'i test et
    print("Heartbeat test (10 saniye)...")
    await sync.start_heartbeat()


if __name__ == "__main__":
    asyncio.run(main())
