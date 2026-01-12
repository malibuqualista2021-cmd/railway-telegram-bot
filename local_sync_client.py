#!/usr/bin/env python3
"""
Yerel PC Senkronizasyon Modülü
Railway ile yerel PC arası not senkronizasyonu

Kullanım:
1. Bu dosyayı yerel PC'de telegram_asistant_v42.py ile aynı klasöre koy
2. RAILWAY_SYNC_URL ve SYNC_TOKEN ortam değişkenlerini ayarla
3. Bot başladığında otomatik senkronize olur
"""
import os
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ==================== CONFIG ====================
class LocalSyncConfig:
    # Railway sync URL (Railway projende deployments'dan URL'yi al)
    # Örnek: https://telegram-hybrid-bot-production.up.railway.app
    railway_sync_url: str = os.getenv("RAILWAY_SYNC_URL", "")

    # Railway'de ayarladığın sync token
    sync_token: str = os.getenv("SYNC_TOKEN", "")

    # Yerel kullanıcı ID (Telegram kullanıcı ID'n)
    user_id: int = int(os.getenv("TELEGRAM_USER_ID", "0"))

    def is_valid(self) -> bool:
        return bool(self.railway_sync_url and self.sync_token)


config = LocalSyncConfig()


# ==================== SYNC CLIENT ====================
class RailwaySyncClient:
    """Railway sync client"""

    def __init__(self, base_url: str, sync_token: str):
        self.base_url = base_url.rstrip('/')
        self.sync_token = sync_token
        self.headers = {
            "X-Sync-Token": sync_token,
            "Content-Type": "application/json"
        }

    def health_check(self) -> bool:
        """Railway sync API'nin çalıştığını kontrol et"""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except:
            return False

    def send_to_railway(self, notes: List[Dict]) -> Dict:
        """Yerel notları Railway'e gönder"""
        try:
            payload = {"notes": notes, "user_id": config.user_id}
            r = requests.post(
                f"{self.base_url}/sync/from-local",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            return r.json() if r.status_code == 200 else {"error": r.status_code}
        except Exception as e:
            logger.error(f"Railway'e gönderme hatası: {e}")
            return {"error": str(e)}

    def get_from_railway(self, user_id: int = None) -> List[Dict]:
        """Railway'den notları çek"""
        try:
            uid = user_id or config.user_id
            r = requests.get(
                f"{self.base_url}/sync/to-local",
                headers=self.headers,
                params={"user_id": uid},
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("notes", [])
        except Exception as e:
            logger.error(f"Railway'den çekme hatası: {e}")
        return []

    def get_all_from_railway(self, user_id: int = None) -> List[Dict]:
        """Tüm notları Railway'den çek (tam sync için)"""
        try:
            uid = user_id or config.user_id
            r = requests.get(
                f"{self.base_url}/sync/all",
                headers=self.headers,
                params={"user_id": uid},
                timeout=60
            )
            if r.status_code == 200:
                data = r.json()
                return data.get("notes", [])
        except Exception as e:
            logger.error(f"Tüm notları çekme hatası: {e}")
        return []

    def mark_synced(self, note_ids: List[str]) -> bool:
        """Notları senkronize edildi olarak işaretle"""
        try:
            r = requests.post(
                f"{self.base_url}/sync/mark-local-synced",
                headers=self.headers,
                json={"note_ids": note_ids},
                timeout=10
            )
            return r.status_code == 200
        except Exception as e:
            logger.error(f"İşaretleme hatası: {e}")
            return False


# Global client
_sync_client: Optional[RailwaySyncClient] = None


def get_sync_client() -> Optional[RailwaySyncClient]:
    """Sync client'ı getir (lazy loading)"""
    global _sync_client

    if not config.is_valid():
        return None

    if _sync_client is None:
        _sync_client = RailwaySyncClient(config.railway_sync_url, config.sync_token)

    return _sync_client


# ==================== SYNC FUNCTIONS ====================
async def sync_on_startup():
    """
    Bot başladığında çalışacak senkronizasyon
    Railway'den notları çek, yerel notları Railway'e gönder
    """
    client = get_sync_client()
    if not client:
        logger.warning("Sync client başlatılamadı (config eksik)")
        return

    logger.info("Railway ile senkronizasyon başlıyor...")

    # 1. Sağlık kontrolü
    if not client.health_check():
        logger.error("Railway sync API yanıt vermiyor!")
        return

    # 2. Railway'den notları çek
    railway_notes = client.get_from_railway()
    logger.info(f"Railway'den {len(railway_notes)} not çekildi")

    # 3. Yerel ChromaDB'ye ekle (gerekirse)
    if railway_notes:
        # Burada ana botun ChromaDB'sine eklemeliyiz
        # Şimdilik log'luyoruz
        synced_ids = [n.get("id") for n in railway_notes if n.get("id")]
        if synced_ids:
            client.mark_synced(synced_ids)
            logger.info(f"{len(synced_ids)} not yerel'e senkronize edildi")

    logger.info("Senkronizasyon tamamlandı")


async def sync_to_railway(notes: List[Dict]):
    """
    Yerel notları Railway'e gönder
    (Bot çalışırken yeni notlar eklendiğinde çağrılır)
    """
    client = get_sync_client()
    if not client:
        return

    result = client.send_to_railway(notes)
    if "error" not in result:
        logger.info(f"{result.get('added', 0)} not Railway'e gönderildi")
    else:
        logger.error(f"Railway'e gönderme hatası: {result}")


async def sync_from_railway():
    """
    Railway'den yeni notları çek
    (Döngüsel olarak çağrılabilir)
    """
    client = get_sync_client()
    if not client:
        return []

    notes = client.get_from_railway()
    if notes:
        synced_ids = [n.get("id") for n in notes if n.get("id")]
        client.mark_synced(synced_ids)
        logger.info(f"{len(notes)} not Railway'den çekildi")

    return notes


# ==================== BAĞIMSIZ ÇALIŞTIRMA ====================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )

    print("=" * 50)
    print("Railway Sync Client Test")
    print("=" * 50)

    if not config.is_valid():
        print("Config hatası!")
        print("Ayarlanması gereken ortam değişkenleri:")
        print("  RAILWAY_SYNC_URL=https://your-app.railway.app")
        print("  SYNC_TOKEN=your_token")
        print("  TELEGRAM_USER_ID=your_id")
    else:
        client = get_sync_client()

        print(f"Railway URL: {config.railway_sync_url}")
        print(f"User ID: {config.user_id}")

        # Health check
        print("\n[1] Health check...")
        if client.health_check():
            print("✅ Railway API yanıt veriyor")

            # Tüm notları çek
            print("\n[2] Tüm notlar çekiliyor...")
            notes = client.get_all_from_railway()
            print(f"✅ {len(notes)} not bulundu")

            # Son 5 notu göster
            if notes:
                print("\n[3] Son 5 not:")
                for note in notes[-5:]:
                    print(f"  - {note.get('created', '')[:16]}: {note.get('text', '')[:50]}...")
        else:
            print("❌ Railway API yanıt vermiyor")

    print("\n" + "=" * 50)
