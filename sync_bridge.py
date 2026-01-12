#!/usr/bin/env python3
"""
Senkronizasyon Köprüsü
Railway <-> Yerel PC arası not senkronizasyonu

Railway'de çalışır, HTTP API sunar
Yerel PC bu API'ye bağlanır
"""
import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== CONFIG ====================
class SyncConfig:
    """Senkronizasyon config"""
    sync_token: str = os.getenv("SYNC_TOKEN", "change-me-secure-token")
    storage_path: str = os.getenv("RAILWAY_VOLUME_URL", "/data/storage")

    def validate(self) -> bool:
        return self.sync_token != "change-me-secure-token"


config = SyncConfig()


# ==================== STORAGE ====================
class SyncStorage:
    """Railway storage için"""

    def __init__(self, storage_path: str = "/data/storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.notes_file = self.storage_path / "notes.json"
        self.sync_log_file = self.storage_path / "sync_log.json"

        self.notes = self._load_notes()
        self.sync_log = self._load_sync_log()

    def _load_notes(self) -> List[Dict]:
        if self.notes_file.exists():
            try:
                return json.loads(self.notes_file.read_text(encoding='utf-8'))
            except:
                pass
        return []

    def _save_notes(self):
        try:
            self.notes_file.write_text(
                json.dumps(self.notes, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Not kaydetme hatası: {e}")

    def _load_sync_log(self) -> Dict:
        if self.sync_log_file.exists():
            try:
                return json.loads(self.sync_log_file.read_text(encoding='utf-8'))
            except:
                pass
        return {"last_sync": None, "synced_notes": []}

    def _save_sync_log(self):
        try:
            self.sync_log_file.write_text(
                json.dumps(self.sync_log, ensure_ascii=False, indent=2, default=str),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f"Sync log kaydetme hatası: {e}")

    def add_notes(self, notes: List[Dict]) -> int:
        """Yerelden gelen notları ekle"""
        added = 0
        for note in notes:
            # Zaten var mı kontrol et (ID ile)
            if not any(n.get("id") == note.get("id") for n in self.notes):
                note["synced_from"] = "local"
                self.notes.append(note)
                added += 1
        if added > 0:
            self._save_notes()
        return added

    def get_pending_notes(self, user_id: int = None) -> List[Dict]:
        """Yerel'e gönderilecek notlar"""
        pending = []
        for note in self.notes:
            # Yerel'den gelen değilse ve Railway'de oluşturulduysa
            if note.get("synced_from") != "local" and not note.get("synced_to_local", False):
                if user_id is None or note.get("user_id") == user_id:
                    pending.append(note)
        return pending

    def mark_synced_to_local(self, note_ids: List[str]):
        """Notları yerel'e senkronize edildi olarak işaretle"""
        count = 0
        for note in self.notes:
            if note.get("id") in note_ids:
                note["synced_to_local"] = True
                count += 1
        if count > 0:
            self._save_notes()
            self.sync_log["last_sync"] = datetime.now().isoformat()
            self.sync_log["synced_notes"].extend(note_ids)
            self._save_sync_log()

    def get_all_notes(self) -> List[Dict]:
        """Tüm notları getir (tam sync için)"""
        return self.notes

    def update_sync_log(self):
        """Sync log'u güncelle"""
        self.sync_log["last_sync"] = datetime.now().isoformat()
        self._save_sync_log()


# ==================== FLASK API ====================
app = Flask(__name__)
CORS(app)  # CORS enabled

storage = SyncStorage(config.storage_path)


def check_auth():
    """Token kontrolü"""
    token = request.headers.get("X-Sync-Token")
    if token != config.sync_token:
        return None
    return True


@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "service": "railway-sync-bridge",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/sync/from-local", methods=["POST"])
def from_local():
    """
    Yerel PC'den not al
    Body: {"notes": [...], "user_id": 123}
    """
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.json
        notes = data.get("notes", [])
        user_id = data.get("user_id")

        # Notları ekle
        added = storage.add_notes(notes)

        # Sync log'u güncelle
        storage.update_sync_log()

        return jsonify({
            "status": "ok",
            "added": added,
            "total_notes": len(storage.notes)
        })
    except Exception as e:
        logger.error(f"Sync from local error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/sync/to-local", methods=["GET"])
def to_local():
    """
    Yerel PC'ye not ver
    Query: ?user_id=123
    """
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        user_id = request.args.get("user_id", type=int)

        # Bekleyen notları getir
        pending = storage.get_pending_notes(user_id)

        return jsonify({
            "status": "ok",
            "notes": pending,
            "count": len(pending)
        })
    except Exception as e:
        logger.error(f"Sync to local error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/sync/mark-local-synced", methods=["POST"])
def mark_local_synced():
    """
    Yerel PC'ye senkronize edilen notları işaretle
    Body: {"note_ids": ["id1", "id2", ...]}
    """
    if not check_auth():
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


@app.route("/sync/status", methods=["GET"])
def sync_status():
    """Senkronizasyon durumu"""
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify({
        "status": "ok",
        "total_notes": len(storage.notes),
        "last_sync": storage.sync_log.get("last_sync"),
        "synced_count": len(storage.sync_log.get("synced_notes", []))
    })


@app.route("/sync/all", methods=["GET"])
def get_all():
    """Tüm notları getir (tam sync için)"""
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401

    user_id = request.args.get("user_id", type=int)

    notes = storage.get_all_notes()
    if user_id:
        notes = [n for n in notes if n.get("user_id") == user_id]

    return jsonify({
        "status": "ok",
        "notes": notes,
        "count": len(notes)
    })


# ==================== MAIN ====================
if __name__ == "__main__":
    if not config.validate():
        logger.error("SYNC_TOKEN ortam değişkeni gerekli!")
        logger.info("Örnek: export SYNC_TOKEN=güvenli_bir_token")
        exit(1)

    logger.info("=" * 50)
    logger.info("Railway Sync Bridge Başlatılıyor...")
    logger.info(f"Storage: {config.storage_path}")
    logger.info("=" * 50)

    # Flask app'i çalıştır
    # Railway portunu ortam değişkeninden al
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
