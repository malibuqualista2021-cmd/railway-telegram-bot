#!/usr/bin/env python3
"""
Groq ile Sesli Asistan Çekirdeği
Metin girişini alır, Groq (Llama) ile kategorilere ayırır ve ilgili işlemi yapar.
"""

import subprocess
import os
import json
from datetime import datetime
from pathlib import Path
from groq import Groq


class GroqAsistan:
    """Groq destekli sesli asistan."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY gerekli! Ortam değişkeni veya parametre olarak verin.")

        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # En güçlü ücretsiz model
        self.bilgi_dizini = Path.home() / "asistant_notlar"
        self.bilgi_dizini.mkdir(exist_ok=True)

    def groq_cagri(self, prompt: str, system: str = None, json_cikti: bool = False) -> str:
        """Groq API çağrısı yapar."""
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

    def kategorile(self, metin: str) -> dict:
        """Groq ile metni kategorilere ayırır."""
        system_prompt = """Sen bir sesli asistan kategorize edicisin.
Gelen metni analiz et ve JSON formatında döndür.

Kategoriler:
- "otomasyon": Terminal komutu çalıştırma istekleri (çalıştır, komut, bash, run, execute, ac, kapat, sil vb.)
- "bilgi": Not kaydetme, bilgi saklama istekleri (not al, kaydet, hatırla, deftere yaz vb.)
- "iletisim": E-posta veya mesaj gönderme istekleri (mail at, mesaj gönder, e-posta yaz vb.)

JSON formatı:
{
    "kategori": "otomasyon|bilgi|iletisim",
    "icerik": "işlenecek asıl içerik (komut, not veya mesaj metni)",
    "aciklama": "kısa açıklama"
}

Sadece JSON döndür, başka hiçbir şey yazma."""

        response = self.groq_cagri(metin, system=system_prompt, json_cikti=True)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # JSON başarısız olursa basit fallback
            return self._basit_kategori(metin)

    def _basit_kategori(self, metin: str) -> dict:
        """Fallback kategorilendirme."""
        metin_lower = metin.lower()

        oto_keywords = ["çalıştır", "komut", "terminal", "bash", "run", "execute", "ac", "kapat", "sil"]
        bilgi_keywords = ["not", "kaydet", "hatırla", "not al", "yaz", "bilgi", "defter"]
        iletisim_keywords = ["mail", "e-posta", "mesaj", "gonder", "mail at", "email"]

        for kw in oto_keywords:
            if kw in metin_lower:
                return {"kategori": "otomasyon", "icerik": metin, "aciklama": "Terminal komutu"}

        for kw in bilgi_keywords:
            if kw in metin_lower:
                return {"kategori": "bilgi", "icerik": metin, "aciklama": "Not kaydı"}

        for kw in iletisim_keywords:
            if kw in metin_lower:
                return {"kategori": "iletisim", "icerik": metin, "aciklama": "Mesaj taslağı"}

        return {"kategori": "bilgi", "icerik": metin, "aciklama": "Varsayılan not"}

    def otomasyon_isle(self, metin: str) -> str:
        """Terminal komutu çalıştırır."""
        komut = metin.strip()

        print(f"\n[Otomasyon] Çalıştırılıyor: {komut}")

        try:
            if os.name == 'nt':
                result = subprocess.run(
                    komut,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=30
                )
            else:
                result = subprocess.run(
                    komut,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

            cikti = result.stdout or result.stderr
            if not cikti:
                cikti = "✓ Başarılı (çıktı yok)"

            return f"Çıktı:\n{cikti}"

        except subprocess.TimeoutExpired:
            return "❌ Hata: Komut zaman aşımına uğradı"
        except Exception as e:
            return f"❌ Hata: {str(e)}"

    def bilgi_isle(self, metin: str) -> str:
        """Metni markdown dosyasına kaydeder."""
        icerik = metin.strip()
        tarih = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dosya_adi = f"not_{tarih}.md"
        dosya_yolu = self.bilgi_dizini / dosya_adi

        md_icerik = f"""# Not - {tarih}

{icerik}

---
*Kaydedilme: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}*
"""

        with open(dosya_yolu, "w", encoding="utf-8") as f:
            f.write(md_icerik)

        return f"✓ Not kaydedildi: {dosya_yolu}"

    def iletisim_isle(self, metin: str) -> str:
        """E-posta/mesaj taslağı hazırlar."""
        tarih = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dosya_adi = f"mesaj_taslak_{tarih}.md"
        dosya_yolu = self.bilgi_dizini / dosya_adi

        # GLM'den alıcı ve konu önerisi al
        system_prompt = "Verilen metinden e-posta alıcısı ve konu bilgisini çıkar. JSON formatında döndür: {alici: '...', konu: '...'}"

        try:
            response = self.glm_cagri(
                f"Bu mesaj için alıcı ve konu belirle: {metin}",
                system=system_prompt,
                json_cikti=True
            )
            veri = json.loads(response)
            alici = veri.get("alici", "[Alıcı]")
            konu = veri.get("konu", "[Konu]")
        except:
            alici = "[Alıcı]"
            konu = "[Konu]"

        taslak = f"""# Mesaj Taslağı
{tarih}: {datetime.now().strftime("%d.%m.%Y %H:%M")}

**Konu:** {konu}
**Alıcı:** {alici}

---
{metin.strip()}

---
*Gönderilmeyi bekliyor.*
"""

        with open(dosya_yolu, "w", encoding="utf-8") as f:
            f.write(taslak)

        return f"✓ Mesaj taslağı hazır: {dosya_yolu}"

    def isle(self, metin: str) -> str:
        """Ana işlem metodu."""
        if not metin or not metin.strip():
            return "Hata: Boş metin"

        sonuc = self.kategorile(metin)
        kategori = sonuc.get("kategori", "bilgi")
        icerik = sonuc.get("icerik", metin)

        print(f"[Kategori: {kategori.upper()}] - {sonuc.get('aciklama', '')}")

        if kategori == "otomasyon":
            return self.otomasyon_isle(icerik)
        elif kategori == "bilgi":
            return self.bilgi_isle(icerik)
        elif kategori == "iletisim":
            return self.iletisim_isle(icerik)
        else:
            return self.bilgi_isle(icerik)

    def chat(self, metin: str) -> str:
        """Genel sohbet için Groq çağrısı."""
        system_prompt = "Sen yardımcı bir Türkçe asistansın. Kısa ve öz cevap ver."
        return self.groq_cagri(metin, system=system_prompt)


def main():
    """Komut satırı arayüzü."""
    import getpass

    print("=" * 50)
    print("    Groq (Llama 3.3) Asistan Çekirdeği")
    print("=" * 50)

    # API key kontrolü
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Groq API Key: ")

    try:
        asistan = GroqAsistan(api_key=api_key)
    except Exception as e:
        print(f"Bağlantı hatası: {e}")
        return

    print("\nKategoriler: Bilgi | Otomasyon | İletişim")
    print("Çıkış için: quit, exit, q\n")

    while True:
        try:
            metin = input("Siz: ").strip()

            if metin.lower() in ["quit", "exit", "q"]:
                print("Güle güle!")
                break

            sonuc = asistan.isle(metin)
            print(f"\nAsistan: {sonuc}\n")

        except KeyboardInterrupt:
            print("\nGüle güle!")
            break


if __name__ == "__main__":
    main()
