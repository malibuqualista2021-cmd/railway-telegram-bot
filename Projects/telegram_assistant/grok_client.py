"""
xAI Grok API Client
"""
import requests
import json

class GrokClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def chat(self, message: str, system_prompt: str = "You are a helpful assistant.", model: str = "grok-4-latest"):
        """Mesaj gönder ve yanıt al"""

        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "model": model,
            "stream": False,
            "temperature": 0
        }

        response = requests.post(self.base_url, headers=self.headers, json=data)
        return response.json()

    def chat_with_history(self, messages: list, model: str = "grok-4-latest"):
        """Mesaj geçmişi ile chat"""

        data = {
            "messages": messages,
            "model": model,
            "stream": False,
            "temperature": 0
        }

        response = requests.post(self.base_url, headers=self.headers, json=data)
        return response.json()


# Kullanım örneği
if __name__ == "__main__":
    # API anahtarını buraya yapıştır
    API_KEY = "xai-1VXFoqNfvUhEWdoUS3iXbFfSa9xCFtPuVr0stLl1QCWwkVAOaVCVRSntWGUi546mlH0RxCu4y3S4zAG8"

    client = GrokClient(API_KEY)

    # Basit mesaj
    result = client.chat("Merhaba, nasılsın?")
    print(result)

    # Daha okunabilir çıktı
    if "choices" in result:
        print("\n--- Yanıt ---")
        print(result["choices"][0]["message"]["content"])
