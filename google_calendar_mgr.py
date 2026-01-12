import os
import json
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# SCOPES = ['https://www.googleapis.com/auth/calendar']
SCOPES = ['https://www.googleapis.com/auth/calendar']

class GoogleCalendarManager:
    def __init__(self, credentials_data: str, token_data: str = None, is_path: bool = True):
        """
        credentials_data: URL veya JSON string
        token_data: URL veya JSON string
        is_path: True ise data parametreleri dosya yoludur, False ise direkt JSON stringdir
        """
        self.credentials_data = credentials_data
        self.token_data = token_data
        self.is_path = is_path
        self.token_path = token_data if is_path else None
        self.creds = None
        self._load_credentials()

    def _load_credentials(self):
        # Önce Token'ı yükleyelim (giriş yapılmış mı?)
        if self.is_path:
            if self.token_data and os.path.exists(self.token_data):
                try:
                    self.creds = Credentials.from_authorized_user_file(self.token_data, SCOPES)
                except Exception as e:
                    logger.error(f"Error loading token file: {e}")
        else:
            if self.token_data:
                try:
                    token_info = json.loads(self.token_data)
                    self.creds = Credentials.from_authorized_user_info(token_info, SCOPES)
                except Exception as e:
                    logger.error(f"Error loading token string: {e}")

    def is_authenticated(self) -> bool:
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                    # Token tazelendiğinde kaydet
                    if self.is_path and self.token_path:
                        with open(self.token_path, 'w') as token:
                            token.write(self.creds.to_json())
                    return True
                except Exception as e:
                    logger.error(f"Error refreshing token: {e}")
                    return False
            return False
        return True

    def get_auth_url(self) -> str:
        """Kullanıcının giriş yapması için URL oluştur"""
        if self.is_path:
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_data, SCOPES)
        else:
            flow = InstalledAppFlow.from_client_config(json.loads(self.credentials_data), SCOPES)
        
        flow.redirect_uri = 'http://localhost'
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        return auth_url

    def finalize_auth(self, auth_url_with_code: str):
        """URL içindeki kodu alarak yetkilendirmeyi tamamla"""
        if self.is_path:
            flow = InstalledAppFlow.from_client_secrets_file(self.credentials_data, SCOPES)
        else:
            flow = InstalledAppFlow.from_client_config(json.loads(self.credentials_data), SCOPES)
            
        flow.redirect_uri = 'http://localhost'
        flow.fetch_token(authorization_response=auth_url_with_code)
        self.creds = flow.credentials
        
        # Dosya yolu varsa kaydet
        if self.is_path and self.token_path:
            with open(self.token_path, 'w') as token:
                token.write(self.creds.to_json())
        
        return self.creds.to_json()

        return self.creds.to_json()

    def add_event(self, summary: str, start_time_iso: str, duration_minutes: int = 30):
        """Etkinlik ekle"""
        if not self.is_authenticated():
            return None
        
        service = build('calendar', 'v3', credentials=self.creds)
        start = start_time_iso
        # ISO string bitişi hesapla
        dt_start = datetime.fromisoformat(start.replace('Z', '+00:00'))
        dt_end = dt_start + timedelta(minutes=duration_minutes)
        end = dt_end.isoformat()

        event = {
            'summary': summary,
            'description': 'Railway Bot tarafından eklendi',
            'start': {'dateTime': start, 'timeZone': 'UTC'},
            'end': {'dateTime': end, 'timeZone': 'UTC'},
            'reminders': {
                'useDefault': True
            },
        }
        
        event_result = service.events().insert(calendarId='primary', body=event).execute()
        return event_result.get('id')

    def clear_events_by_query(self, query: str, delete_limit: int = 50):
        """Belirli bir kelimeyi içeren tüm etkinlikleri takvimden sil (Toplu Temizlik)"""
        if not self.is_authenticated():
            return 0
        
        service = build('calendar', 'v3', credentials=self.creds)
        now = datetime.utcnow().isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary', timeMin=now,
            maxResults=delete_limit, q=query, singleEvents=True,
            orderBy='startTime').execute()
        events = events_result.get('items', [])
        
        deleted_count = 0
        for event in events:
            service.events().delete(calendarId='primary', eventId=event['id']).execute()
            deleted_count += 1
            
        return deleted_count
