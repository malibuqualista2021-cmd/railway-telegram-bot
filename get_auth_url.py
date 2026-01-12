from google_auth_oauthlib.flow import InstalledAppFlow
import sys

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_url():
    try:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        flow.redirect_uri = 'http://localhost'
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        print(f"AUTH_URL:{auth_url}")
    except Exception as e:
        print(f"ERROR:{e}")

if __name__ == "__main__":
    get_url()
