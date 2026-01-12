from google_auth_oauthlib.flow import InstalledAppFlow
import json

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_url():
    try:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        flow.redirect_uri = 'http://localhost'
        auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
        with open('auth.json', 'w') as f:
            json.dump({"url": auth_url}, f)
    except Exception as e:
        with open('auth.json', 'w') as f:
            json.dump({"error": str(e)}, f)

if __name__ == "__main__":
    get_url()
