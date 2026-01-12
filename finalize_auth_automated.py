from google_calendar_mgr import GoogleCalendarManager
import os

SCOPES = ['https://www.googleapis.com/auth/calendar']

def finalize():
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    creds_path = 'credentials.json'
    # Railway'deki path'i simüle edelim veya yerelde kaydedelim
    # Bot çalışınca /data/storage/token.json'a bakacak.
    # Ben yerelde oluşturup bota haber vereceğim.
    token_path = 'token.json'
    
    mgr = GoogleCalendarManager(creds_path, token_path)
    url = "http://localhost/?state=AJHNukueyrYAFhqxwVwfcQlGjzhEHU&code=4/0ASc3gC37hr7S_erRG6mozzmielcY4FUrC6itct8LvI7CwRP957XvHWBxnqV1i6K5R3vQZw&scope=https://www.googleapis.com/auth/calendar"
    
    try:
        mgr.finalize_auth(url)
        print("SUCCESS: token.json created.")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    finalize()
