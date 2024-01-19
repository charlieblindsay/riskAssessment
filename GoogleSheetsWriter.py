import csv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime

class GoogleSheetsWriter:
    def __init__(self, sheet_name):
        self.credentials_path = 'google_api_credentials.json'
        self.spreadsheet_id = '1O6ztnca_NPC0TXxmSTrw5xV51vUow0I8nAebEsEntbQ'

        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=scopes
        )

        self.service = build('sheets', 'v4', credentials=credentials)
        self.sheet_name = sheet_name

    def write_to_sheets(self, new_line_data):
        sheet_range = f"{self.sheet_name}!A:A"  # Adjust the range as per your needs

        values = [new_line_data]

        body = {
            'values': values
        }

        self.service.spreadsheets().values().append(
            spreadsheetId=self.spreadsheet_id,
            range=sheet_range,
            valueInputOption='RAW',
            body=body
        ).execute()