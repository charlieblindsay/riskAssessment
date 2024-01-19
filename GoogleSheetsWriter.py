from google.oauth2 import service_account
import gspread
from googleapiclient.discovery import build

class GoogleSheetsWriter:
    def __init__(self, sheet_name):
        # self.credentials_path = 'google_api_credentials.json'
        # self.spreadsheet_id = spreadsheet_id



        # scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        # credentials = service_account.Credentials.from_service_account_file(
        #     secrets, scopes=scopes
        # )

        # self.client=gspread.authorize(credentials)

        self.credentials_path = 'google_api_credentials.json'
        self.spreadsheet_id = '1d7Tq7qEaNTrhm1E7qcGvl3Dkr8cFNdSpOul9RezjVs4'

        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=scopes
        )

        self.service = build('sheets', 'v4', credentials=credentials)
        self.sheet_name = sheet_name
        

    def write_to_sheets(self, new_line_data):
        # sheet = self.client.open_by_key(self.spreadsheet_id).sheet1
        # sheet.append_row(new_line_data, table_range='A1:B1')
        
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