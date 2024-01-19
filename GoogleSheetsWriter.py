from google_auth_oauthlib.flow import InstalledAppFlow
import gspread

class GoogleSheetsWriter:
    def __init__(self, spreadsheet_id, secrets):
        self.credentials_path = 'google_api_credentials.json'
        self.spreadsheet_id = spreadsheet_id

        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        
        flow = InstalledAppFlow.from_client_config(
            secrets,
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
    
        credentials = flow.run_local_server()
        self.client=gspread.authorize(credentials)
        

    def write_to_sheets(self, column_names, new_line_data):
        sheet = self.client.open_by_key(self.spreadsheet_id).sheet1
        sheet.append_row(new_line_data, table_range='A1:B1')
        
        # sheet_range = f"{self.sheet_name}!A:A"  # Adjust the range as per your needs

        # values = [new_line_data]

        # body = {
        #     'values': values
        # }

        # self.service.spreadsheets().values().append(
        #     spreadsheetId=self.spreadsheet_id,
        #     range=sheet_range,
        #     valueInputOption='RAW',
        #     body=body
        # ).execute()