import os
import pickle
import json
import pandas as pd
import numpy as np
import io
import datetime as dt
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# Google Drive APIのスコープ
SCOPES = ['https://www.googleapis.com/auth/drive']

tactical_action_name_list = ['longcounter', 'shortcounter', 'opposition_half_possession', 'own_half_possession', 'counterpressing', 'highpressing', 'middlepressing']
num_tactical_action = len(tactical_action_name_list)


def skillcorner_download(a, train, test_data):


    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    if train:
        tactical_action_sequence_np_folder_ID = '1FidT_CGBPbs63cZkm9Ckl3ej8CO3syxG'
        tactical_action_label_np_folder_ID = '1jYLOh_V7czR1BlaOCMeWBKJsHIeRJigr'

        # フォルダ内のファイルをリスト化
        tactical_action_sequence_np_files = list_files_in_folder(service, tactical_action_sequence_np_folder_ID, kind_of_file = 'np')
        if not tactical_action_sequence_np_files:
            print('No sequence_half_np files found in folder.')
            return
        
        tactical_action_label_np_files = list_files_in_folder(service, tactical_action_label_np_folder_ID, kind_of_file = 'np')
        if not tactical_action_label_np_files:
            print('No label_half_np files found in folder.')
            return

        tactical_action_name = tactical_action_name_list[a]

        tactical_action_sequence_np_file_name = tactical_action_name + '_sequence_np.npy'
        tactical_action_label_np_file_name = tactical_action_name + '_label_np.npy'

        sequence_np_file = find_same_match_file(tactical_action_sequence_np_files, tactical_action_sequence_np_file_name)
        label_np_file = find_same_match_file(tactical_action_label_np_files, tactical_action_label_np_file_name)

        # numpy 取得
        sequence_np = read_file(service, sequence_np_file['id'], kind_of_file = 'np')
        label_np = read_file(service, label_np_file['id'], kind_of_file = 'np')

        return sequence_np, label_np
    
    else:
        test_np_folder_ID = '1prcX_7Qd_EWZ-FkcYt-G9ELa9nv6qKIM'

        # フォルダ内のファイルをリスト化
        test_np_files = list_files_in_folder(service, test_np_folder_ID, kind_of_file = 'np')
        if not test_np_files:
            print('No sequence_half_np files found in folder.')
            return
        
        test_sequence_np_file_name = test_data + '_sequence.np'
        test_label_np_file_name = test_data + '_label.np'

        sequence_np_file = find_same_match_file(test_np_files, test_sequence_np_file_name)
        label_np_file = find_same_match_file(test_np_files, test_label_np_file_name)

        # numpy 取得
        sequence_np = read_file(service, sequence_np_file['id'], kind_of_file = 'np')
        label_np = read_file(service, label_np_file['id'], kind_of_file = 'np')

        return sequence_np, label_np


def authenticate():
    creds = None
    # トークンファイルが存在する場合、それを読み込む
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # トークンが存在しないか、無効な場合、ログインして新しいトークンを取得する
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # トークンを保存して、将来のために保存する
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds


def list_files_in_folder(service, folder_id, kind_of_file):
    if kind_of_file == 'json':
        query = f"'{folder_id}' in parents and mimeType='application/json'"
    elif kind_of_file == 'csv':
        query = f"'{folder_id}' in parents and mimeType='text/csv'"
    elif kind_of_file == 'pkl':
        query = f"'{folder_id}' in parents and name contains '.pkl'"
    elif kind_of_file == 'folder':
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    elif kind_of_file == 'np':
        query = f"'{folder_id}' in parents and (name contains '.npy' or name contains 'np')"
    page_token = None
    items = []
    while True:
        response = service.files().list(q=query, fields="nextPageToken, files(id, name)", pageToken=page_token).execute()
        items.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return items


def find_same_match_file(files, file_name): # files の中から欲しい file_name を探して file を返す

    for file in files:
        if file['name'] == file_name:
            return_file = file
            break

    return return_file


def read_file(service, file_id, kind_of_file):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    if kind_of_file == 'json':
        return json.load(fh)
    elif kind_of_file == 'pkl':
        return pickle.load(fh)
    elif kind_of_file == 'csv':
        return pd.read_csv(fh, low_memory=False)
    elif kind_of_file == 'np':
        return np.load(fh, allow_pickle=True)