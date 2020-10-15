# Data Fetching, unzip
import os
from zipfile import ZipFile
import urllib

Root_Download = './master/..'
url_download = Root_Download +'Data.zip'
Data_Path = os.path.join ('Data')

def fetch_Data (url = url_download, Path =Data_Path):
    os.makedirs(Path, exist_ok=True)
    zip_path = os.path.join (Path, 'Dataset.zip')
    filepath, _ = urllib.request.urlretrieve(url, zip_path)
    with ZipFile(filepath, 'r') as zip:
        zip.extractall(Path)
