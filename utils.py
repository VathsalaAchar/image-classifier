import requests
from pathlib import Path


def download_file(source_url: str, dl_file_path: Path):
    '''
    Downloads a file from the `source_url` and writes to `dl_file_path` 
    '''
    try:
        file_to_dl = requests.get(source_url)
    except Exception as e:
        print(f"Error {e} while downloading from {source_url}")
        return
    with open(dl_file_path, 'wb') as fp:
        fp.write(file_to_dl.content)

