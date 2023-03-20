import os
import requests
import shutil
import time

import pandas as pd

DATA_FP = '../../data'
MEDIUMS = ['oil', 'watercolor', 'pastel', 'pencil', 'tempera']


def create_dir(des_fp: str):
    """
    Iteratively creates a file path.
    """
    if des_fp is None:
        raise ValueError(f'Expecting a valid file path, got: {des_fp}')

    # Iteratively check if the file path exists
    split_path = des_fp.split('/')
    curr_path = ''
    for i in range(len(split_path)):
        curr_path += split_path[i] + '/'
        if not os.path.exists(curr_path):
            os.mkdir(curr_path)


def detect_medium(raw_medium: str):
    """
    Given a sentence describing the medium used, identify the medium and make it one word.
    """
    if str(raw_medium) == 'nan' or raw_medium is None:
        return False

    raw_medium = raw_medium.lower()

    for curr_med in MEDIUMS:
        if curr_med in raw_medium:
            return curr_med

    return False


def clean_string(dirty_str: str):
    """
    Formats a string to snakecase.
    """
    clean_str = dirty_str.lower().replace(',', '').replace(' ', '_')
    return clean_str


def handle_modern_art():
    """
    Structure the museum of modern art dataset.
    """
    csv = pd.read_csv(f'{DATA_FP}/external/Artworks.csv')
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/50.0.2661.102 Safari/537.36'}

    for i, row in csv.iterrows():
        medium = detect_medium(row['Medium'])

        if medium is not False:
            url = row['ThumbnailURL']
            if str(url) == 'nan':
                continue

            image_data = requests.get(url, headers=headers)
            time.sleep(0.1)
            if image_data.status_code == 200:
                create_dir(f'{DATA_FP}/processed/moma/{medium}')

                with open(f'{DATA_FP}/processed/moma/{medium}/{i}.jpg', 'wb') as file_obj:
                    file_obj.write(image_data.content)


def main():
    """
    Controls this script.
    """
    handle_modern_art()


if __name__ == '__main__':
    main()
